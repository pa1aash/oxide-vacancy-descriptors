"""Build the modelling universe: one row per neutral oxygen vacancy site.

Source: Kumagai et al. (PRM 5, 123803, 2021) release, neutral ML subset
(``vacancy_formation_energy_ml/charge0.csv``), filtered by the decisions D1-D3 named in
``config.data.filters``.

Identifiers
-----------
site_id  ``<formula>_Va_<label>_0``, the release's own defect key for the neutral entry
         (e.g. ``ZnO_Va_O1_0``). ``<label>`` is the irreducible-O label shared by
         ``charge0.csv``, ``cell_info.txt`` and ``bulk_visual_data.json``.
host_id  Materials Project ID of the host, read from the per-host ``bulk_visual_data.json``.

Row hash
--------
``row_sha256`` is the sha256 of the UTF-8 string
``target_Ef_eV=<hex>;desc_<a>=<hex>;...`` over the target and the 70 ``desc_*`` columns in
sorted column order, each value written with ``float.hex`` (exact, platform independent).

Supercell paths --------------- Every path is resolved from the per-host JSON (its ``formula`` field names the
``site_info/<formula>/`` directory); nothing is globbed. Prefix globs are unsafe in this release: 14 host
directories hold files of a longer formula.

Defect magnetic moment
----------------------
The release stores no magnetisation field. The final total moment of the neutral defect
calculation is derived from ``defect_visual_data.json -> defect_details[<site>_0][1]``
(pydefect ``BandEdgeOrbitalInfos`` of the final calculation): two entries in
``orbital_infos`` mean a spin-polarised run, and
``m = sum_k w_k sum_b (occ_up[k][b] - occ_down[k][b])`` over the stored band window.
This equals the cell moment when every band below the window is filled in both channels;
``window_floor_min_occupation`` records the lowest-band occupation as a check.
Non-spin-polarised entries have ``m = 0`` by construction.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dftgnn.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_SUBDIR = "unpacked/oxygen_vacancies_db"
K25_SUBDIR = "unpacked/ML_charged_defects/oxy_vac_data"
TAXONOMY = REPO_ROOT / "docs" / "descriptor_taxonomy.csv"
IDS_FILE = REPO_ROOT / "data" / "universe_v1_ids.csv"

DYN_UNSTABLE = ("K2Cd2O3", "Ba2Ti(GeO4)2", "Cs2O")  # K21 Sec. III A
D1_FLAGS = ("vacancy_split", "unknown", "not same config from init")
D2_HOSTS = ("Sr2Zr7O16",)
D3_HOSTS = ("Ba4Nb2WO12", "Sr2SnO4")
D3_TOL_A = 0.02
KNOWN_FILTERS = (
    "D1_defect_type_flags",
    "D2_Sr2Zr7O16_label_mismatch",
    "D3_cell_metadata_shell_check",
)
ID_COLUMNS = ["site_id", "host_id", "formula", "site_label", "vacancy_atom_index", "row_sha256"]


# ----------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class ReleasePaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "oxygen_vacancies_db_data"

    @property
    def site_info(self) -> Path:
        return self.root / "site_info"

    @property
    def ml_csv(self) -> Path:
        return self.root / "vacancy_formation_energy_ml" / "charge0.csv"

    def host_json(self, formula: str, which: str) -> Path:
        return self.data / formula / f"{which}_visual_data.json"


def release_paths(cfg: Config) -> ReleasePaths:
    raw = Path(cfg.data.raw_dir)
    raw = raw if raw.is_absolute() else REPO_ROOT / raw
    return ReleasePaths(raw / DB_SUBDIR)


def resolve_supercell_cif(paths: ReleasePaths, formula: str, bulk: dict | None = None) -> Path:
    """Supercell CIF of ``formula``, resolved through the per-host JSON. Never globs."""
    bulk = bulk if bulk is not None else json.loads(paths.host_json(formula, "bulk").read_text())
    if bulk["formula"] != formula:
        raise ValueError(f"per-host JSON formula {bulk['formula']!r} != directory {formula!r}")
    cif = paths.site_info / bulk["formula"] / "supercell.cif"
    if not cif.is_file():
        raise FileNotFoundError(cif)
    return cif


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- parsing


def parse_cell_info(txt: str) -> dict[str, dict]:
    """Irreducible sites of ``cell_info.txt`` keyed by label (O1, Zn1, ...)."""
    out = {}
    for blk in txt.split("Irreducible element: ")[1:]:
        label = blk.split("\n", 1)[0].strip()
        get = lambda key, b=blk: re.search(key + r": (.*)", b).group(1).strip()
        idx = [i for seg in get("Equivalent atoms").split()
               for i in range(int(seg.split("..")[0]), int(seg.split("..")[-1]) + 1)]
        out[label] = {
            "equiv": idx,
            "frac": [float(x) for x in get("Fractional coordinates").split()],
            "cutoff_A": float(get("Cutoff radius")),
            "coordination": get("Coordination"),
        }
    return out


def defect_moment(detail: list) -> tuple[bool, float, float]:
    """(spin_polarised, moment_muB, lowest-band minimum occupation) of one defect entry."""
    info = next(x for x in detail if isinstance(x, dict) and "orbital_infos" in x)
    w = np.asarray(info["kpt_weights"], dtype=float)
    occ = [np.array([[b["occupation"] for b in kb] for kb in ch], dtype=float)
           for ch in info["orbital_infos"]]
    floor = float(min(o[:, 0].min() for o in occ))
    if len(occ) == 1:
        return False, 0.0, floor
    if len(occ) != 2:
        raise ValueError(f"unexpected number of spin channels: {len(occ)}")
    return True, float(w @ (occ[0] - occ[1]).sum(axis=1)), floor


def _host_record(args: tuple[Path, str]) -> dict:
    root, formula = args
    paths = ReleasePaths(root)
    d = json.loads(paths.host_json(formula, "defect").read_text())
    b = json.loads(paths.host_json(formula, "bulk").read_text())
    unusual = d["unusual_defects"]
    rows = []
    for site, de in d["defect_energy_summary"]["defect_energies"].items():
        for q, e in zip(de["charges"], de["defect_energies"], strict=True):
            if q != 0:
                continue
            key = f"{site}_0"
            sp, m, floor = defect_moment(d["defect_details"][key])
            rows.append({
                "formula": formula, "site_label": site.removeprefix("Va_"),
                "release_Ef_eV": e["formation_energy"], "is_shallow": e["is_shallow"],
                "release_flags": "|".join(unusual.get(key, [])),
                "spin_polarised": sp, "defect_moment_muB": m, "window_floor_min_occupation": floor,
            })
    cif = resolve_supercell_cif(paths, formula, b)
    return {"formula": formula, "mp_id": b["mp_id"], "cif": cif, "rows": rows}


def release_neutral(cfg: Config, processes: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, hosts): every completed neutral entry in the release, and per-host metadata.

    Host directories are listed from ``oxygen_vacancies_db_data`` by exact name (directory
    entries, one per host archive), not by a formula pattern.
    """
    paths = release_paths(cfg)
    formulas = sorted(p.name for p in paths.data.iterdir() if p.is_dir())
    with Pool(processes) as pool:
        recs = pool.map(_host_record, [(paths.root, f) for f in formulas], chunksize=8)
    entries = pd.DataFrame([r for x in recs for r in x["rows"]])
    hosts = pd.DataFrame([{"formula": x["formula"], "host_id": x["mp_id"],
                           "supercell_cif": x["cif"]} for x in recs])
    entries["full_name"] = entries.formula + "_Va_" + entries.site_label
    return entries, hosts


# ----------------------------------------------------------------------------- D3


def shell_check(paths: ReleasePaths, formula: str, label: str, tol: float = D3_TOL_A) -> dict:
    """Compare the first cation shell of a supercell O site with the unit-cell site.

    The shell is every non-O neighbour within the release's own cutoff radius for that
    label (``cell_info.txt``). Pass requires: supercell atom is O; the same cation species
    and counts; sorted per-species distances equal within ``tol``; and the same for every
    equivalent supercell atom of the label.
    """
    from pymatgen.core import Structure

    bulk = json.loads(paths.host_json(formula, "bulk").read_text())
    cif = resolve_supercell_cif(paths, formula, bulk)
    sc = Structure.from_file(cif)
    uc = Structure.from_dict(bulk["structure_graph"]["structure"])
    ci = parse_cell_info((paths.site_info / formula / "cell_info.txt").read_text())[label]
    r = ci["cutoff_A"]

    def shell(st, i: int) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for n in st.get_neighbors(st[i], r):
            if n.specie.symbol != "O":
                out.setdefault(n.specie.symbol, []).append(float(n.nn_distance))
        return {k: sorted(v) for k, v in sorted(out.items())}

    def same(a: dict, b: dict) -> tuple[bool, float]:
        if {k: len(v) for k, v in a.items()} != {k: len(v) for k, v in b.items()}:
            return False, float("inf")
        dev = max((abs(x - y) for k in a for x, y in zip(a[k], b[k], strict=True)), default=0.0)
        return dev <= tol, dev

    uc_idx = bulk["sites"][label]["equivalent_atoms"][0]
    sc_idx = ci["equiv"][0]
    uc_shell, sc_shell = shell(uc, uc_idx), shell(sc, sc_idx)
    ok_first, dev_first = same(sc_shell, uc_shell)
    devs = [same(shell(sc, i), uc_shell) for i in ci["equiv"]]
    is_o = sc[sc_idx].specie.symbol == "O" and uc[uc_idx].specie.symbol == "O"
    all_o = all(sc[i].specie.symbol == "O" for i in ci["equiv"])
    passed = bool(is_o and all_o and ok_first and all(ok for ok, _ in devs))
    return {
        "formula": formula, "site_label": label, "supercell_cif": _rel(cif),
        "supercell_atom_index": sc_idx, "supercell_atom_element": sc[sc_idx].specie.symbol,
        "unit_cell_atom_index": uc_idx, "unit_cell_atom_element": uc[uc_idx].specie.symbol,
        "cutoff_A": r, "cell_info_coordination": ci["coordination"],
        "unit_cell_shell_A": {k: [round(x, 4) for x in v] for k, v in uc_shell.items()},
        "supercell_shell_A": {k: [round(x, 4) for x in v] for k, v in sc_shell.items()},
        "max_abs_distance_diff_A": dev_first,
        "n_equivalent_supercell_atoms": len(ci["equiv"]),
        "all_equivalent_atoms_O": all_o,
        "all_equivalent_shells_match": all(ok for ok, _ in devs),
        "max_abs_distance_diff_over_equivalents_A": max(dv for _, dv in devs),
        "tolerance_A": tol, "pass": passed,
    }


# ----------------------------------------------------------------------------- helpers


def descriptor_classes() -> dict[str, str]:
    """``desc_<name>`` -> class from docs/descriptor_taxonomy.csv."""
    t = pd.read_csv(TAXONOMY)
    return {f"desc_{n}": c for n, c in zip(t["name"], t["class"], strict=True)}


def kiyohara_split(cfg: Config) -> dict[str, str]:
    """formula -> train/val/test from Kiyohara et al.'s released lists without PHS."""
    raw = Path(cfg.data.raw_dir)
    raw = raw if raw.is_absolute() else REPO_ROOT / raw
    out: dict[str, str] = {}
    for sp in ("train", "val", "test"):
        for f in (raw / K25_SUBDIR / f"{sp}_wo_PHS.txt").read_text().splitlines():
            if f.strip():
                if f.strip() in out:
                    raise ValueError(f"{f} in two K25 splits")
                out[f.strip()] = sp
    return out


def row_sha256(row: pd.Series, desc_cols: list[str]) -> str:
    parts = [f"target_Ef_eV={float(row['target_Ef_eV']).hex()}"]
    parts += [f"{c}={float(row[c]).hex()}" for c in sorted(desc_cols)]
    return hashlib.sha256(";".join(parts).encode()).hexdigest()


def ids_table(universe: pd.DataFrame) -> pd.DataFrame:
    return universe[ID_COLUMNS].sort_values("site_id").reset_index(drop=True)


def ids_csv_bytes(universe: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    ids_table(universe).to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode()


# ----------------------------------------------------------------------------- cascade


@dataclass
class Cascade:
    steps: list[dict] = field(default_factory=list)
    d3: dict = field(default_factory=dict)

    def add(self, step: str, criterion: str, df: pd.DataFrame, removed: pd.DataFrame,
            reasons: pd.Series | None = None) -> None:
        from pymatgen.core import Composition

        red = df.formula.map(lambda f: Composition(f).reduced_formula)
        rem = []
        for i, r in removed.iterrows():
            rem.append({"site_id": f"{r.formula}_Va_{r.site_label}_0", "formula": r.formula,
                        "reason": reasons.loc[i] if reasons is not None else criterion})
        self.steps.append({
            "step": step, "criterion": criterion, "sites": len(df),
            "hosts": int(df.formula.nunique()), "formulas": int(red.nunique()),
            "n_removed": len(removed), "removed": sorted(rem, key=lambda x: x["site_id"]),
        })


def _release_to_ml_reason(r: pd.Series) -> str:
    why = []
    if r.is_shallow is True:
        why.append("PHS")
    if r.is_shallow is None or (isinstance(r.is_shallow, float) and np.isnan(r.is_shallow)):
        why.append("band edges undetermined")
    if r.formula in DYN_UNSTABLE:
        why.append("dynamically unstable host")
    if r.release_flags:
        why.append("flags: " + r.release_flags)
    return "; ".join(why) if why else "no release flag (not in K21 ML subset; criterion unstated)"


def build_universe(config: Config | None = None, *, processes: int | None = None,
                   return_cascade: bool = False):
    """One row per neutral O-vacancy site of the filtered universe (see module docstring)."""
    cfg = config if config is not None else load_config()
    unknown = set(cfg.data.filters) - set(KNOWN_FILTERS)
    if unknown:
        raise ValueError(f"unknown filters in config: {sorted(unknown)}")
    paths = release_paths(cfg)
    entries, hosts = release_neutral(cfg, processes)
    hosts = hosts.set_index("formula")
    cas = Cascade()
    cas.add("release", "Completed neutral entries in the release", entries, entries.iloc[:0])

    ml = pd.read_csv(paths.ml_csv, index_col=0)
    if not ml.full_name.isin(entries.full_name).all():
        raise ValueError("ML subset contains entries absent from the release")
    in_ml = entries.full_name.isin(ml.full_name)
    gone = entries[~in_ml]
    cas.add("ml_subset", "Kumagai neutral ML subset (charge0.csv)", entries[in_ml], gone,
            gone.apply(_release_to_ml_reason, axis=1))
    cur = entries[in_ml].copy()

    if "D1_defect_type_flags" in cfg.data.filters:
        fl = cur.release_flags.str.split("|")
        hit = fl.map(lambda x: any(f in x for f in D1_FLAGS))
        cas.add("D1", "Vacancy split, unknown type or not same config from init", cur[~hit],
                cur[hit], "D1 flags: " + cur.release_flags[hit])
        cur = cur[~hit]
    if "D2_Sr2Zr7O16_label_mismatch" in cfg.data.filters:
        hit = cur.formula.isin(D2_HOSTS)
        cas.add("D2", "Host Sr2Zr7O16 (ML site labels do not match release targets)", cur[~hit],
                cur[hit], pd.Series("D2 data-integrity exclusion: ML-table site labels do not "
                                    "match release targets", index=cur.index[hit]))
        cur = cur[~hit]
    if "D3_cell_metadata_shell_check" in cfg.data.filters:
        fail_hosts = []
        for h in D3_HOSTS:
            labels = sorted(cur.site_label[cur.formula == h])
            if not labels:
                cas.d3[h] = {"in_universe": False, "outcome": "not applicable (host not in universe)"}
                continue
            checks = [shell_check(paths, h, lab) for lab in labels]
            ok = all(c["pass"] for c in checks)
            cas.d3[h] = {"in_universe": True, "outcome": "pass, kept" if ok else "fail, excluded",
                         "sites": checks}
            if not ok:
                fail_hosts.append(h)
        hit = cur.formula.isin(fail_hosts)
        cas.add("D3", "Ba4Nb2WO12, Sr2SnO4: vacancy site is O with matching first cation shell",
                cur[~hit], cur[hit], pd.Series("D3 data-integrity exclusion: shell check failed",
                                               index=cur.index[hit]))
        cur = cur[~hit]

    # assemble rows
    ml = ml.set_index("full_name")
    desc_names = [c for c in ml.columns if c not in ("formula", "vacancy_formation_energy")]
    classes = descriptor_classes()
    if sorted(f"desc_{c}" for c in desc_names) != sorted(classes):
        raise ValueError("ML descriptor columns do not match docs/descriptor_taxonomy.csv")
    m = ml.loc[cur.full_name]
    target = m.vacancy_formation_energy.to_numpy()
    dev = np.abs(target - cur.release_Ef_eV.to_numpy())
    if dev.max() > 1e-9:
        raise ValueError(f"ML targets disagree with release (max {dev.max():.3g} eV)")

    from pymatgen.core import Composition

    split = kiyohara_split(cfg)
    cells: dict[str, dict] = {}
    natoms: dict[str, int] = {}
    rows = []
    for (_, r), (_, mr) in zip(cur.iterrows(), m.iterrows(), strict=True):
        f = r.formula
        if f not in cells:
            cif = Path(hosts.loc[f, "supercell_cif"])
            cells[f] = parse_cell_info((cif.parent / "cell_info.txt").read_text())
            natoms[f] = _cif_natoms(cif)
        rows.append({
            "site_id": f"{f}_Va_{r.site_label}_0", "host_id": hosts.loc[f, "host_id"],
            "formula": f, "reduced_formula": Composition(f).reduced_formula,
            "site_label": r.site_label,
            "supercell_cif_path": _rel(Path(hosts.loc[f, "supercell_cif"])),
            "vacancy_atom_index": cells[f][r.site_label]["equiv"][0],
            "n_atoms": natoms[f],
            "target_Ef_eV": float(mr.vacancy_formation_energy),
            **{f"desc_{c}": float(mr[c]) for c in desc_names},
            "spin_polarised": bool(r.spin_polarised),
            "defect_moment_muB": float(r.defect_moment_muB),
            "window_floor_min_occupation": float(r.window_floor_min_occupation),
            "kiyohara_split": split.get(f, "absent"),
        })
    uni = pd.DataFrame(rows)
    desc_cols = [c for c in uni.columns if c.startswith("desc_")]
    uni["row_sha256"] = uni.apply(row_sha256, axis=1, desc_cols=desc_cols)
    uni = uni.sort_values("site_id").reset_index(drop=True)
    uni.attrs["descriptor_class"] = classes
    cas.add("final", "Final universe", uni, uni.iloc[:0])
    if not uni.site_id.is_unique:
        raise ValueError("duplicate site_id")
    return (uni, cas) if return_cascade else uni


def _cif_natoms(cif: Path) -> int:
    """Atom count of a release CIF (one atom per ``_atom_site`` loop row)."""
    lines = cif.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith("_atom_site_"))
    while lines[i].strip().startswith("_atom_site_"):
        i += 1
    n = 0
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("loop_", "_")):
        n += 1
        i += 1
    return n


def write_universe(universe: pd.DataFrame, cfg: Config | None = None) -> dict[str, Path]:
    """Persist the parquet (gitignored), the tracked IDs CSV and its sha256 file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = cfg if cfg is not None else load_config()
    proc = Path(cfg.data.processed_dir)
    proc = proc if proc.is_absolute() else REPO_ROOT / proc
    proc.mkdir(parents=True, exist_ok=True)
    pq_path = proc / f"{cfg.data.universe}.parquet"
    tbl = pa.Table.from_pandas(universe.drop(columns=[]), preserve_index=False)
    meta = dict(tbl.schema.metadata or {})
    meta[b"descriptor_class"] = json.dumps(universe.attrs["descriptor_class"], sort_keys=True).encode()
    pq.write_table(tbl.replace_schema_metadata(meta), pq_path)
    ids = ids_csv_bytes(universe)
    IDS_FILE.write_bytes(ids)
    sha = hashlib.sha256(ids).hexdigest()
    sha_path = IDS_FILE.with_name(IDS_FILE.name + ".sha256")
    sha_path.write_text(f"{sha}  {IDS_FILE.name}\n")
    return {"parquet": pq_path, "ids": IDS_FILE, "ids_sha256": sha_path}


def read_universe(cfg: Config | None = None) -> pd.DataFrame:
    """Load the persisted universe with its descriptor-class mapping in ``attrs``."""
    import pyarrow.parquet as pq

    cfg = cfg if cfg is not None else load_config()
    proc = Path(cfg.data.processed_dir)
    proc = proc if proc.is_absolute() else REPO_ROOT / proc
    tbl = pq.read_table(proc / f"{cfg.data.universe}.parquet")
    df = tbl.to_pandas()
    df.attrs["descriptor_class"] = json.loads(tbl.schema.metadata[b"descriptor_class"])
    return df
