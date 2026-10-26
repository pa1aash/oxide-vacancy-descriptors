"""Host classes by cation d-shell filling, fixed before any gain is computed.

Physics: the energy to remove a neutral O atom from an oxide is set largely by where the two electrons it leaves
behind can go. In a d0 host (Ti4+, Nb5+, W6+) they fall into the empty cation d band; in a d10 host (Zn2+, Cd2+) the
d shell is full and they occupy the cation s-like conduction band edge; in a partially filled d host they enter an
open shell. A main-group host has no d-block cation at all. This script only assigns each host to one of these
classes from its formula; nothing here reads a model, a prediction or an error.

Rule (st1b classes spec, implemented exactly):
    1. oxidation states: the FIRST (most probable) guess of ``Composition(reduced formula).oxi_state_guesses()``,
       pymatgen defaults. Unassigned if there is no guess, O is not exactly -2, a cation state is not an integer,
       or a d-block cation has a d count outside 0..10 (the reason string is recorded).
    2. blocks by atomic number: d = 21-30, 39-48, 72-80; f = 57-71, 89-103; any other cation is main-group.
    3. d count of a d-block cation = group - oxidation state.
    4. class: any f-block cation -> mixed_f; else no d-block cation -> main_group; else all d0 -> d0; all d10 ->
       d10; all d1-d9 -> partial_d; else (d-block cations of different classes) -> mixed_f.
    5. descriptive flag post_transition_d10_core: a main-group cation with Z >= 31 in groups 13-18 (filled
       (n-1)d10 core: Ga, Ge, As, In, Sn, Sb, Te, Tl, Pb, Bi, ...). Not a class.
A cation is an element with a positive state in the guess; other elements with a negative state are anions.

Counts: hosts per class over the whole universe; over hosts tested in at least one of the 30 outer resamples at
654 training hosts (test lists of splits/outer_r0-9.json and splits/resamples_10_29.json, read through
``dftgnn.split.load_split``; membership only); over hosts with >= 2 vacancy sites in the universe.

Writes through ``write_result`` into ``results/derived_dshell/dshell_classes.json``.

    PYTHONPATH=$PWD/src python analysis/dshell/classes.py [--data-root ~/Desktop/dft-gnn] [--dry-run [PATH]]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "derived_dshell"
UNIVERSE = "data/processed/universe_v1.parquet"
N_RESAMPLES = 30
N_TEST_HOSTS = 164
TOL = 1e-9

D_BLOCK = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(72, 81))
F_BLOCK = frozenset(range(57, 72)) | frozenset(range(89, 104))
CLASSES = ("d0", "d10", "partial_d", "main_group", "mixed_f", "unassigned")

RULE = (
    "Oxidation states: first (most probable) guess of pymatgen Composition(reduced formula).oxi_state_guesses() "
    "with defaults; unassigned if no guess, O not exactly -2, any cation state non-integer, or any d-block cation "
    "with d count outside 0..10. Cation = element with positive state. d-block = Z 21-30, 39-48, 72-80; f-block = "
    "Z 57-71, 89-103; other cations main-group. d count = group - oxidation state. Class: any f-block cation -> "
    "mixed_f; else no d-block cation -> main_group; else all d-block cations d0 -> d0; all d10 -> d10; all d1-d9 -> "
    "partial_d; else -> mixed_f. Flag post_transition_d10_core (not a class): any main-group cation with Z >= 31 in "
    "groups 13-18.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block(z: int) -> str:
    return "d" if z in D_BLOCK else "f" if z in F_BLOCK else "main"


def classify_states(states: Mapping[str, float] | None) -> dict:
    """Class of one host from its oxidation-state guess {element: state} (None = no guess)."""
    from pymatgen.core import Element

    out = {"class": "unassigned", "cation_states": {}, "d_counts": {}, "post_transition_d10_core": False,
           "reason": None}
    if not states:
        out["reason"] = "no oxidation-state guess"
        return out
    out["cation_states"] = {el: float(s) for el, s in sorted(states.items()) if s > 0}
    if "O" not in states or abs(states["O"] + 2) > TOL:
        out["reason"] = f"O state {states.get('O')} is not -2"
        return out
    bad = sorted(el for el, s in states.items() if s > 0 and abs(s - round(s)) > TOL)
    if bad:
        out["reason"] = "non-integer cation state: " + ", ".join(f"{el}{states[el]:+.4g}" for el in bad)
        return out
    cations = {el: round(s) for el, s in states.items() if s > 0}
    out["cation_states"] = dict(sorted(cations.items()))
    blocks = {el: block(Element(el).Z) for el in cations}
    d_counts = {el: Element(el).group - s for el, s in sorted(cations.items()) if blocks[el] == "d"}
    out["d_counts"] = d_counts
    out["post_transition_d10_core"] = any(
        blocks[el] == "main" and Element(el).Z >= 31 and 13 <= Element(el).group <= 18 for el in cations)
    off = sorted(el for el, n in d_counts.items() if not 0 <= n <= 10)
    if off:
        out["reason"] = "d count outside 0..10: " + ", ".join(f"{el} d{d_counts[el]}" for el in off)
        return out
    if any(b == "f" for b in blocks.values()):
        out["class"] = "mixed_f"
    elif not d_counts:
        out["class"] = "main_group"
    elif all(n == 0 for n in d_counts.values()):
        out["class"] = "d0"
    elif all(n == 10 for n in d_counts.values()):
        out["class"] = "d10"
    elif all(1 <= n <= 9 for n in d_counts.values()):
        out["class"] = "partial_d"
    else:
        out["class"] = "mixed_f"
    return out


def first_guess(formula: str) -> tuple[str, dict | None]:
    """Reduced formula and the first oxidation-state guess (pymatgen defaults)."""
    from pymatgen.core import Composition

    red = Composition(formula).reduced_formula
    g = Composition(red).oxi_state_guesses()
    return red, (dict(g[0]) if g else None)


def classify_host(host_id: str, formula: str) -> dict:
    red, states = first_guess(formula)
    return {"host_id": host_id, "formula": formula, "reduced_formula": red, **classify_states(states)}


def tested_hosts() -> set[str]:
    """Hosts in the test list of at least one of the 30 outer resamples (654 training hosts)."""
    from dftgnn.split import load_split

    out: set[str] = set()
    for r in range(N_RESAMPLES):
        sp = load_split(f"outer_r{r}")
        if len(sp["test"]) != N_TEST_HOSTS or len(sp["budget_order"]) != 654:
            raise ValueError(f"outer_r{r}: unexpected test or pool size")
        out.update(sp["test"])
    return out


def count(rows: Sequence[dict], key: str | None = None) -> dict:
    sel = [h for h in rows if key is None or h[key]]
    n = {k: sum(h["class"] == k for h in sel) for k in CLASSES}
    flag = {k: sum(h["class"] == k and h["post_transition_d10_core"] for h in sel) for k in CLASSES}
    return {"n_hosts": n, "total": len(sel), "post_transition_d10_core": flag}


def run(data_root: Path) -> dict:
    import pandas as pd
    import pymatgen.core

    upath = data_root / UNIVERSE
    u = pd.read_parquet(upath, columns=["site_id", "host_id", "formula"])
    if u.groupby("host_id").formula.nunique().max() != 1:
        raise ValueError("a host carries more than one formula")
    hosts = u.groupby("host_id", sort=True).agg(formula=("formula", "first"), n_sites=("site_id", "size"))
    tested = tested_hosts()
    if not tested <= set(hosts.index):
        raise ValueError("test hosts missing from the universe table")
    rows = []
    for hid, h in hosts.iterrows():
        rec = classify_host(str(hid), str(h.formula))
        rec["n_sites"] = int(h.n_sites)
        rec["tested_654"] = hid in tested
        rows.append(rec)
    rows.sort(key=lambda r: r["host_id"])
    return {
        "label": "ST1b host classes by cation d-shell filling (step 1; no model output read)",
        "rule": RULE,
        "classes": list(CLASSES),
        "hosts": rows,
        "counts": {"all": count(rows), "tested_654_any_of_30": count(rows, "tested_654"),
                   "ge2_sites": count([r for r in rows if r["n_sites"] >= 2])},
        "unassigned": [{k: r[k] for k in ("host_id", "formula", "reason")} for r in rows
                       if r["class"] == "unassigned"],
        "n_hosts": len(rows), "n_tested_654": len(tested),
        "pymatgen_version": pymatgen.core.__version__,
        "inputs": {"universe": {"path": UNIVERSE, "sha256": sha256(upath)},
                   "splits": "splits/outer_r0-9.json, splits/resamples_10_29.json (test lists, r = 0-29)"},
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None,
                    help="checkout holding data/processed/universe_v1.parquet (default: this repo)")
    ap.add_argument("--dry-run", nargs="?", const="-", default=None, metavar="PATH",
                    help="write the payload JSON to PATH (stdout if omitted) instead of calling write_result")
    a = ap.parse_args(argv)
    payload = run((a.data_root or ROOT).expanduser().resolve())
    if a.dry_run is not None:
        text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if a.dry_run == "-":
            print(text, end="")
        else:
            out = Path(a.dry_run).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(out)
        return
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    print(write_result("dshell_classes", payload, config=load_config(), repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
