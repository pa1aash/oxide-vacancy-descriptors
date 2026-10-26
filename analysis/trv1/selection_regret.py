"""Energy cost of a wrong site choice: regret of the lowest-predicted oxygen site, and the release sites that the
perturbed-host-state filter removed before any site could be chosen.

Nothing is trained. Physically: reduction empties the oxygen site of lowest vacancy formation energy first, and the
equilibrium population of a site type relative to the lowest scales as exp(-dE / kB T) (multiplicity aside). A
screening model that picks the wrong site therefore errs by how far the chosen site lies above the true minimum, not
by the fact of the miss. The regret of a host is E_f(DFT) of the lowest-predicted site minus the host's DFT minimum
(zero on a hit). The selection is defined over the retained localised sites of the release: sites flagged as
perturbed host states (PHS) in PBEsol(+U) never enter the universe, so a host whose PHS sites were removed is ranked
over its survivors only.

Estimator (headline design of analysis/coord/site_selection.py, whose loaders and host tables are reused): 654
training hosts, 30 resamples x 5 seeds, S and D from precision_eval_long.parquet (v2), D* from results/dretrain_v2;
seed-ensemble mean prediction per (model, r, site); ties in the prediction take the first site in site_id order (none
occur). Per resample a host statistic is averaged over eligible hosts (>= 2 or >= 3 tested sites); point = mean over
the 30 resamples; bootstrap = HierDraws.make([164] * 30, 2000, 20261008) with hosts sorted by host_id, replicate =
mean over slots of sum(w x) / sum(w eligible); 95% = 2.5 / 97.5 percentiles, upper_one_sided_95 = 95th percentile;
differences D* - S and D - S per replicate on the same draws (paired).

Blocks (each for hosts with >= 2 and >= 3 tested sites)
    regret               mean regret (eV); median and 90th percentile of regret over host appearances (all 30
                         resamples pooled; replicate = weighted inverted-CDF quantile with the draw weights); mean
                         regret conditional on a miss (sum regret / number of misses, per resample).
    hit_within_delta     fraction of hosts with regret <= delta = 0.05 eV (configs/config.yaml stats/delta_eV).
    separated_hosts      exact selection rate on hosts whose two lowest DFT sites differ by more than delta, the
                         number of such hosts, and the chance rate there (1 / n_sites; no tied minima occur).
    phs_truncation       per host, the number of release neutral entries removed by the PHS flag, counted directly
                         from the raw release (defect_visual_data.json, is_shallow of every charge-0 entry) and checked
                         against results/filter_cascade.json and the label gaps of data/universe_v1_ids.csv (a gap in
                         O1..On marks a removed site; a removed last site leaves none, so gaps are a lower bound).
                         Hosts with >= 1 PHS site removed ("partially filtered"), by published cation class
                         (results/derived_dshell/dshell_classes.json); per class, the fraction of release neutral
                         entries removed by the flag (release hosts outside the universe classified by the same rule,
                         analysis/dshell/classes.classify_host); selection rates, mean regret and D* - S, D - S
                         recomputed with the partially filtered hosts made ineligible.
    concentration_factor exp(mean regret / kB T) at 1000 K for S, D, D* (kB in eV/K from scipy.constants); the
                         interval maps the regret interval (monotone). Also the host mean of exp(-regret / kB T), the
                         population of the chosen site type relative to the lowest, averaged as the regret.

Reproduction: every value of results/derived_coord/site_selection.json payload.results (selection rates, chance,
D* - S, D - S with their intervals, per-resample values and counts) and its within-host gap check are recomputed
here (tolerance 1e-9). Writes through write_result into results/derived_trv1_selection/selection_regret.json.

    PYTHONPATH=$PWD/src python analysis/trv1/selection_regret.py [--dry-run --out FILE] [--release-root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis.coord import site_selection as ss

NAME = "selection_regret"
OUT_DIR = ROOT / "results" / "derived_trv1_selection"
RELEASE_SUB = Path("data/raw/unpacked/oxygen_vacancies_db/oxygen_vacancies_db_data")
RELEASE_CANDIDATES = (ROOT, Path.home() / "Desktop" / "dft-gnn")
MODELS = ss.MODELS
GAPS = ss.GAPS
MIN_SITES = ss.MIN_SITES
N_R, N_HOSTS = ss.N_R, ss.N_HOSTS
TOL = 1e-9
T_K = 1000.0
QUANTS = (("median", 0.5), ("p90", 0.9))
LABEL = ("site-selection regret: DFT energy of the lowest-predicted oxygen site above the host minimum, hit within "
         "delta, selection on separated hosts, PHS truncation of hosts; S, D, D* at 654 hosts, 30 x 5")


# ----------------------------------------------------------------------------- per-host tables

def host_table(frame: pd.DataFrame, delta: float) -> pd.DataFrame:
    """ss.host_table plus regret, gap between the two lowest DFT sites and hit within delta (sorted by host_id)."""
    t = ss.host_table(frame)
    reg, gap2 = [], []
    for _, g in frame.sort_values("site_id").groupby("host_id", sort=True):
        y, p = g.y_true.to_numpy(float), g.y_pred.to_numpy(float)
        reg.append(float(y[int(np.argmin(p))] - y.min()))
        ys = np.sort(y)
        gap2.append(float(ys[1] - ys[0]) if len(ys) > 1 else np.nan)
    t["regret"] = reg
    t["gap2"] = gap2
    t["within_delta"] = (t.regret <= delta).astype(int)
    if ((t.regret == 0) != (t.hit == 1)).any():
        raise AssertionError("regret zero and hit disagree")
    return t


def columns(t: pd.DataFrame, k: int, delta: float, drop: set[str] | None = None) -> dict[str, np.ndarray]:
    """Per-host numerators (already multiplied by eligibility) and the eligibility denominators."""
    e = (t.n_sites >= k).to_numpy(float)
    if drop:
        e = e * (~t.host_id.isin(drop)).to_numpy(float)
    sep = e * (t.gap2.fillna(-1.0).to_numpy() > delta)
    hit = t.hit.to_numpy(float)
    return {"elig": e, "hit": hit * e, "chance": (t.n_minima / t.n_sites).to_numpy(float) * e,
            "regret": t.regret.to_numpy(float) * e, "within": t.within_delta.to_numpy(float) * e,
            "miss": (1 - hit) * e, "sep": sep, "sep_hit": hit * sep,
            "sep_chance": (1.0 / t.n_sites).to_numpy(float) * sep,
            "pop": np.exp(-t.regret.to_numpy(float) / kbt()) * e}


def kbt() -> float:
    from scipy.constants import physical_constants

    return physical_constants["Boltzmann constant in eV/K"][0] * T_K


# ----------------------------------------------------------------------------- estimators

def ratio(cols: list[dict], num: str, den: str, draws) -> tuple[np.ndarray, np.ndarray]:
    """(per-resample ratio, replicate ratio) of sum num / sum den; replicate = mean over slots of slot ratios."""
    per = np.array([cl[num].sum() / cl[den].sum() for cl in cols])
    tot = np.zeros((draws.n_boot, len(cols), 2))
    for j, cl in enumerate(cols):
        m = draws.pick == j
        tot[m] = draws.w[m][:, : len(cl[num])] @ np.column_stack([cl[num], cl[den]])
    if (tot[..., 1] <= 0).any():
        raise AssertionError(f"a bootstrap slot has no host for {num} / {den}")
    return per, (tot[..., 0] / tot[..., 1]).mean(axis=1)


def wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Inverted-CDF weighted quantile (smallest value whose cumulative weight reaches q of the total)."""
    o = np.argsort(values, kind="stable")
    v, cw = values[o], np.cumsum(weights[o])
    return float(v[np.searchsorted(cw, q * cw[-1] - 1e-12 * cw[-1])])


def pooled_quantiles(tabs: list[pd.DataFrame], k: int, draws) -> dict[str, tuple[float, np.ndarray]]:
    """Median and 90th percentile of regret over eligible host appearances: point and replicates."""
    vals = [t.regret.to_numpy(float) for t in tabs]
    elig = [(t.n_sites >= k).to_numpy(float) for t in tabs]
    allv = np.concatenate([v[e > 0] for v, e in zip(vals, elig, strict=True)])
    out = {}
    for name, q in QUANTS:
        point = float(np.quantile(allv, q, method="inverted_cdf"))
        reps = np.empty(draws.n_boot)
        for b in range(draws.n_boot):
            v = np.concatenate([vals[j] for j in draws.pick[b]])
            w = np.concatenate([draws.w[b, s, : len(vals[j])] * elig[j] for s, j in enumerate(draws.pick[b])])
            reps[b] = wquantile(v, w, q)
        out[name] = (point, reps)
    return out


def summary(per: np.ndarray, reps: np.ndarray) -> dict:
    return {k: v for k, v in ss.summary(per, reps).items() if k != "per_resample"}


def point_summary(point: float, reps: np.ndarray) -> dict:
    q = c.quantiles(reps)
    return {"value": point, **q, "boot_sd": float(reps.std(ddof=1))}


def with_gaps(est: dict[str, tuple[np.ndarray, np.ndarray]], lower_better: bool) -> dict:
    blk = {m: summary(*est[m]) for m in MODELS}
    for name, a, b in GAPS:
        d = summary(est[a][0] - est[b][0], est[a][1] - est[b][1])
        d["ci95_excludes_zero"] = bool(d["ci95"][0] > 0 or d["ci95"][1] < 0)
        blk[name] = d
    blk["sign_convention"] = ("difference = model minus S; " +
                              ("negative = smaller regret than S" if lower_better else "positive = more often than S"))
    return blk


# ----------------------------------------------------------------------------- PHS truncation

def release_root(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for base in RELEASE_CANDIDATES:
        if (base / RELEASE_SUB).is_dir():
            return base / RELEASE_SUB
    raise FileNotFoundError(f"release data not found under {[str(b) for b in RELEASE_CANDIDATES]}")


def raw_release(root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """(entries, O sites): one row per completed neutral entry of the release (formula = directory, host_id,
    site_label, phs), and per formula the inequivalent O site labels of site_info/<formula>/cell_info.txt."""
    from dftgnn.data.universe import parse_cell_info

    rows, osites = [], {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        de = json.loads((d / "defect_visual_data.json").read_text())
        mp = json.loads((d / "bulk_visual_data.json").read_text())["mp_id"]
        cell = parse_cell_info((root.parent / "site_info" / d.name / "cell_info.txt").read_text())
        osites[d.name] = sorted((k for k in cell if re.fullmatch(r"O\d+", k)), key=lambda k: int(k[1:]))
        for site, v in de["defect_energy_summary"]["defect_energies"].items():
            for q, e in zip(v["charges"], v["defect_energies"], strict=True):
                if q == 0:
                    rows.append({"formula": d.name, "host_id": mp, "site_label": site.removeprefix("Va_"),
                                 "phs": e["is_shallow"] is True})
    return pd.DataFrame(rows), osites


def label_gaps(ids: pd.DataFrame) -> dict[str, list[str]]:
    """Per formula: labels missing from O1..O(max retained) (a lower bound on removed sites)."""
    out = {}
    for f, g in ids.groupby("formula"):
        nums = sorted(int(re.fullmatch(r"O(\d+)", s).group(1)) for s in g.site_label)
        miss = sorted(set(range(1, nums[-1] + 1)) - set(nums))
        if miss:
            out[f] = [f"O{i}" for i in miss]
    return out


def phs_tables(rel_root: Path) -> tuple[pd.DataFrame, dict]:
    """Per release host: entries, PHS entries, retained sites, class; plus the cross-checks."""
    from analysis.dshell import classes as cl

    raw, osites = raw_release(rel_root)
    ids = pd.read_csv(ROOT / "data" / "universe_v1_ids.csv")
    casc = c.record("filter_cascade")["payload"]
    rel_step = next(s for s in casc["steps"] if s["step"] == "release")
    removed = {r["site_id"]: r["reason"] for s in casc["steps"] for r in s["removed"]}
    casc_phs = {sid for sid, why in removed.items() if why.split(";")[0] == "PHS"}
    raw["site_id"] = raw.formula + "_Va_" + raw.site_label + "_0"
    raw_phs = set(raw.site_id[raw.phs])
    retained = set(ids.site_id)
    checks = {
        "raw_neutral_entries": len(raw), "cascade_release_entries": int(rel_step["sites"]),
        "raw_hosts": int(raw.formula.nunique()), "cascade_release_hosts": int(rel_step["hosts"]),
        "raw_phs_entries": len(raw_phs), "cascade_phs_entries": len(casc_phs),
        "phs_sets_identical": raw_phs == casc_phs,
        "retained_plus_removed_equals_release": (retained | set(removed)) == set(raw.site_id)
        and not (retained & set(removed)),
        "no_phs_site_retained": not (raw_phs & retained)}
    sar = c.record("screening_absence")["payload"]["release_neutral_completed"]
    checks["screening_absence_entries_phs"] = [int(sar["entries"]), int(sar["phs"])]
    ok = (checks["raw_neutral_entries"] == checks["cascade_release_entries"] == sar["entries"]
          and checks["raw_phs_entries"] == checks["cascade_phs_entries"] == sar["phs"]
          and checks["phs_sets_identical"] and checks["retained_plus_removed_equals_release"]
          and checks["no_phs_site_retained"] and checks["raw_hosts"] == checks["cascade_release_hosts"])
    if not ok:
        raise AssertionError(f"PHS cross-checks failed: {checks}")

    pub = {h["host_id"]: h["class"] for h in c.record("derived_dshell/dshell_classes")["payload"]["hosts"]}
    f2h = ids.drop_duplicates("formula").set_index("formula").host_id.to_dict()
    gaps = label_gaps(ids)
    rows = []
    for f, g in raw.groupby("formula", sort=True):
        in_uni = f in f2h
        cls = pub[f2h[f]] if in_uni else cl.classify_host(g.host_id.iloc[0], f)["class"]
        n_ret = int((ids.formula == f).sum())
        removed_f = [s for s in g.site_id if s in removed]
        if not set(g.site_label) <= set(osites[f]):
            raise AssertionError(f"{f}: release entries outside the cell_info O sites")
        rows.append({"formula": f, "host_id": f2h.get(f, g.host_id.iloc[0]), "in_universe": in_uni,
                     "class": cls, "class_source": "published" if in_uni else "same rule (not in universe)",
                     "n_release": len(g), "n_phs": int(g.phs.sum()), "n_retained": n_ret,
                     "n_removed_any": len(removed_f), "n_label_gaps": len(gaps.get(f, [])),
                     "n_o_sites": len(osites[f]), "n_not_completed": len(osites[f]) - len(g)})
    tab = pd.DataFrame(rows)
    if (tab.n_release != tab.n_retained + tab.n_removed_any).any():
        raise AssertionError("release entries != retained + removed for some host")
    # a label gap is either a removed release entry or an O site with no completed neutral entry
    have = set(raw.site_id)
    bad_gap = [f for f, labs in gaps.items()
               if not all(f"{f}_Va_{s}_0" in removed or f"{f}_Va_{s}_0" not in have for s in labs)]
    if bad_gap:
        raise AssertionError(f"label gaps not explained by the release: {bad_gap[:5]}")
    gap_kind = {"phs": 0, "other_filter": 0, "no_completed_neutral_entry": 0}
    for f, labs in gaps.items():
        for s in labs:
            sid = f"{f}_Va_{s}_0"
            kind = ("no_completed_neutral_entry" if sid not in have else
                    "phs" if sid in casc_phs else "other_filter")
            gap_kind[kind] += 1
    checks["label_gap_sites_by_cause"] = gap_kind
    checks["release_hosts_without_completed_neutral_entry"] = len(set(osites) - set(raw.formula))
    uni = tab[tab.in_universe]
    if len(uni) != 818 or set(uni.host_id) != set(pub):
        raise AssertionError("universe hosts do not match the published classes")
    checks["universe_hosts"] = len(uni)
    checks["pymatgen_core_version"] = _pmg_version()
    checks["published_classes_pymatgen_version"] = c.record("derived_dshell/dshell_classes")["payload"][
        "pymatgen_version"]
    return tab, checks


def _pmg_version() -> str:
    from importlib.metadata import version

    return version("pymatgen-core")


def phs_counts(tab: pd.DataFrame) -> dict:
    uni = tab[tab.in_universe]
    part = uni[uni.n_phs > 0]
    gap_hosts = uni[uni.n_label_gaps > 0]
    by_class = {}
    for k in ss_classes():
        u, a = uni[uni["class"] == k], tab[tab["class"] == k]
        by_class[k] = {
            "universe_hosts": len(u),
            "partially_filtered_hosts": int((u.n_phs > 0).sum()),
            "partially_filtered_hosts_ge2_retained_sites": int(((u.n_phs > 0) & (u.n_retained >= 2)).sum()),
            "hosts_with_label_gaps": int((u.n_label_gaps > 0).sum()),
            "hosts_with_o_site_lacking_completed_neutral_entry": int((u.n_not_completed > 0).sum()),
            "universe_hosts_release_entries": int(u.n_release.sum()),
            "universe_hosts_phs_entries": int(u.n_phs.sum()),
            "universe_hosts_phs_fraction": float(u.n_phs.sum() / u.n_release.sum()) if len(u) else 0.0,
            "release_hosts": len(a), "release_hosts_outside_universe": int((~a.in_universe).sum()),
            "release_entries": int(a.n_release.sum()), "release_phs_entries": int(a.n_phs.sum()),
            "release_phs_fraction": float(a.n_phs.sum() / a.n_release.sum()) if len(a) else 0.0}
    out_uni = tab[~tab.in_universe]
    return {
        "universe_hosts": len(uni),
        "partially_filtered_hosts": len(part),
        "partially_filtered_hosts_ge2_retained_sites": int((part.n_retained >= 2).sum()),
        "phs_sites_removed_from_partially_filtered_hosts": int(part.n_phs.sum()),
        "partially_filtered_hosts_by_number_removed": {str(n): int((part.n_phs == n).sum())
                                                      for n in sorted(part.n_phs.unique())},
        "hosts_with_any_removed_site": int((uni.n_removed_any > 0).sum()),
        "hosts_with_o_site_lacking_completed_neutral_entry": int((uni.n_not_completed > 0).sum()),
        "o_sites_lacking_completed_neutral_entry": int(uni.n_not_completed.sum()),
        "universe_hosts_o_sites_cell_info": int(uni.n_o_sites.sum()),
        "universe_hosts_retained_sites": int(uni.n_retained.sum()),
        "hosts_with_label_gaps": len(gap_hosts),
        "partially_filtered_hosts_without_label_gap": int(((uni.n_phs > 0) & (uni.n_label_gaps == 0)).sum()),
        "release_hosts": len(tab), "release_hosts_outside_universe": len(out_uni),
        "release_hosts_outside_universe_all_phs": int((out_uni.n_phs == out_uni.n_release).sum()),
        "release_entries": int(tab.n_release.sum()), "release_phs_entries": int(tab.n_phs.sum()),
        "release_phs_fraction": float(tab.n_phs.sum() / tab.n_release.sum()),
        "by_class": by_class}


def ss_classes() -> tuple[str, ...]:
    return tuple(c.record("derived_dshell/dshell_classes")["payload"]["classes"])


# ----------------------------------------------------------------------------- reproduction

def compare_stored(mine: dict, stored: dict, path: str = "") -> list[tuple[str, float]]:
    """Max abs differences between every numeric leaf of stored and the same key of mine (missing = inf)."""
    out = []
    if isinstance(stored, dict):
        for k, v in stored.items():
            out += compare_stored(mine.get(k, {}) if isinstance(mine, dict) else {}, v, f"{path}/{k}")
    elif isinstance(stored, list):
        if not isinstance(mine, list) or len(mine) != len(stored):
            return [(path, float("inf"))]
        for i, (a, b) in enumerate(zip(mine, stored, strict=True)):
            out += compare_stored(a, b, f"{path}/{i}")
    elif isinstance(stored, bool):
        out.append((path, 0.0 if mine == stored else float("inf")))
    elif isinstance(stored, (int, float)):
        out.append((path, abs(float(mine) - float(stored)) if isinstance(mine, (int, float)) else float("inf")))
    return out


# ----------------------------------------------------------------------------- run

def run(rel_root: Path, delta: float) -> dict:
    from dftgnn.stats.g1 import HierDraws, seed_ensemble

    long, inputs = ss.headline_frame()
    ens = seed_ensemble(long)
    if (ens.n_seeds != ss.N_SEEDS).any() or sorted(ens.r.unique()) != list(range(N_R)):
        raise AssertionError("654-host ensemble: wrong seeds or resamples")
    draws = HierDraws.make([N_HOSTS] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    within = ss.reproduce(ens, draws)
    if not within["reproduced"]:
        raise AssertionError(f"within-host gap not reproduced: {within}")
    tabs = {m: [host_table(ens[(ens.model == m) & (ens.r == r)], delta) for r in range(N_R)] for m in MODELS}
    for m in MODELS:
        if any(len(t) != N_HOSTS for t in tabs[m]):
            raise AssertionError(f"{m}: test hosts per resample != {N_HOSTS}")
    for m in MODELS[1:]:
        for a, b in zip(tabs[m], tabs["S"], strict=True):
            if not (a.host_id.equals(b.host_id) and a.n_sites.equals(b.n_sites) and a.gap2.equals(b.gap2)):
                raise AssertionError(f"{m}: test hosts or targets differ from S")

    phs_tab, phs_checks = phs_tables(rel_root)
    partial = set(phs_tab.host_id[phs_tab.in_universe & (phs_tab.n_phs > 0)])

    # reproduction of the stored site-selection results (same code path as site_selection.run)
    stored = c.record("derived_coord/site_selection")["payload"]
    repro_ss = {}
    for k in MIN_SITES:
        st = {m: [ss.stats(t, k) for t in tabs[m]] for m in MODELS}
        per = {m: ss.per_resample(st[m]) for m in MODELS}
        rep = {m: ss.replicates(st[m], draws) for m in MODELS}
        blk = {m: ss.summary(per[m][:, 0], rep[m][:, 0]) for m in MODELS}
        blk["chance"] = ss.summary(per["S"][:, 1], rep["S"][:, 1])
        for name, a, b in GAPS:
            d = ss.summary(per[a][:, 0] - per[b][:, 0], rep[a][:, 0] - rep[b][:, 0])
            d["ci95_excludes_zero"] = bool(d["ci95"][0] > 0 or d["ci95"][1] < 0)
            blk[name] = d
        elig = np.array([s[:, 1].sum() for s in st["S"]])
        blk["counts"] = {"eligible_hosts_per_resample": {"mean": float(elig.mean()), "min": int(elig.min()),
                                                         "max": int(elig.max())},
                         "host_resample_cases": int(elig.sum()),
                         "distinct_eligible_hosts": len({h for t in tabs["S"] for h in t.host_id[t.n_sites >= k]}),
                         "cases_with_tied_target_minima": sum(int(((t.n_sites >= k) & (t.n_minima > 1)).sum())
                                                              for t in tabs["S"]),
                         "cases_with_tied_lowest_prediction": {m: sum(int((t.p_tie & (t.n_sites >= k)).sum())
                                                                      for t in tabs[m]) for m in MODELS}}
        repro_ss[f"hosts_ge{k}_sites"] = blk
    diffs = compare_stored(c.to_jsonable(repro_ss), stored["results"])
    worst = max(diffs, key=lambda x: x[1])
    reproduction = {
        "site_selection": {"source": "results/derived_coord/site_selection.json payload.results (every numeric "
                                     "leaf, per-resample values and counts included)",
                           "n_values_compared": len(diffs), "max_abs_difference": float(worst[1]),
                           "worst_pointer": worst[0], "reproduced": bool(worst[1] <= TOL)},
        "within_host_gap": within}
    if not reproduction["site_selection"]["reproduced"]:
        raise AssertionError(f"site_selection not reproduced: {worst}")

    results = {}
    phs_sel = {}
    for k in MIN_SITES:
        cols = {m: [columns(t, k, delta) for t in tabs[m]] for m in MODELS}
        est = lambda num, den, cols=cols: {m: ratio(cols[m], num, den, draws) for m in MODELS}
        reg = est("regret", "elig")
        qs = {m: pooled_quantiles(tabs[m], k, draws) for m in MODELS}
        quant = {}
        for name, _ in QUANTS:
            q = {m: point_summary(qs[m][name][0], qs[m][name][1]) for m in MODELS}
            for gname, a, b in GAPS:
                q[gname] = point_summary(qs[a][name][0] - qs[b][name][0], qs[a][name][1] - qs[b][name][1])
            quant[name] = q
        sep_counts = {
            "separated_hosts_per_resample": {"mean": float(np.mean([cl["sep"].sum() for cl in cols["S"]])),
                                             "min": int(min(cl["sep"].sum() for cl in cols["S"])),
                                             "max": int(max(cl["sep"].sum() for cl in cols["S"]))},
            "separated_host_resample_cases": int(sum(cl["sep"].sum() for cl in cols["S"])),
            "distinct_separated_hosts": len({h for t in tabs["S"] for h in
                                             t.host_id[(t.n_sites >= k) & (t.gap2 > delta)]}),
            "eligible_host_resample_cases": int(sum(cl["elig"].sum() for cl in cols["S"]))}
        sep = with_gaps(est("sep_hit", "sep"), lower_better=False)
        sep["chance"] = summary(*ratio(cols["S"], "sep_chance", "sep", draws))
        sep["counts"] = sep_counts
        pop = est("pop", "elig")
        conc = {}
        for m in MODELS:
            s = summary(*reg[m])
            conc[m] = {"exp_mean_regret_over_kT": float(np.exp(s["mean"] / kbt())),
                       "ci95": [float(np.exp(x / kbt())) for x in s["ci95"]],
                       "mean_population_ratio_exp_minus_regret_over_kT": summary(*pop[m])}
        conc["kT_eV"] = kbt()
        conc["T_K"] = T_K
        miss_cases = {m: int(sum(cl["miss"].sum() for cl in cols[m])) for m in MODELS}
        results[f"hosts_ge{k}_sites"] = {
            "regret": {"mean_eV": with_gaps(reg, lower_better=True),
                       "median_eV": quant["median"], "p90_eV": quant["p90"],
                       "conditional_on_miss_eV": with_gaps(est("regret", "miss"), lower_better=True),
                       "miss_cases": miss_cases},
            "hit_within_delta": with_gaps(est("within", "elig"), lower_better=False),
            "separated_hosts": sep,
            "concentration_factor": conc}
        # selection without partially filtered hosts
        colsd = {m: [columns(t, k, delta, drop=partial) for t in tabs[m]] for m in MODELS}
        estd = lambda num, den, cols=colsd: {m: ratio(cols[m], num, den, draws) for m in MODELS}
        sel = with_gaps(estd("hit", "elig"), lower_better=False)
        sel["chance"] = summary(*ratio(colsd["S"], "chance", "elig", draws))
        removed_cases = int(sum(cl["elig"].sum() for cl in cols["S"]) - sum(cl["elig"].sum() for cl in colsd["S"]))
        sel["counts"] = {"host_resample_cases": int(sum(cl["elig"].sum() for cl in colsd["S"])),
                         "host_resample_cases_removed": removed_cases,
                         "distinct_eligible_hosts": len({h for t in tabs["S"] for h in
                                                         t.host_id[(t.n_sites >= k) & ~t.host_id.isin(partial)]}),
                         "distinct_partially_filtered_eligible_hosts": len(
                             {h for t in tabs["S"] for h in t.host_id[(t.n_sites >= k) & t.host_id.isin(partial)]})}
        phs_sel[f"hosts_ge{k}_sites"] = {"selection": sel,
                                         "mean_regret_eV": with_gaps(estd("regret", "elig"), lower_better=True)}

    phs = phs_counts(phs_tab)
    phs["without_partially_filtered_hosts"] = phs_sel
    phs["checks"] = phs_checks
    phs["partially_filtered_host_list"] = sorted(
        ({"host_id": r.host_id, "formula": r.formula, "class": r["class"], "n_release": int(r.n_release),
          "n_phs_removed": int(r.n_phs), "n_retained": int(r.n_retained)}
         for _, r in phs_tab[phs_tab.in_universe & (phs_tab.n_phs > 0)].iterrows()), key=lambda x: x["formula"])
    for k in MIN_SITES:
        results[f"hosts_ge{k}_sites"]["phs_truncation"] = phs_sel[f"hosts_ge{k}_sites"]
    return {
        "label": LABEL, "design": "654 hosts, outer resamples r = 0-29 x seeds 0-4, 164 test hosts per resample",
        "delta_eV": delta,
        "estimator": {
            "regret": "per host E_f(DFT) of the lowest-predicted site (seed ensemble; first in site_id order on a "
                      "tie) minus the host's DFT minimum over its tested retained sites; zero on a hit",
            "mean": "per resample sum over eligible hosts / number of eligible hosts; point = mean over the 30 "
                    "resamples; replicate = mean over slots of sum(w x) / sum(w eligible) on "
                    "HierDraws.make([164] * 30, 2000, 20261008); 95% = 2.5 / 97.5 percentiles",
            "quantiles": "median and 90th percentile of regret over eligible host appearances pooled over the 30 "
                         "resamples (inverted CDF); replicate = weighted quantile with the draw weights",
            "conditional_on_miss": "sum regret / number of misses per resample (same averaging)",
            "hit_within_delta": "regret <= delta",
            "separated_hosts": "eligible hosts whose two lowest DFT sites differ by more than delta; exact hit "
                               "rate there; chance = mean of 1 / n_sites",
            "phs_truncation": "PHS entries per host counted from the raw release (is_shallow true, charge 0); "
                              "partially filtered = universe host with >= 1 PHS entry removed; selection recomputed "
                              "with those hosts ineligible on the same draws",
            "concentration_factor": "exp(mean regret / kB T) at 1000 K, kB from scipy.constants (eV/K); interval = "
                                    "the map of the regret interval; mean_population_ratio = mean over hosts of "
                                    "exp(-regret / kB T) with the regret averaging",
            "eligibility": "hosts with >= 2 (or >= 3) tested sites"},
        "sign_convention": "D*-S and D-S are model minus S on the same replicates; for regret negative = the "
                           "model's chosen site lies closer to the DFT minimum than S's; for rates positive = "
                           "the model selects more often than S",
        "results": results, "phs_truncation": phs, "reproduction": reproduction,
        "inputs": {**inputs, "site_selection_record": "results/derived_coord/site_selection.json",
                   "filter_cascade": "results/filter_cascade.json", "ids": "data/universe_v1_ids.csv",
                   "classes": "results/derived_dshell/dshell_classes.json",
                   "release": "oxygen_vacancies_db_data/*/defect_visual_data.json (Kumagai release, raw)"}}


def main(argv: list[str] | None = None) -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(f"{NAME}_dryrun.json"))
    ap.add_argument("--release-root", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_config()
    delta = float(cfg.stats.delta_eV)
    payload = run(release_root(a.release_root), delta)
    payload["runtime_s"] = round(time.time() - t0, 1)
    payload = c.to_jsonable(payload)
    text = json.dumps(payload, indent=1, allow_nan=False)
    if a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print(a.out, f"{payload['runtime_s']} s")
        return
    print(write_result(NAME, payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
