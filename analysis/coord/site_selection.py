"""Site selection: does the network pick the oxygen site with the lowest vacancy formation energy of a host?

Nothing is trained. Physically, the site of lowest formation energy is the one that dominates the equilibrium vacancy
population of an oxide (its concentration carries exp(-E_f / kT)), so getting that site right is the within-host
decision a screening study needs, independent of the absolute energy scale. The metric is fixed before any gain
was computed (coordination spec (b)).

Estimator (654 hosts, 30 resamples x 5 seeds, S, D and D*)
    ensemble    seed-ensemble mean prediction per (model, r, site) (dftgnn.stats.g1.seed_ensemble)
    hit         for a test host with >= 2 tested sites: 1 if the lowest-predicted site is a site of lowest DFT
                target (ties in the target: a hit if the lowest-predicted site is among the minima), else 0; ties
                in the prediction take the first site in site_id order (none occur, counted)
    fraction    per resample hits / eligible hosts; point = mean over the 30 resamples
    chance      per eligible host n_minima / n_sites (the hit probability of a random pick); same averaging
    bootstrap   the headline draws HierDraws.make([164] * 30, 2000, 20261008) on the per-host statistics (hit,
                eligible) with hosts sorted by host_id; replicate = mean over slots of the slot fraction
                (sum w hit / sum w eligible); 95% = 2.5 / 97.5 percentiles; differences D* - S and D - S taken per
                replicate (paired); positive = better selection than S
    restriction the same quantities on hosts with >= 3 tested sites

Before use, the loaded tables reproduce the within-host residual gap S - D* at 654 hosts (30 x 5) of
results/derived_dstar/dstar_extras.json payload.a_within.headline_654_30x5.S_minus_Dstar (tolerance 1e-9 eV).

    PYTHONPATH=$PWD/src python analysis/coord/site_selection.py [--dry-run] [--out <path>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis.dstar import dstar_analysis as da

OUT_DIR = ROOT / "results" / "derived_coord"
DRY_OUT = Path("site_selection_dryrun.json")
HEADLINE_B = 654
N_R, N_SEEDS, N_HOSTS = 30, 5, 164
MODELS = ("S", "D", "Dstar")
GAPS = (("Dstar_minus_S", "Dstar", "S"), ("D_minus_S", "D", "S"))
MIN_SITES = (2, 3)
TOL = 1e-9
LABEL = ("site selection: fraction of test hosts whose lowest-predicted oxygen site is a site of lowest DFT "
         "vacancy formation energy; S, D, D* at 654 hosts, 30 x 5, headline draws")


def host_hit(y: np.ndarray, p: np.ndarray) -> tuple[int, int, bool]:
    """(hit, number of target minima, prediction tie) of one host; ties in y count as a hit for any minimum."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    i = int(np.argmin(p))
    minima = y == y.min()
    return int(minima[i]), int(minima.sum()), bool((p == p[i]).sum() > 1)


def host_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per host (sorted by host_id) of one (model, r) ensemble: n_sites, hit, n_minima, prediction tie."""
    rows = []
    for h, g in frame.sort_values("site_id").groupby("host_id", sort=True):
        hit, nmin, tie = host_hit(g.y_true.to_numpy(), g.y_pred.to_numpy())
        rows.append({"host_id": h, "n_sites": len(g), "hit": hit, "n_minima": nmin, "p_tie": tie})
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame, k: int) -> np.ndarray:
    """(n_hosts, 3) per-host statistics [hit, eligible, chance x eligible] for hosts with >= k sites."""
    e = (t.n_sites >= k).to_numpy(float)
    return np.column_stack([t.hit.to_numpy(float) * e, e, (t.n_minima / t.n_sites).to_numpy(float) * e])


def replicates(st: list[np.ndarray], draws) -> np.ndarray:
    """(n_boot, 2) replicate fractions [hit, chance]: mean over slots of the weighted slot ratio."""
    tot = np.zeros((draws.n_boot, len(st), 3))
    for j, s in enumerate(st):
        m = draws.pick == j
        tot[m] = draws.w[m][:, : len(s)] @ s
    if (tot[..., 1] <= 0).any():
        raise AssertionError("a bootstrap slot has no eligible host")
    return np.stack([tot[..., 0] / tot[..., 1], tot[..., 2] / tot[..., 1]], axis=-1).mean(axis=1)


def per_resample(st: list[np.ndarray]) -> np.ndarray:
    tot = np.array([s.sum(axis=0) for s in st])
    return np.column_stack([tot[:, 0] / tot[:, 1], tot[:, 2] / tot[:, 1]])


def headline_frame() -> tuple[pd.DataFrame, dict]:
    """Seed-level S, D (precision_eval_long, v2) and D* (dretrain_v2) rows at 654 hosts, 30 x 5."""
    ref, _ = da.s_reference()
    dstar, info = da.load_dstar(ref)
    prec, f_prec = c.load_table("precision_eval_long")
    hd = prec[(prec.backbone == "v2") & (prec.B == HEADLINE_B) & prec.model.isin(["S", "D"])][da.COLS]
    hd = pd.concat([hd, dstar[dstar.B == HEADLINE_B][da.COLS]], ignore_index=True)
    return hd, {"precision_eval_long": f_prec, "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl",
                                                                     "sha256": info["spec_sha256"]},
                "n_dstar_runs": len(info["runs"])}


def reproduce(ens: pd.DataFrame, draws) -> dict:
    """Within-host residual gap S - D* at 654 (30 x 5) against results/derived_dstar/dstar_extras.json."""
    from dftgnn.stats.g1 import contrast, resample_stats

    st = {m: {HEADLINE_B: [resample_stats(ens[(ens.model == m) & (ens.r == r)].sort_values("site_id"))
                           for r in range(N_R)]} for m in ("S", "Dstar")}
    p = contrast(st["S"], st["Dstar"], draws, "within_host_mae", ci=c.CI, upper_quantile=c.UPPER_Q)[HEADLINE_B]
    rec = c.record("derived_dstar/dstar_extras")["payload"]["a_within"]["headline_654_30x5"]["S_minus_Dstar"]
    got = {"A": p["diff"]["mean"], "ci95": p["diff"]["ci"], "upper_one_sided_95": p["diff"]["upper_one_sided"]}
    dev = max(abs(got["A"] - rec["A"]), *(abs(a - b) for a, b in zip(got["ci95"], rec["ci95"], strict=True)),
              abs(got["upper_one_sided_95"] - rec["upper_one_sided_95"]))
    return {"source": "results/derived_dstar/dstar_extras.json payload.a_within.headline_654_30x5.S_minus_Dstar",
            "recomputed": got, "max_abs_difference": float(dev), "reproduced": bool(dev <= TOL)}


def summary(per: np.ndarray, reps: np.ndarray) -> dict:
    from dftgnn.stats.g1 import summarize

    s = summarize(per, reps, ci=c.CI, upper_quantile=c.UPPER_Q)
    return {"mean": s["mean"], "ci95": s["ci"], "upper_one_sided_95": s["upper_one_sided"], "boot_sd": s["boot_sd"],
            "sd_over_resamples": s["sd_over_resamples"], "per_resample": s["per_resample"]}


def run() -> dict:
    from dftgnn.stats.g1 import HierDraws, seed_ensemble

    long, inputs = headline_frame()
    ens = seed_ensemble(long)
    if (ens.n_seeds != N_SEEDS).any() or sorted(ens.r.unique()) != list(range(N_R)):
        raise AssertionError("654-host ensemble: wrong seeds or resamples")
    draws = HierDraws.make([N_HOSTS] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    repro = reproduce(ens, draws)
    if not repro["reproduced"]:
        raise AssertionError(f"within-host gap not reproduced: {repro}")
    tabs = {}
    for m in MODELS:
        tabs[m] = []
        for r in range(N_R):
            t = host_table(ens[(ens.model == m) & (ens.r == r)])
            if len(t) != N_HOSTS:
                raise AssertionError(f"{m}, r = {r}: {len(t)} test hosts")
            tabs[m].append(t)
    for m in MODELS[1:]:
        for a, b in zip(tabs[m], tabs["S"], strict=True):
            if not (a.host_id.equals(b.host_id) and a.n_sites.equals(b.n_sites) and a.n_minima.equals(b.n_minima)):
                raise AssertionError(f"{m}: test hosts or targets differ from S")
    out = {}
    for k in MIN_SITES:
        st = {m: [stats(t, k) for t in tabs[m]] for m in MODELS}
        per = {m: per_resample(st[m]) for m in MODELS}
        rep = {m: replicates(st[m], draws) for m in MODELS}
        elig = np.array([s[:, 1].sum() for s in st["S"]])
        tied_y = sum(int(((t.n_sites >= k) & (t.n_minima > 1)).sum()) for t in tabs["S"])
        blk = {m: summary(per[m][:, 0], rep[m][:, 0]) for m in MODELS}
        blk["chance"] = summary(per["S"][:, 1], rep["S"][:, 1])
        for name, a, b in GAPS:
            d = summary(per[a][:, 0] - per[b][:, 0], rep[a][:, 0] - rep[b][:, 0])
            d["ci95_excludes_zero"] = bool(d["ci95"][0] > 0 or d["ci95"][1] < 0)
            blk[name] = d
        blk["counts"] = {"eligible_hosts_per_resample": {"mean": float(elig.mean()), "min": int(elig.min()),
                                                         "max": int(elig.max())},
                         "host_resample_cases": int(elig.sum()),
                         "distinct_eligible_hosts": len({h for t in tabs["S"] for h in t.host_id[t.n_sites >= k]}),
                         "cases_with_tied_target_minima": tied_y,
                         "cases_with_tied_lowest_prediction": {m: sum(int((t.p_tie & (t.n_sites >= k)).sum())
                                                                      for t in tabs[m]) for m in MODELS}}
        out[f"hosts_ge{k}_sites"] = blk
    return {"label": LABEL,
            "definition": {"hit": "lowest-predicted site (seed ensemble) is a site of lowest DFT target; ties in the "
                                  "target count for any minimum; ties in the prediction take the first site in "
                                  "site_id order",
                           "fraction": "hits / eligible hosts per resample; point = mean over 30 resamples",
                           "chance": "mean over eligible hosts of n_minima / n_sites (random pick)",
                           "bootstrap": "HierDraws.make([164] * 30, 2000, 20261008); replicate = mean over slots of "
                                        "sum(w hit) / sum(w eligible); paired differences per replicate",
                           "sign": "Dstar_minus_S, D_minus_S: positive = better selection than S"},
            "design": "654 hosts, outer resamples r = 0-29 x seeds 0-4",
            "results": out, "reproduction": repro, "inputs": inputs}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    payload = c.to_jsonable(run())
    if "--dry-run" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DRY_OUT
        out.write_text(json.dumps(payload, indent=1, allow_nan=False))
        print(out)
        return
    print(write_result("site_selection", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
