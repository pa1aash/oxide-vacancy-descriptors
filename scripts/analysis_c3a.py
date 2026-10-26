"""Staging analysis on the v2 arm.

    python scripts/analysis_c3a.py [--no-write]

Per budget: the seed-ensemble MAE of S-v2, D-v2 and P-v2 on identical test sites of every outer resample, and two
paired contrasts with the hierarchical bootstrap: D - P (the cost of predicting rather than knowing the descriptors)
and P - S (staging against end-to-end, negative: P better). P - S is classified by the four-outcome rule: P better
(95% interval below 0), P worse (above 0), equivalent (90% interval inside +/- delta_P), inconclusive (otherwise).
P1-v2's per-descriptor test R^2 and MAE are averaged over resamples and seeds. Only budgets where every resample has
all seeds of S, D and P enter.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnostics_sweep as D

from dftgnn.config import load_config
from dftgnn.stats.metrics import aggregate_delta, aggregate_resamples


def outcome(ci95, ci90, margin) -> str:
    within = -margin < ci90[0] and ci90[1] < margin
    if ci95[1] < 0:
        return "P better" + (" (within the equivalence margin)" if within else "")
    if ci95[0] > 0:
        return "P worse" + (" (within the equivalence margin)" if within else "")
    return "equivalent" if within else "inconclusive"


def p_index(root: Path):
    import pandas as pd

    rows = []
    for f in sorted(glob.glob(str(root / "results" / "p_v2" / "*.json"))):
        p = json.loads(Path(f).read_text())["payload"]
        s = p["spec"]
        rows.append({"run_id": p["run_id"], "model": "P", "split": s["split"], "r": s["r"], "B": s["budget"],
                     "seed": s["seed"], "pred": root / p["predictions"]["path"]})
    return pd.DataFrame(rows)


def p1_descriptors(root: Path) -> dict:
    per = {}
    for f in sorted(glob.glob(str(root / "results" / "p1_v2" / "*.json"))):
        p = json.loads(Path(f).read_text())["payload"]
        if not p["spec"]["split"].startswith("outer"):
            continue
        for name, v in p["metrics"]["per_descriptor"].items():
            per.setdefault((p["spec"]["budget"], name), []).append((v["r2"], v["mae"]))
    out = {}
    for (b, name), vals in per.items():
        r2 = [x[0] for x in vals if x[0] is not None]
        out.setdefault(str(b), {})[name] = {"r2_mean": float(np.mean(r2)) if r2 else None,
                                            "mae_mean": float(np.mean([x[1] for x in vals])), "n_runs": len(vals)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", default=str(ROOT), help="checkout holding results/v2_sweep, p_v2 and p1_v2")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    root = Path(a.artefacts)
    d = cfg.models.injection_mode
    n_boot, margin, n_seeds = cfg.analysis.bootstrap.draws, cfg.analysis.equivalence_margin_eV, len(cfg.training.seeds)
    ix = D.run_index(str(root / "results" / "v2_sweep"))
    ix = ix[ix.split.str.startswith("outer")]
    px = p_index(root)
    px = px[px.split.str.startswith("outer")]
    rows = {}
    for b in cfg.budgets.hosts:
        rs = [r for r in range(cfg.split.n_outer_resamples)
              if all(len(t[(t.model == m) & (t.B == b) & (t.r == r)]) == n_seeds for t, m in ((ix, "S"), (ix, d), (px, "P")))]
        if len(rs) < cfg.split.n_outer_resamples:
            continue
        st = {}
        for name, t, m in (("S", ix, "S"), ("D", ix, d), ("P", px, "P")):
            st[name] = {r: D.ensemble(t[(t.model == m) & (t.B == b) & (t.r == r)]) for r in rs}
        for name in ("D", "P"):
            st[name] = {r: D._align(st[name][r], st["S"][r]) for r in rs}
        hs = {k: [D.stats_of(v[r]) for r in rs] for k, v in st.items()}
        mae = {k: aggregate_resamples(v, n_boot=n_boot)["mae"] for k, v in hs.items()}
        dp = aggregate_delta(hs["D"], hs["P"], n_boot=n_boot)["mae"]
        ps95 = aggregate_delta(hs["P"], hs["S"], n_boot=n_boot, ci=0.95)["mae"]
        ps90 = aggregate_delta(hs["P"], hs["S"], n_boot=n_boot, ci=cfg.analysis.equivalence_tost_ci)["mae"]
        rows[str(b)] = {"mae": {k: {"mean": v["mean"], "ci": v["ci"]} for k, v in mae.items()},
                        "D_minus_P": {"mean": dp["mean"], "ci95": dp["ci"]},
                        "P_minus_S": {"mean": ps95["mean"], "ci95": ps95["ci"], "ci90": ps90["ci"],
                                      "outcome": outcome(ps95["ci"], ps90["ci"], margin)}}
    res = {"budgets": rows, "equivalence_margin_eV": margin, "p1_descriptors": p1_descriptors(root)}
    print(f"{'B':>4} {'S':>6} {'D':>6} {'P':>6} {'D-P':>16} {'P-S':>16}  outcome")
    for b, r in rows.items():
        m = r["mae"]
        print(f"{b:>4} {m['S']['mean']:6.3f} {m['D']['mean']:6.3f} {m['P']['mean']:6.3f} "
              f"{r['D_minus_P']['mean']:+.3f} [{r['D_minus_P']['ci95'][0]:+.3f},{r['D_minus_P']['ci95'][1]:+.3f}] "
              f"{r['P_minus_S']['mean']:+.3f} [{r['P_minus_S']['ci95'][0]:+.3f},{r['P_minus_S']['ci95'][1]:+.3f}]  "
              f"{r['P_minus_S']['outcome']}")
    if not a.no_write and rows:
        from dftgnn.io.results import write_result

        print(write_result("c3a_v2", res, config=cfg))


if __name__ == "__main__":
    main()
