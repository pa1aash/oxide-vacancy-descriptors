"""Screening demonstration.

    python scripts/screening.py [--no-write]

For each candidate oxide of ``screening.candidates`` (all polymorphs: every universe host whose reduced
formula matches), averaged over the seed-ensemble predictions of every outer resample in which the host was a
test host (at least ``screening.min_resamples_tested``), against the DFT target. The v2 backbone's S-v2
prediction is given alongside. Candidates absent from universe v1 or never tested are listed with the reason.
A demonstration, not a claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnostics_sweep as D

from dftgnn.config import load_config
from dftgnn.data.universe import read_universe


def oos(subdir: str, budget: int) -> pd.DataFrame:
    ix = D.run_index(subdir)
    ix = ix[ix.split.str.startswith("outer") & (ix.model == "S") & (ix.B == budget)]
    frames = [D.ensemble(g).assign(r=r) for r, g in ix.groupby("r")]
    allp = pd.concat(frames, ignore_index=True)
    return allp.groupby(["site_id", "host_id", "y_true"]).agg(pred=("y_pred", "mean"), n_resamples=("r", "nunique")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    sc = cfg.screening
    uni = read_universe(cfg)
    v1, v2 = oos("", sc.budget), oos("v2_sweep", sc.budget)
    rows, missing = [], {}
    for cand in sc.candidates:
        hosts = uni[uni.reduced_formula == cand]
        if hosts.empty:
            missing[cand] = "not in universe v1"
            continue
        m = hosts[["site_id", "host_id", "formula"]].merge(v1, on=["site_id", "host_id"], how="left")
        m = m.merge(v2[["site_id", "pred"]].rename(columns={"pred": "pred_v2"}), on="site_id", how="left")
        m = m[m.n_resamples >= sc.min_resamples_tested]
        if m.empty:
            missing[cand] = "never a test host in the outer resamples"
            continue
        for _, r in m.iterrows():
            rows.append({"candidate": cand, "host_id": r.host_id, "site_id": r.site_id, "dft_eV": r.y_true,
                         "S_eV": r.pred, "abs_err_eV": abs(r.pred - r.y_true), "S_v2_eV": r.pred_v2,
                         "n_resamples": int(r.n_resamples)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print("missing:", missing)
    print(f"MAE over {len(t)} sites of {t.candidate.nunique()} candidates: S {t.abs_err_eV.mean():.3f} eV, "
          f"S-v2 {(t.S_v2_eV - t.dft_eV).abs().mean():.3f} eV")
    if not a.no_write:
        from dftgnn.io.results import write_result

        print(write_result("screening_c5", {"sites": t.to_dict("records"), "missing": missing,
                                            "mae_S": float(t.abs_err_eV.mean()),
                                            "mae_S_v2": float((t.S_v2_eV - t.dft_eV).abs().mean()),
                                            "budget": sc.budget}, config=cfg))


if __name__ == "__main__":
    main()
