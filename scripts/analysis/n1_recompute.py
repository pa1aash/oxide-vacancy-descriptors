"""GPU repeatability, recomputed on the Mac from the stored run files with our code (not
scripts/nrp/n1_compare.py).

    python scripts/analysis/n1_recompute.py            # writes results/nrp/n1_recompute.json via write_result

Inputs: every run record under results/nrp/n1/<class>/<node>/[pass2/] and its prediction parquet
(hash-checked), the original sweep runs of the same (model, split, B, seed) found through
``dftgnn.train.runindex`` (references), and the seed SD of single-seed MAE from results/g1_variance.json.
Everything is descriptive: no threshold is applied.

Quantities (eV): per run, the maximum and mean absolute per-site prediction difference against the reference and the
single-run MAE minus the reference MAE; per (class, node, pass, model, B), the seed-ensemble (mean of seeds 0-2 per
site) MAE minus the reference ensemble MAE and the maximum absolute ensemble prediction difference; bitwise identity
rates for the same-node repeat, across nodes of one class, and between classes; hardware seen; wall time and peak GPU
memory per class. The numbers are compared with the collaborator's results/nrp/nrp_n1_gpu_repeat.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dftgnn.io.results import write_result
from dftgnn.train import runindex

ROOT = Path(__file__).resolve().parents[2]
N1 = ROOT / "results" / "nrp" / "n1"
INTENDED = sorted((m, 0, b, s) for m in ("S", "D-state") for b in (654, 200) for s in (0, 1, 2))
SPEC_KEYS = ("model", "hp", "lr", "weight_decay", "batch_size", "split", "r", "budget", "seed", "max_epochs", "patience",
             "eval_test", "ablate", "exclude_sites", "hosts", "results_subdir")
FAULTY = "hcc-nrp-shor-c6017.unl.edu"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preds(path: Path) -> pd.Series:
    f = pd.read_parquet(path)
    return f.set_index("site_id")[["y_true", "y_pred"]].sort_index()


def load_runs() -> list[dict]:
    out = []
    for f in sorted(N1.rglob("*.json")):
        rel = f.relative_to(N1).parts
        cls, node = rel[0], rel[1]
        pas = "pass2" if "pass2" in rel else "pass1"
        pay = json.loads(f.read_text())["payload"]
        pp = ROOT / pay["predictions"]["path"]
        if sha(pp) != pay["predictions"]["sha256"]:
            raise SystemExit(f"{pp}: sha256 differs from its record")
        out.append({"class": cls, "node": node, "pass": pas, "payload": pay, "pred": preds(pp),
                    "pred_sha256": pay["predictions"]["sha256"]})
    return out


def main() -> None:
    runs = load_runs()
    prim = runindex.primary(runindex.load())
    gv = json.loads((ROOT / "results" / "g1_variance.json").read_text())["payload"]["models"]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["class"], r["node"], r["pass"])].append(r)

    # spec check: each (class, node, pass) holds exactly the intended 12 cells, with the sweep's spec
    spec_check = {}
    for g, rs in sorted(groups.items()):
        cells = sorted((r["payload"]["spec"]["model"], r["payload"]["spec"]["r"], r["payload"]["spec"]["budget"],
                        r["payload"]["spec"]["seed"]) for r in rs)
        mism = []
        for r in rs:
            sp = r["payload"]["spec"]
            ref = prim[(sp["model"], sp["split"], sp["budget"], sp["seed"])]["spec"]
            bad = [k for k in SPEC_KEYS if sp.get(k) != ref.get(k)]
            if bad:
                mism.append({"run_id": r["payload"]["run_id"], "keys": bad})
        spec_check["|".join(g)] = {"cells_match_intended": cells == [tuple(c) for c in INTENDED], "n": len(cells),
                                   "spec_mismatches_vs_sweep": mism}

    # per run against the reference
    per_run = []
    refp = {}
    for r in runs:
        sp = r["payload"]["spec"]
        key = (sp["model"], sp["split"], sp["budget"], sp["seed"])
        ref = prim[key]
        if key not in refp:
            refp[key] = preds(ROOT / ref["predictions"]["path"])
        rp = refp[key]
        a = r["pred"]
        if not a.index.equals(rp.index) or not np.array_equal(a.y_true.values, rp.y_true.values):
            raise SystemExit(f"{r['payload']['run_id']}: test sites differ from the reference")
        d = (a.y_pred - rp.y_pred).abs()
        per_run.append({"class": r["class"], "node": r["node"], "pass": r["pass"], "model": sp["model"],
                        "B": sp["budget"], "seed": sp["seed"], "run_id": r["payload"]["run_id"],
                        "ref_run_id": ref["run_id"], "bitwise_equal_ref": bool(np.array_equal(a.y_pred.values,
                                                                                               rp.y_pred.values)),
                        "max_abs_diff_eV": float(d.max()), "mean_abs_diff_eV": float(d.mean()),
                        "mae": float((a.y_pred - a.y_true).abs().mean()),
                        "mae_minus_ref": float((a.y_pred - a.y_true).abs().mean() - (rp.y_pred - rp.y_true).abs().mean()),
                        "best_epoch": r["payload"]["best_epoch"], "epochs_run": r["payload"]["epochs_run"],
                        "wall_s": r["payload"]["wall_time_s"],
                        "peak_gpu_gb": r["payload"]["peak_memory_bytes"] / 2**30,
                        "gpu_model": r["payload"]["hardware"]["gpu_model"],
                        "gpu_uuid": r["payload"]["hardware"]["gpu_uuid"],
                        "pod_hostname": r["payload"]["hardware"]["pod_hostname"]})

    # seed ensembles
    ens = []
    for (cls, node, pas), rs in sorted(groups.items()):
        for m, b in itertools.product(("S", "D-state"), (654, 200)):
            sel = [r for r in rs if r["payload"]["spec"]["model"] == m and r["payload"]["spec"]["budget"] == b]
            if len(sel) != 3:
                continue
            e = np.mean([r["pred"].y_pred.values for r in sel], axis=0)
            refs = [refp[(m, "outer_r0", b, s)] for s in (0, 1, 2)]
            er = np.mean([x.y_pred.values for x in refs], axis=0)
            y = refs[0].y_true.values
            gm = gv["D" if m == "D-state" else "S"][str(b)]
            ens.append({"class": cls, "node": node, "pass": pas, "model": m, "B": b,
                        "ensemble_mae": float(np.abs(e - y).mean()), "ref_ensemble_mae": float(np.abs(er - y).mean()),
                        "ensemble_mae_minus_ref": float(np.abs(e - y).mean() - np.abs(er - y).mean()),
                        "max_abs_ensemble_pred_diff_eV": float(np.abs(e - er).max()),
                        "seed_sd_resample0": gm["seed_sd_per_resample"][0],
                        "seed_sd_mean_over_resamples": gm["seed_sd_mean_over_resamples"]})

    # identity rates
    def pairs(ga, gb):
        a = {(r["payload"]["spec"]["model"], r["payload"]["spec"]["budget"], r["payload"]["spec"]["seed"]): r
             for r in groups[ga]}
        b = {(r["payload"]["spec"]["model"], r["payload"]["spec"]["budget"], r["payload"]["spec"]["seed"]): r
             for r in groups[gb]}
        ks = sorted(set(a) & set(b))
        same = [np.array_equal(a[k]["pred"].y_pred.values, b[k]["pred"].y_pred.values) for k in ks]
        mx = max(float((a[k]["pred"].y_pred - b[k]["pred"].y_pred).abs().max()) for k in ks)
        return {"a": "|".join(ga), "b": "|".join(gb), "n": len(ks), "bitwise_identical": int(sum(same)),
                "max_abs_diff_eV": mx}

    same_node = [pairs(ga, (ga[0], ga[1], "pass2")) for ga in groups if ga[2] == "pass1"
                 and (ga[0], ga[1], "pass2") in groups]
    by_class = defaultdict(list)
    for g in groups:
        if g[2] == "pass1":
            by_class[g[0]].append(g)
    cross_node = [pairs(x, y) for gs in by_class.values() for x, y in itertools.combinations(sorted(gs), 2)]
    first = {c: min(gs) for c, gs in by_class.items()}
    cross_class = [pairs(first[x], first[y]) for x, y in itertools.combinations(sorted(first), 2)]

    # per class summary (first node, pass 1; A10 nodes are bitwise identical, see cross_node)
    per_class = []
    for c, g in sorted(first.items()):
        for m, b in itertools.product(("S", "D-state"), (654, 200)):
            pr = [x for x in per_run if (x["class"], x["node"], x["pass"]) == g and x["model"] == m and x["B"] == b]
            en = next(x for x in ens if (x["class"], x["node"], x["pass"]) == g and x["model"] == m and x["B"] == b)
            per_class.append({"class": c, "model": m, "B": b, "max_abs_pred_diff_single_run_eV":
                              max(x["max_abs_diff_eV"] for x in pr),
                              "max_abs_ensemble_pred_diff_eV": en["max_abs_ensemble_pred_diff_eV"],
                              "ensemble_mae_minus_ref_eV": en["ensemble_mae_minus_ref"],
                              "seed_sd_resample0": en["seed_sd_resample0"],
                              "seed_sd_mean_over_resamples": en["seed_sd_mean_over_resamples"],
                              "abs_ens_diff_gt_seed_sd_resample0": abs(en["ensemble_mae_minus_ref"]) > en["seed_sd_resample0"],
                              "mean_wall_s": float(np.mean([x["wall_s"] for x in pr])),
                              "max_peak_gpu_gb": max(x["peak_gpu_gb"] for x in pr)})

    hw = sorted({(r["class"], r["node"], r["payload"]["hardware"]["gpu_model"], r["payload"]["hardware"]["driver"],
                  r["payload"]["hardware"]["gpu_memory"]) for r in runs})
    pods = sorted({r["payload"]["hardware"]["pod_hostname"] for r in runs})

    # comparison with the collaborator's summary
    theirs = json.loads((ROOT / "results" / "nrp" / "nrp_n1_gpu_repeat.json").read_text())["payload"]
    diffs = []
    tk = {(e["class"], e["node"], e["pass"], e["model"], e["B"]): e for e in theirs["ensemble"]}
    for e in ens:
        t = tk.get((e["class"], e["node"], e["pass"], e["model"], e["B"]))
        if t is None:
            diffs.append({"what": "ensemble entry missing in theirs", "key": [e["class"], e["node"], e["pass"],
                                                                                 e["model"], e["B"]]})
            continue
        for k in ("ensemble_mae_minus_ref", "max_abs_ensemble_pred_diff_eV", "ensemble_mae", "ref_ensemble_mae"):
            if abs(t[k] - e[k]) > 1e-9:
                diffs.append({"what": k, "key": [e["class"], e["node"], e["pass"], e["model"], e["B"]],
                              "ours": e[k], "theirs": t[k]})
    tr = {(x["run_id"], x["class"], x["node"], x["pass"]): x for x in theirs["per_run"]}
    for x in per_run:
        t = tr.get((x["run_id"], x["class"], x["node"], x["pass"]))
        if t is None:
            diffs.append({"what": "per-run entry missing in theirs", "run_id": x["run_id"]})
            continue
        for k in ("max_abs_diff_eV", "mean_abs_diff_eV", "mae"):
            if abs(t[k] - x[k]) > 1e-9:
                diffs.append({"what": k, "run_id": x["run_id"], "ours": x[k], "theirs": t[k]})
        if t["ref_run_id"] != x["ref_run_id"] or t["bitwise_equal_ref"] != x["bitwise_equal_ref"]:
            diffs.append({"what": "reference or bitwise flag", "run_id": x["run_id"]})
    if len(theirs["per_run"]) != len(per_run):
        diffs.append({"what": "run count", "ours": len(per_run), "theirs": len(theirs["per_run"])})

    payload = {"n_runs": len(runs), "spec_check": spec_check, "intended_cells": INTENDED,
               "same_node_repeat": same_node, "cross_node_same_class": cross_node, "cross_class": cross_class,
               "per_class": per_class, "ensemble": ens, "per_run": per_run,
               "hardware": [dict(zip(("class", "node", "gpu_model", "driver", "gpu_memory"), h, strict=True)) for h in hw],
               "pod_hostnames": pods, "faulty_node": FAULTY,
               "runs_on_faulty_node": sum(r["node"] == FAULTY for r in runs),
               "classes_requested_by_collaborator": theirs.get("classes_requested"),
               "classes_without_runs": theirs.get("classes_without_runs"),
               "differences_vs_collaborator_summary": diffs,
               "note": "descriptive; recomputed with our code from the stored N1 files; no threshold applied"}
    path = write_result("n1_recompute", payload, results_dir=ROOT / "results" / "nrp")
    print(f"wrote {path}")
    print(f"runs {len(runs)}; spec check all match: {all(v['cells_match_intended'] and not v['spec_mismatches_vs_sweep'] for v in spec_check.values())}")
    for s in same_node + cross_node + cross_class:
        print(f"  {s['a']} vs {s['b']}: {s['bitwise_identical']}/{s['n']} bitwise, max |d| {s['max_abs_diff_eV']:.4f}")
    print("class model B | max single | max ens | dMAE_ens | seed SD r0 (mean) | wall s | peak GB")
    for p in per_class:
        print(f"  {p['class']:17s} {p['model']:7s} {p['B']:3d} | {p['max_abs_pred_diff_single_run_eV']:.3f} | "
              f"{p['max_abs_ensemble_pred_diff_eV']:.3f} | {p['ensemble_mae_minus_ref_eV']:+.3f} | "
              f"{p['seed_sd_resample0']:.3f} ({p['seed_sd_mean_over_resamples']:.3f}) | {p['mean_wall_s']:.0f} | "
              f"{p['max_peak_gpu_gb']:.2f}")
    print("hardware:", hw)
    print("pods:", pods)
    print(f"runs on {FAULTY}: {payload['runs_on_faulty_node']}; differences vs collaborator summary: {len(diffs)}")
    for d in diffs[:20]:
        print("  ", d)


if __name__ == "__main__":
    main()
