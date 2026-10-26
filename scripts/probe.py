"""Latent probe on the S checkpoints of an arm.

    python scripts/probe.py --artefacts /workspace/dft-gnn-v2 --subdir v2_sweep --device cuda

Reads every S run with seed 0 from ``<artefacts>/results/<subdir>/`` and its checkpoint from
``<artefacts>/checkpoints/``; the graph store and splits come from this checkout (identical by manifest). For
each run: ``dftgnn.probe.probe`` on the trained encoder, the shuffled-label control and the untrained-encoder
control (same architecture, seed of the run). Rule (config ``probe.pass_rule``): the class-mean test R^2
increases with budget (Spearman rho over all runs > 0, with a 95% bootstrap interval over outer resamples
excluding 0) and exceeds both controls at every budget >= ``beats_controls_min_budget`` (mean over resamples
of trained minus control, 95% bootstrap interval over resamples above 0). Writes
``results/probe_<subdir>.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dftgnn import probe as P
from dftgnn.config import load_config
from dftgnn.models import HParams, build_model
from dftgnn.train import RunSpec, Store, load_checkpoint, resolve_hosts

KINDS = ("trained", "shuffled", "random")


def run_one(pay: dict, data: Store, art: Path, device, cfg) -> dict:
    spec = RunSpec.from_dict(pay["spec"])
    hosts = resolve_hosts(spec, cfg)
    tr, te = data.positions(hosts["train"]), data.positions(hosts["test"])
    desc = data.sites["desc"].numpy()
    mu, sd = desc[tr].mean(0), desc[tr].std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    y = (desc - mu) / sd
    g_tr = data.site_host[tr]
    net, _ = load_checkpoint(art / pay["checkpoint"]["path"], device)
    torch.manual_seed(spec.seed)
    fresh = build_model("S", HParams(**spec.hp), n_host=data.n_host, n_site=desc.shape[1] - data.n_host,
                        cutoff=data.meta["cutoff_A"])
    x = {"trained": P.features(net, data, tr, device), "random": P.features(fresh, data, tr, device)}
    xe = {"trained": P.features(net, data, te, device), "random": P.features(fresh, data, te, device)}
    out = {"run_id": pay["run_id"], "r": spec.r, "B": spec.budget}
    out["trained"] = P.probe(x["trained"], y[tr], g_tr, xe["trained"], y[te], folds=cfg.probe.alpha_cv.folds)
    out["shuffled"] = P.probe(x["trained"], P.shuffle_by_host(y[tr], g_tr, seed=spec.r), g_tr, xe["trained"], y[te],
                              folds=cfg.probe.alpha_cv.folds)
    out["random"] = P.probe(x["random"], y[tr], g_tr, xe["random"], y[te], folds=cfg.probe.alpha_cv.folds)
    return out


def class_means(runs: list[dict], n_host: int) -> None:
    for r in runs:
        for k in KINDS:
            v = np.array(r[k]["r2"], float)
            r[k]["class_mean"] = {"host": float(np.nanmean(v[:n_host])), "site": float(np.nanmean(v[n_host:])),
                                  "all": float(np.nanmean(v))}


def c4(runs: list[dict], cfg, n_boot: int = 2000) -> dict:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    rs = sorted({r["r"] for r in runs})
    out = {}
    for cls in ("host", "site", "all"):
        b = np.array([r["B"] for r in runs])
        v = np.array([r["trained"]["class_mean"][cls] for r in runs])
        rho = spearmanr(b, v).statistic
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(rs, size=len(rs))
            idx = np.concatenate([[i for i, r in enumerate(runs) if r["r"] == p] for p in pick])
            boots.append(spearmanr(b[idx], v[idx]).statistic)
        ci = [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]
        beats = {}
        for bud in sorted({r["B"] for r in runs}):
            if bud < cfg.probe.pass_rule.beats_controls_min_budget:
                continue
            for ctl in ("shuffled", "random"):
                diff = {r["r"]: r["trained"]["class_mean"][cls] - r[ctl]["class_mean"][cls] for r in runs if r["B"] == bud}
                d = np.array([diff[x] for x in rs if x in diff])
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
                beats[f"{bud}|{ctl}"] = {"mean": float(d.mean()), "ci": [float(np.quantile(bs, 0.025)),
                                                                          float(np.quantile(bs, 0.975))]}
        out[cls] = {"spearman_rho": float(rho), "rho_ci": ci, "trend_holds": ci[0] > 0,
                    "beats_controls": beats, "beats_all": all(x["ci"][0] > 0 for x in beats.values())}
        out[cls]["C4_holds"] = out[cls]["trend_holds"] and out[cls]["beats_all"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", required=True, help="checkout holding results/<subdir>/ and checkpoints/")
    ap.add_argument("--subdir", default="v2_sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N runs (smoke test; implies --no-write)")
    a = ap.parse_args()
    cfg = load_config()
    art = Path(a.artefacts)
    data = Store()
    pays = []
    for f in sorted(glob.glob(str(art / "results" / a.subdir / "*.json"))):
        p = json.loads(Path(f).read_text())["payload"]
        if p["spec"]["model"] == "S" and p["spec"]["seed"] in cfg.probe.checkpoints.seeds and p["spec"]["split"].startswith("outer"):
            pays.append(p)
    dev = torch.device(a.device)
    runs = []
    for p in sorted(pays, key=lambda p: (p["spec"]["budget"], p["spec"]["r"]))[: a.limit]:
        runs.append(run_one(p, data, art, dev, cfg))
        print(f"B={runs[-1]['B']} r={runs[-1]['r']}: mean R2 trained {np.nanmean(runs[-1]['trained']['r2']):.3f} "
              f"shuffled {np.nanmean(runs[-1]['shuffled']['r2']):.3f} random {np.nanmean(runs[-1]['random']['r2']):.3f}",
              flush=True)
    class_means(runs, data.n_host)
    res = {"subdir": a.subdir, "n_runs": len(runs), "runs": runs, "descriptors": data.sites["desc_host_names"]
           + data.sites["desc_site_names"], "C4": c4(runs, cfg), "alphas": P.ALPHAS.tolist()}
    print(json.dumps(res["C4"], indent=1))
    if not a.no_write and a.limit is None:
        from dftgnn.io.results import write_result

        print(write_result(f"probe_{a.subdir}", res, config=cfg))


if __name__ == "__main__":
    main()
