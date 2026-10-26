"""C3b: geometry robustness under MACE-MP-0 host geometry.

    python scripts/mlip_eval.py --artefacts /workspace/dft-gnn-v2 --budgets 654 200 --device cuda

Inputs: ``data/processed/mlip_inputs.json.gz``, ``results/mlip/relaxed_shard*.json.gz`` (this checkout), and the S-v2
and D-v2 sweep runs (``<artefacts>/results/v2_sweep``) and P-v2 evaluations (``<artefacts>/results/p_v2``) with their
checkpoints. Hosts enter if they are mapped and their relaxation converged; every condition uses the same sites.

Conditions: (i) DFT geometry (the verified graph store), (ii) MLIP geometry (relaxed unit cell tiled onto the
release supercell atom order), (iii) MLIP geometry rescaled isotropically to the DFT volume. Models are not
retrained; D keeps its DFT descriptors; P is P1-v2 then D-v2. For each (model, budget, resample) the
seed-ensemble MAE per condition; dMAE = condition minus (i), paired hierarchical bootstrap. Staged vs
end-to-end: dMAE_P - dMAE_S with the TOST of (delta 0.05 eV, 90% interval) and the four-outcome rule. Per
host: |error(ii)| - |error(i)| of S, D and P (ensemble over seeds and resamples in which the host was tested)
regressed on the three geometry components (OLS on standardised components; slope margin delta / IQR);
stratified by the d-electron class of the cations (pymatgen oxidation-state guess: d0, d10, other). Null rule:
median internal RMSD below the configured threshold and every |dMAE| interval including 0.
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dftgnn.config import load_config
from dftgnn.graphs import host_graph
from dftgnn.mlip import nrp as mlip
from dftgnn.models import StagedP, Standardiser
from dftgnn.stats.metrics import aggregate_delta, aggregate_resamples, host_stats
from dftgnn.train import RunSpec, Store, _predict, load_checkpoint, resolve_hosts

CONDS = ("dft", "mlip", "rescaled")


def _structures(args):
    hid, inp, relaxed, dft_cif_dict, cutoff = args
    from pymatgen.core import Structure

    dft = Structure.from_dict(dft_cif_dict)
    sc = mlip.supercell_from_unit(Structure.from_dict(relaxed), inp["matrix"], inp["perm"], inp["rotation"],
                                  inp["shift"])
    resc = mlip.rescale_to_volume(sc, dft)
    return hid, mlip.geometry(dft, sc), host_graph(sc, cutoff), host_graph(resc, cutoff)


def build_conditions(data: Store, cfg, processes: int) -> tuple[dict, dict, list[str]]:
    from pymatgen.core import Lattice, Structure

    inputs = json.loads(gzip.decompress((ROOT / "data/processed/mlip_inputs.json.gz").read_bytes()))["hosts"]
    relaxed = {}
    for f in sorted(glob.glob(str(ROOT / "results/mlip/relaxed_shard*.json.gz"))):
        relaxed.update(json.loads(gzip.decompress(Path(f).read_bytes()))["hosts"])
    ok = [h for h in data.meta["host_ids"] if h in inputs and h in relaxed and relaxed[h].get("converged")]
    hidx = {h: i for i, h in enumerate(data.meta["host_ids"])}
    jobs = []
    for h in ok:
        g = data.graphs[hidx[h]]
        dft = Structure(Lattice(g["lattice"].numpy()), g["z"].tolist(), g["pos"].numpy(), coords_are_cartesian=True)
        jobs.append((h, inputs[h], relaxed[h]["unit_cell"], dft.as_dict(), cfg.graph.cutoff_A))
    with get_context("spawn").Pool(processes) as pool:
        res = pool.map(_structures, jobs, chunksize=4)
    geo = {h: g for h, g, _, _ in res}
    graphs = {"mlip": copy.copy(data.graphs), "rescaled": copy.copy(data.graphs)}
    for h, _, gm, gr in res:
        graphs["mlip"][hidx[h]] = gm
        graphs["rescaled"][hidx[h]] = gr
    return geo, graphs, ok


def predict(pay: dict, data: Store, graphs: dict, keep: set, art: Path, device) -> pd.DataFrame:
    """Test-site predictions of one run under every condition, restricted to the hosts in ``keep``."""
    spec = RunSpec.from_dict(pay["spec"])
    cfg = load_config()
    te = data.positions([h for h in resolve_hosts(spec, cfg)["test"] if h in keep])
    if spec.model == "P":
        p1, c1 = load_checkpoint(art / "checkpoints" / f"{spec.p1_run}.pt", device)
        d, cd = load_checkpoint(art / "checkpoints" / f"{spec.d_run}.pt", device)
        net = StagedP(p1, d, Standardiser(c1["desc_mean"], c1["desc_sd"]),
                      Standardiser(cd["desc_mean"], cd["desc_sd"])).to(device).eval()
        ck, desc = cd, None
    else:
        net, ck = load_checkpoint(art / pay["checkpoint"]["path"], device)
        net = net.to(device)
        raw = data.sites["desc"].double()
        desc = ((raw - ck["desc_mean"].cpu().double()) / ck["desc_sd"].cpu().double()).float()
    t_mean, t_sd = float(ck["target_mean"].reshape(-1)[0]), float(ck["target_sd"].reshape(-1)[0])
    frame = pd.DataFrame({"site_id": [data.sites["site_id"][i] for i in te], "host_id": data.site_host[te],
                          "y_true": data.sites["target"][te].numpy()})
    for c in CONDS:
        dc = data if c == "dft" else copy.copy(data)
        if c != "dft":
            dc.graphs = graphs[c]
        out = _predict(net, dc, te, None, desc, spec.batch_size, device)
        frame[f"pred_{c}"] = out.double().numpy() * t_sd + t_mean
    return frame


def contrast(sets: list[list], coefs: list[float], n_boot: int, seed: int = 0) -> tuple[float, dict]:
    """sum_k c_k MAE_k (mean over resamples) and its hierarchical-bootstrap quantiles; every set shares the host
    rows of each resample, so all models are redrawn identically."""
    from dftgnn.stats.metrics import _draw_weights

    arr = [[np.asarray(x, float) for x in st] for st in sets]
    rr = len(arr[0])
    point = np.mean([sum(c * a[r][:, 1].sum() / a[r][:, 0].sum() for c, a in zip(coefs, arr, strict=True))
                     for r in range(rr)])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, rr, size=(n_boot, rr))
    vals = np.zeros((n_boot, rr))
    for k in range(rr):
        for j in range(rr):
            rows = np.nonzero(pick[:, k] == j)[0]
            if rows.size:
                w = _draw_weights(rng, arr[0][j].shape[0], rows.size)
                vals[rows, k] = sum(c * (w @ a[j][:, 1]) / (w @ a[j][:, 0]) for c, a in zip(coefs, arr, strict=True))
    boot = vals.mean(1)
    return float(point), {q: float(np.quantile(boot, q)) for q in (0.025, 0.05, 0.95, 0.975)}


def tost(delta_block: dict, margin: float) -> str:
    lo95, hi95 = delta_block["ci95"]
    lo90, hi90 = delta_block["ci90"]
    within = -margin < lo90 and hi90 < margin
    if hi95 < 0:
        return "P better" + (" (within the equivalence margin)" if within else "")
    if lo95 > 0:
        return "P worse" + (" (within the equivalence margin)" if within else "")
    return "equivalent" if within else "inconclusive"


def d_class(z: list[int]) -> str:
    """d0, d10 or other from the most likely oxidation states of the host's cations (pymatgen guess); hosts
    without a d-block cation or without a guess count as ``other``."""
    from collections import Counter

    from pymatgen.core import Composition, Element

    comp = Composition({Element.from_Z(k): v for k, v in Counter(z).items()})
    try:
        guess = comp.oxi_state_guesses(max_sites=-50)
    except Exception:  # noqa: BLE001 - an unguessable composition is reported as other
        guess = []
    if not guess:
        return "other"
    counts = [Element(el).group - round(ox) for el, ox in guess[0].items() if Element(el).block == "d"]
    if not counts:
        return "other"
    if all(c == 0 for c in counts):
        return "d0"
    if all(c == 10 for c in counts):
        return "d10"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[654, 200])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--subdirs", nargs="+", default=["v2_sweep", "p_v2"], help="result subdirectories to evaluate")
    a = ap.parse_args()
    cfg = load_config()
    art = Path(a.artefacts)
    dev = torch.device(a.device)
    data = Store()
    geo, graphs, ok = build_conditions(data, cfg, a.processes)
    keep = set(ok)
    print(f"{len(ok)} hosts with MLIP geometry", flush=True)
    recs = []
    for sub in a.subdirs:
        for f in sorted(glob.glob(str(art / "results" / sub / "*.json"))):
            p = json.loads(Path(f).read_text())["payload"]
            if (p["spec"]["split"].startswith("outer") and p["spec"]["budget"] in a.budgets
                    and p["spec"].get("ablate") is None):
                recs.append(p)
    rows = []
    for p in recs:
        fr = predict(p, data, graphs, keep, art, dev)
        s = p["spec"]
        fr["model"] = {"S": "S", "D-state": "D", "D-late": "D", "P": "P"}[s["model"]]
        fr["B"], fr["r"], fr["seed"] = s["budget"], s["r"], s["seed"]
        rows.append(fr)
        print(f"{s['model']} B={s['budget']} r={s['r']} seed={s['seed']}", flush=True)
    allp = pd.concat(rows, ignore_index=True)
    ens = allp.groupby(["model", "B", "r", "site_id", "host_id", "y_true"], as_index=False)[
        [f"pred_{c}" for c in CONDS]].mean()
    margin, n_boot = cfg.analysis.equivalence_margin_eV, cfg.analysis.bootstrap.draws
    out = {"n_hosts_mlip": len(ok), "geometry": {}, "dMAE": {}, "staging": {}, "per_host": {}}
    g = pd.DataFrame(geo).T
    out["geometry"] = {k: {"median": float(g[k].median()), "iqr": float(g[k].quantile(0.75) - g[k].quantile(0.25)),
                           "p95": float(g[k].quantile(0.95))} for k in g.columns}
    for b in a.budgets:
        for m in ("S", "D", "P"):
            e = ens[(ens.model == m) & (ens.B == b)]
            rs = sorted(e.r.unique())
            st = {c: [host_stats(e[e.r == r].y_true, e[e.r == r][f"pred_{c}"], e[e.r == r].host_id) for r in rs]
                  for c in CONDS}
            out["dMAE"][f"{m}|{b}"] = {"mae": {c: aggregate_resamples(st[c], n_boot=n_boot)["mae"]["mean"] for c in CONDS}}
            for c in ("mlip", "rescaled"):
                dd = aggregate_delta(st[c], st["dft"], n_boot=n_boot)["mae"]
                out["dMAE"][f"{m}|{b}"][c] = {"mean": dd["mean"], "ci95": dd["ci"]}
        for c in ("mlip", "rescaled"):        # staged vs end-to-end degradation, paired on the same hosts
            sets = []
            for m, cc in (("P", c), ("P", "dft"), ("S", c), ("S", "dft")):
                e = ens[(ens.model == m) & (ens.B == b)].sort_values(["r", "site_id"])
                sets.append([host_stats(e[e.r == r].y_true, e[e.r == r][f"pred_{cc}"], e[e.r == r].host_id)
                             for r in sorted(e.r.unique())])
            point, q = contrast(sets, [1.0, -1.0, -1.0, 1.0], n_boot)
            blk = {"dMAE_P_minus_dMAE_S": point, "ci95": [q[0.025], q[0.975]],
                   "ci90": [q[0.05], q[0.95]]}
            blk["outcome"] = tost(blk, margin)
            out["staging"][f"{b}|{c}"] = blk
    # per host: |error| change under MLIP geometry, averaged over the resamples in which the host was tested
    ens["d_abs"] = (ens.pred_mlip - ens.y_true).abs() - (ens.pred_dft - ens.y_true).abs()
    ph = ens.groupby(["model", "B", "host_id"]).d_abs.mean().reset_index()
    dcls = {h: d_class(data.graphs[i]["z"].tolist()) for i, h in enumerate(data.meta["host_ids"]) if h in keep}
    for (m, b), f in ph.groupby(["model", "B"]):
        x = g.loc[f.host_id, ["isotropic_strain", "deviatoric_strain", "internal_rmsd_A"]].astype(float)
        xs = (x - x.mean()) / x.std()
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(xs)), xs.to_numpy()], f.d_abs.abs().to_numpy(), rcond=None)
        iqr = (x.quantile(0.75) - x.quantile(0.25))
        out["per_host"][f"{m}|{b}"] = {
            "slopes_per_sd": dict(zip(x.columns, coef[1:].tolist(), strict=True)),
            "slope_margin_per_iqr_eV": {k: margin for k in x.columns}, "iqr": iqr.to_dict(), "n_hosts": len(f),
            "by_d_class": {k: {"mean_abs_d_error_eV": float(v.d_abs.abs().mean()), "n_hosts": len(v)}
                           for k, v in f.assign(cls=f.host_id.map(dcls)).groupby("cls")}}
    out["null_rule"] = {"median_internal_rmsd_A": out["geometry"]["internal_rmsd_A"]["median"],
                        "threshold": cfg.mlip.null_rule.median_internal_rmsd_A_below}
    print(json.dumps({k: out[k] for k in ("n_hosts_mlip", "geometry", "dMAE", "staging")}, indent=1))
    if not a.no_write:
        from dftgnn.io.results import write_result

        print(write_result("mlip_eval_v2", out, config=cfg))


if __name__ == "__main__":
    main()
