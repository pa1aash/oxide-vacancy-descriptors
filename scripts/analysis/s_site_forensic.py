"""S10b: read-only forensic of S's within-host site resolution (implementation check, no training).

Checkpoints: resample 0, seed 0, models S and D-state, budgets 25-654 (12 runs), loaded with
``dftgnn.train.load_checkpoint`` on the CPU in eval mode. Hosts: 30 resample-0 test hosts with >= 2 labelled
sites, drawn once with ``numpy.random.default_rng(20261008)`` from the sorted eligible host ids.

(a) flag sweep: the vacancy (flag and readout index) is put on every O atom of the host supercell in turn;
    SD (ddof = 1) of the predictions across all O atoms and across the labelled sites; prediction with the
    flag zeroed (readout index kept on each labelled site). Medians across hosts; target SD within host.
    D-state: labelled sites carry their own site descriptors; unlabelled O atoms carry the host mean of the
    labelled sites' site descriptors (the host descriptors are constant within a host).
(b) readout decomposition: first head layer split into [vacancy node | pooled | global state] blocks;
    Frobenius norm / sqrt(block width), activation SD across the labelled sites (pooled over hosts) and the
    mean within-host activation SD, and their products. D-state: first node-encoder layer norms of the flag
    column and of the site-descriptor columns (the descriptors are merged at the node input).
(c) node level, one host: SD across O atoms of the vacancy-node embedding after the last block, with the
    flag on that atom versus the flag off.
(d) hyperparameters and training bookkeeping of each checkpoint from its result JSON.
(e) init control: an untrained model with the checkpoint's hyperparameters built after
    ``seed_everything(seed)`` (as in training); (a) repeated; outputs mapped to eV with the checkpoint's
    target scaling.
(f) D-state control: (a) with every O atom carrying the host-mean site descriptors.

    python scripts/analysis/s_site_forensic.py [--only 'S|25' ...] [--threads 2]   # compute, cache per checkpoint
    python scripts/analysis/s_site_forensic.py --write --verdict "<verdict>" --reason "<text>"
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from dftgnn.graphs import site_data
from dftgnn.models import HParams, build_model
from dftgnn.split import load_split
from dftgnn.train import Store, load_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "s10b" / "forensic.pkl"
SYNTH = ROOT / ".cache" / "s10b" / "synthetic.json"
BUDGETS = (25, 50, 100, 200, 400, 654)
MODELS = ("S", "D-state")
N_HOSTS, HOST_SEED, BATCH = 30, 20261008, 24
VERDICTS = ("A: implementation error found",
            "B: no implementation error; flag effect is a learned outcome at small budgets")


def run_records() -> dict:
    out = {}
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            p = json.loads(f.read_text()).get("payload")
        except ValueError:
            continue
        if not isinstance(p, dict) or "spec" not in p:
            continue
        s = p["spec"]
        if (s.get("model") in MODELS and s.get("split") == "outer_r0" and s.get("seed") == 0
                and s.get("results_subdir") is None and not p.get("smoke")):
            out[(s["model"], s["budget"])] = p
    assert len(out) == 12, sorted(out)
    return out


class Host:
    def __init__(self, st: Store, h: str):
        self.id = h
        self.g = st.graphs[st.meta["host_ids"].index(h)]
        self.pos = np.nonzero(st.site_host == h)[0]
        self.vac = [int(st.sites["vacancy_atom_index"][p]) for p in self.pos]
        self.target = st.sites["target"][self.pos].numpy()
        self.oxy = torch.nonzero(self.g["z"] == 8).flatten().tolist()
        assert set(self.vac) <= set(self.oxy)
        self.desc = st.sites["desc"][self.pos].double()          # raw, (n_sites, 22)


def predict(net, ck, host: Host, atoms, desc_rows=None, flag_off=False) -> np.ndarray:
    """Predictions in eV with the vacancy on each of ``atoms``; ``desc_rows`` raw (len(atoms), 22) for D."""
    out = []
    for k in range(0, len(atoms), BATCH):
        items = []
        for j, a in enumerate(atoms[k:k + BATCH]):
            kw = {}
            if desc_rows is not None:
                z = ((desc_rows[k + j] - ck["desc_mean"]) / ck["desc_sd"]).float()
                kw = {"desc_host": z[: net.n_host], "desc_site": z[net.n_host:]}
            items.append(site_data(host.g, int(a), **kw))
        b = Batch.from_data_list(items)
        if flag_off:
            b.vac_flag.zero_()
        with torch.no_grad():
            y = net(b).double()
        out.append((y * ck["target_sd"].double() + ck["target_mean"].double()).view(-1).numpy())
    return np.concatenate(out)


LABELLED_ONLY = False      # --labelled-only: skip the all-O sweep (fast first pass; sd_all_O = None)


def sweep(net, ck, host: Host, kind: str, mode: str = "own") -> dict:
    """(a) for one host. ``mode``: 'own' site descriptors on labelled sites, 'hostmean' everywhere."""
    hm = host.desc.mean(0)
    rows_all = rows_lab = None
    if kind == "D-state":
        lab = dict(zip(host.vac, host.desc, strict=True))
        rows_all = torch.stack([lab[a] if (mode == "own" and a in lab) else hm for a in host.oxy])
        rows_lab = torch.stack([host.desc[i] if mode == "own" else hm for i in range(len(host.vac))])
    p_all = None if LABELLED_ONLY else predict(net, ck, host, host.oxy, rows_all)
    p_lab = predict(net, ck, host, host.vac, rows_lab)
    p_off = predict(net, ck, host, host.vac, rows_lab, flag_off=True)
    return {"sd_all_O": None if p_all is None else float(np.std(p_all, ddof=1)), "sd_labelled": float(np.std(p_lab, ddof=1)),
            "sd_flag_off_labelled": float(np.std(p_off, ddof=1)),
            "mean_abs_on_minus_off": float(np.abs(p_lab - p_off).mean()),
            "target_sd": float(np.std(host.target, ddof=1)), "n_O": len(host.oxy), "n_labelled": len(host.vac),
            "pred_labelled": p_lab.tolist(), "pred_flag_off": p_off.tolist()}


def readout_blocks(net, ck, hosts: list[Host], kind: str) -> dict:
    h2 = net.hp.hidden_width // 2
    pooled = 2 * h2 if net.hp.pooling == "set2set" else h2
    sl = {"vacancy_node": slice(0, h2), "pooled": slice(h2, h2 + pooled), "global_state": slice(h2 + pooled, None)}
    W = net.head.layers[0].weight.detach().double()
    vecs, grp = [], []
    for i, host in enumerate(hosts):
        items = []
        for p, a in zip(range(len(host.vac)), host.vac, strict=True):
            kw = {}
            if kind == "D-state":
                z = ((host.desc[p] - ck["desc_mean"]) / ck["desc_sd"]).float()
                kw = {"desc_host": z[: net.n_host], "desc_site": z[net.n_host:]}
            items.append(site_data(host.g, a, **kw))
        with torch.no_grad():
            vecs.append(net.readout_vector(Batch.from_data_list(items)).double())
        grp += [i] * len(items)
    V = torch.cat(vecs).numpy()
    grp = np.array(grp)
    out = {}
    for name, s in sl.items():
        w = W[:, s]
        norm = float(w.norm() / np.sqrt(w.shape[1]))
        act = float(V[:, s].std(0, ddof=1).mean())
        within = float(np.mean([V[grp == i][:, s].std(0, ddof=1).mean() for i in np.unique(grp)]))
        out[name] = {"width": int(w.shape[1]), "weight_norm_per_sqrt_width": norm, "activation_sd": act,
                     "within_host_activation_sd": within, "contribution_scale": norm * act,
                     "within_host_contribution_scale": norm * within}
    if kind == "D-state":
        w0 = net.node_encoder.layers[0].weight.detach().double()      # columns: embedding(16) | flag | site desc
        out["node_encoder_input"] = {
            "flag_column_norm": float(w0[:, 16].norm()),
            "site_descriptor_columns_norm_per_sqrt_width": float(w0[:, 17:].norm() / np.sqrt(w0.shape[1] - 17)),
            "element_embedding_columns_norm_per_sqrt_width": float(w0[:, :16].norm() / 4.0),
            "note": "site descriptors enter as flag x descriptor at the vacancy node input; not separable in the readout"}
    return out


def node_level(net, ck, host: Host, kind: str) -> dict:
    """(c) SD across O atoms of the vacancy-node embedding, flag on that atom vs flag off."""
    hm = ((host.desc.mean(0) - ck["desc_mean"]) / ck["desc_sd"]).float()
    kw = {"desc_host": hm[: net.n_host], "desc_site": hm[net.n_host:]} if kind == "D-state" else {}
    emb = {"on": [], "off": []}
    for k in range(0, len(host.oxy), BATCH):
        atoms = host.oxy[k:k + BATCH]
        for state in ("on", "off"):
            b = Batch.from_data_list([site_data(host.g, a, **kw) for a in atoms])
            if state == "off":
                b.vac_flag.zero_()
            with torch.no_grad():
                h2 = net.hp.hidden_width // 2
                emb[state].append(net.readout_vector(b)[:, :h2].double())
    on, off = torch.cat(emb["on"]), torch.cat(emb["off"])
    return {"host": host.id, "n_O": len(host.oxy),
            "sd_across_O_flag_on": float(on.std(0).mean()), "sd_across_O_flag_off": float(off.std(0).mean()),
            "mean_abs_on_minus_off": float((on - off).abs().mean())}


def init_model(ck) -> torch.nn.Module:
    mc = ck["model_config"]
    seed_everything(int(ck["spec"]["seed"]))
    net = build_model(mc["kind"], HParams(**mc["hp"]), n_host=mc["n_host"], n_site=mc["n_site"],
                      cutoff=mc["cutoff"]).eval()
    torch.use_deterministic_algorithms(False)   # set by seed_everything; CPU inference is deterministic anyway
    return net


def med(rows: list[dict], key: str) -> float | None:
    v = [r[key] for r in rows]
    return None if any(x is None for x in v) else float(np.median(v))


def summary(rows: list[dict]) -> dict:
    keys = ("sd_all_O", "sd_labelled", "sd_flag_off_labelled", "mean_abs_on_minus_off", "target_sd")
    return {f"median_{k}": med(rows, k) for k in keys}


def synthetic() -> dict:
    """Unit-test host (2x2x2 MgO, one Mg -> Ca) at init: near-Ca O vs farthest O, with the near O displaced."""
    from pymatgen.core import Lattice, Structure

    from dftgnn.graphs import host_graph

    out = {}
    for disp in (0.0, 0.2, 0.5):
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s.make_supercell([2, 2, 2])
        s.replace(0, "Ca")
        o = [i for i, x in enumerate(s) if x.specie.symbol == "O"]
        d = [s.lattice.get_distance_and_image(s[i].frac_coords, s[0].frac_coords)[0] for i in o]
        near, far = o[int(np.argmin(d))], o[int(np.argmax(d))]
        if disp:
            s.translate_sites([near], [disp, 0.0, 0.0], frac_coords=False)
        g = host_graph(s, 5.0)
        for pool in ("set2set", "mean"):
            for blocks in (2, 4):
                for hid in (64, 128):
                    torch.manual_seed(0)
                    net = build_model("S", HParams(hidden_width=hid, megnet_blocks=blocks, pooling=pool)).eval()
                    b = Batch.from_data_list([site_data(g, near), site_data(g, far)])
                    bz = Batch.from_data_list([site_data(g, near), site_data(g, far)])
                    bz.vac_flag.zero_()
                    with torch.no_grad():
                        y, v, yz, vz = net(b), net.readout_vector(b), net(bz), net.readout_vector(bz)
                    h2 = hid // 2
                    out[f"disp{disp}|{pool}|{blocks}|{hid}"] = {
                        "abs_out_near_minus_far": float((y[0] - y[1]).abs()),
                        "max_abs_vacancy_block_near_minus_far": float((v[0, :h2] - v[1, :h2]).abs().max()),
                        "max_abs_vacancy_block_near_minus_far_flag_off": float((vz[0, :h2] - vz[1, :h2]).abs().max()),
                        "max_abs_out_flag_on_minus_off": float((y - yz).abs().max()),
                        "max_abs_vacancy_block_flag_on_minus_off": float((v[:, :h2] - vz[:, :h2]).abs().max())}
    return {"units": "standardised model output (untrained)", "displacement_A_of_near_O": [0.0, 0.2, 0.5],
            "values": out}


def compute(only: list[str] | None, threads: int) -> None:
    """Compute the checkpoints in ``only`` (keys 'S|25' etc.; all if None), one cache file per checkpoint."""
    torch.set_num_threads(threads)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not SYNTH.exists():
        SYNTH.write_text(json.dumps(synthetic(), indent=1))
    st = Store()
    test = sorted(set(load_split("outer_r0")["test"]))
    elig = [h for h in test if (st.site_host == h).sum() >= 2]
    hosts_ids = sorted(np.random.default_rng(HOST_SEED).choice(elig, N_HOSTS, replace=False).tolist())
    hosts = [Host(st, h) for h in hosts_ids]
    recs = run_records()
    res = {"hosts": hosts_ids, "n_eligible_hosts": len(elig), "checkpoints": {}}
    for (kind, B), p in sorted(recs.items()):
        if only is not None and f"{kind}|{B}" not in only:
            continue
        t0 = time.time()
        path = ROOT / p["checkpoint"]["path"]
        net, ck = load_checkpoint(path, torch.device("cpu"))
        ck["desc_mean"], ck["desc_sd"] = ck["desc_mean"].double(), ck["desc_sd"].double()
        init = init_model(ck)
        r = {"run_id": p["run_id"], "checkpoint": p["checkpoint"]["path"],
             "hp": {**p["spec"]["hp"], "learning_rate": p["spec"]["lr"], "weight_decay": p["spec"]["weight_decay"],
                    "batch_size": p["spec"]["batch_size"]},
             "epochs_run": p["epochs_run"], "best_epoch": p["best_epoch"], "max_epochs": p["max_epochs"],
             "recorded_test_mae": p["metrics"]["mae"], "recorded_within_host_mae": p["metrics"]["within_host_mae"]}
        r["trained"] = [sweep(net, ck, h, kind) for h in hosts]
        r["init"] = [sweep(init, ck, h, kind) for h in hosts]
        if kind == "D-state":
            r["trained_hostmean_desc"] = [sweep(net, ck, h, kind, "hostmean") for h in hosts]
            r["init_hostmean_desc"] = [sweep(init, ck, h, kind, "hostmean") for h in hosts]
        r["readout"] = {"trained": readout_blocks(net, ck, hosts, kind), "init": readout_blocks(init, ck, hosts, kind)}
        r["node_level"] = {"trained": node_level(net, ck, hosts[0], kind), "init": node_level(init, ck, hosts[0], kind)}
        # check: the stored test predictions of this run are reproduced on the CPU for the labelled sites
        import pandas as pd
        pr = pd.read_parquet(ROOT / p["predictions"]["path"]).set_index("site_id").y_pred
        mine = np.concatenate([np.array(x["pred_labelled"]) for x in r["trained"]])
        ids = [st.sites["site_id"][i] for h in hosts for i in h.pos]
        r["max_abs_diff_vs_stored_gpu_predictions_eV"] = float(np.abs(mine - pr.loc[ids].to_numpy()).max())
        meta = {k: v for k, v in res.items() if k != "checkpoints"}
        (CACHE.parent / f"{'lab_' if LABELLED_ONLY else 'ck_'}{kind}_{B}.pkl").write_bytes(pickle.dumps({**meta, "key": f"{kind}|{B}", "r": r}))
        print(f"{kind} B={B}: {time.time() - t0:.0f}s  trained {summary(r['trained'])['median_sd_labelled']:.2e}  "
              f"init {summary(r['init'])['median_sd_labelled']:.2e}  "
              f"repro {r['max_abs_diff_vs_stored_gpu_predictions_eV']:.1e}", flush=True)


def merged(prefix: str) -> dict:
    parts = [pickle.loads(f.read_bytes()) for f in sorted(CACHE.parent.glob(f"{prefix}*.pkl"))]
    assert len(parts) == 12, f"{len(parts)} of 12 checkpoints computed"
    assert len({tuple(x["hosts"]) for x in parts}) == 1
    return {"hosts": parts[0]["hosts"], "n_eligible_hosts": parts[0]["n_eligible_hosts"],
            "checkpoints": {x["key"]: x["r"] for x in parts}}


def payload(res: dict, verdict: str, reason: str, labelled_only: bool) -> dict:
    tab = {}
    for key, r in res["checkpoints"].items():
        row = {k: r[k] for k in ("run_id", "hp", "epochs_run", "best_epoch", "max_epochs", "recorded_test_mae",
                                 "recorded_within_host_mae", "max_abs_diff_vs_stored_gpu_predictions_eV")}
        row["flag_sweep_trained"] = summary(r["trained"])
        row["flag_sweep_init"] = summary(r["init"])
        if "trained_hostmean_desc" in r:
            row["flag_sweep_trained_hostmean_site_desc"] = summary(r["trained_hostmean_desc"])
            row["flag_sweep_init_hostmean_site_desc"] = summary(r["init_hostmean_desc"])
        row["readout_blocks"] = r["readout"]
        row["node_level"] = r["node_level"]
        row["per_host_trained"] = [{k: v for k, v in x.items() if not k.startswith("pred")} for x in r["trained"]]
        row["per_host_init"] = [{k: v for k, v in x.items() if not k.startswith("pred")} for x in r["init"]]
        tab[key] = row
    synth = json.loads(SYNTH.read_text()) if SYNTH.exists() else None
    return {"verdict": verdict, "verdict_reason": reason, "hosts": res["hosts"],
            "pass": "labelled_sites_only" if labelled_only else "all_O_atoms",
            "pass_note": ("flag sweep over the labelled sites of each host only; the all-O-atom sweep "
                          "(sd_all_O) was not run in this pass (CPU time on battery power) and is None")
            if labelled_only else "flag sweep over every O atom and over the labelled sites",
            "host_selection": f"{N_HOSTS} of {res['n_eligible_hosts']} resample-0 test hosts with >= 2 labelled "
                              f"sites, numpy default_rng({HOST_SEED}).choice without replacement",
            "resample": 0, "seed": 0, "device": "cpu", "checkpoints": tab, "synthetic_unit_test_probe": synth,
            "sd_convention": "sample SD, ddof = 1; medians across the 30 hosts"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verdict")
    ap.add_argument("--reason", default="")
    ap.add_argument("--only", nargs="*", help="checkpoint keys such as 'S|25'; default all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--labelled-only", action="store_true")
    a = ap.parse_args()
    global LABELLED_ONLY
    LABELLED_ONLY = a.labelled_only
    if not a.write:
        compute(a.only, a.threads)
        return
    if not (a.verdict in VERDICTS or (a.verdict or "").startswith("INCONCLUSIVE: ")):
        raise SystemExit(f"verdict must be one of {VERDICTS} or 'INCONCLUSIVE: <reason>'")
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    res = merged("lab_" if a.labelled_only else "ck_")
    print(write_result("s_site_resolution_forensic", payload(res, a.verdict, a.reason, a.labelled_only), config=load_config()))


if __name__ == "__main__":
    main()
