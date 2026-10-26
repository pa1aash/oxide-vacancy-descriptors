"""Readout-input embeddings for the latent probe.

Physics. The probe asks whether the structure-only model S learns internal features that linearly encode the
DFT descriptors D is given. The features are S's readout input vector, [vacancy-node embedding | pooled node
embeddings | global state] after the last MEGNet block, which is everything S's head sees.

For every (r, B): ``trained`` = the seed-0 S checkpoint; ``init<k>`` = an untrained S with that checkpoint's
hyperparameters built after ``torch.manual_seed(k)``. Sites: every site of the budget-B training hosts of
resample r (the validation carve-out hosts of seed 0 flagged "val", the rest "train") and every test site
("test").

Output per (r, B, kind): ``data/processed/embeddings_v1/r<r>_B<B>_<kind>.npz`` with site_ids, host_ids, split,
block_bounds (start of each block and the end), float32 embeddings; and ``<same>.json`` with shape and sha256.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dftgnn import infer
from dftgnn.config import load_config
from dftgnn.graphs.store import sha256_file
from dftgnn.models import build_model, hparams_for
from dftgnn.split import budget_train, load_split, val_split
from dftgnn.train import REPO_ROOT, val_seed

OUT_DIR = REPO_ROOT / "data" / "processed" / "embeddings_v1"
KINDS = ("trained", "init0", "init1", "init2")


def site_split(r: int, budget: int) -> dict[str, list[str]]:
    v = load_config().training.validation
    sp = load_split(f"outer_r{r}")
    tr, va = val_split(budget_train(sp, budget), frac=v.frac, min_hosts=v.min_hosts, seed=val_seed(r, budget, 0))
    return {"train": tr, "val": va, "test": sorted(sp["test"])}


def block_bounds(net) -> list[int]:
    h2 = net.hp.hidden_width // 2
    pooled = 2 * h2 if net.hp.pooling == "set2set" else h2
    b = [0, h2, h2 + pooled, h2 + pooled + h2]
    assert b[-1] == net.readout_dim
    return b


def model_for(kind: str, rec: dict, device=None):
    """(net, checkpoint dict); for init<k> an untrained net with the checkpoint's hyperparameters."""
    net, ck = infer.load(rec, device)
    if kind == "trained":
        return net, ck
    seed = int(kind.removeprefix("init"))
    mc = ck["model_config"]
    torch.manual_seed(seed)
    init = build_model(mc["kind"], hparams_for(mc["kind"], mc["hp"]), n_host=mc["n_host"], n_site=mc["n_site"],
                       cutoff=mc["cutoff"])
    return init.to(device or "cpu").eval(), ck


def extract(kind: str, rec: dict, sites: infer.Sites, r: int, budget: int, *, device=None,
            batch_size: int = 64) -> dict:
    net, ck = model_for(kind, rec, device)
    groups = site_split(r, budget)
    pos, flag = [], []
    for name in ("train", "val", "test"):
        p = sites.positions(groups[name])
        pos.append(p)
        flag += [name] * len(p)
    pos = np.concatenate(pos)
    hosts = sorted(set(sites.host_id[pos]))
    emb = infer.predict(net, ck, sites, sites.dft_graphs(hosts), pos, batch_size=batch_size, device=device,
                        what="readout").astype(np.float32)
    return {"site_ids": np.array([sites.site_id[p] for p in pos]), "host_ids": np.array(sites.host_id[pos]),
            "split": np.array(flag), "block_bounds": np.array(block_bounds(net)),
            "block_names": np.array(["vacancy_node", "pooled_nodes", "global_state"]), "embeddings": emb}


def write(arrays: dict, name: str, meta: dict, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.npz"
    np.savez(p, **arrays)
    side = {**meta, "file": p.name, "sha256": sha256_file(p), "shape": list(arrays["embeddings"].shape),
            "n_by_split": {s: int((arrays["split"] == s).sum()) for s in ("train", "val", "test")},
            "block_bounds": arrays["block_bounds"].tolist()}
    (out_dir / f"{name}.json").write_text(json.dumps(side, indent=1, sort_keys=True))
    return side


def done(name: str, out_dir: Path = OUT_DIR) -> bool:
    j = out_dir / f"{name}.json"
    try:
        side = json.loads(j.read_text())
        return sha256_file(out_dir / side["file"]) == side["sha256"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
