"""Training loop and run bookkeeping.

Physics target: E_f, the neutral O-vacancy formation energy (eV), standardised with the training-site
mean and sd. S and D minimise L1 on the standardised target. P1 minimises an equal-weight MSE over the
22 standardised D descriptors. Optimiser AdamW with ReduceLROnPlateau (factor 0.5, patience 15) on the
validation metric. Early stopping on the validation metric with min_delta 0: an epoch improves only if
the metric is strictly lower than the best so far. The best weights are restored at the end. The
validation metric is the MAE of E_f in eV for S and D, and the MAE over standardised descriptors for P1.

Validation hosts: outer splits use ``dftgnn.split.val_split`` on the training hosts with ``val_seed``; the
Kiyohara split uses its released validation hosts.

Standardisation statistics come from training sites only and are saved in every checkpoint.

Run identity: ``run_id`` is the first 16 hex digits of the sha256 of the canonical JSON of (model,
hyperparameters, split name, r, B, seed, code git SHA, graphs manifest sha256). Explicit host lists
(smoke runs only) also enter the hash.

Determinism: seeds are set for Python, numpy and torch, and torch runs with
``use_deterministic_algorithms(True, warn_only=True)``. On CUDA, the scatter-add aggregations of PyG
have no deterministic kernel, so GPU runs are reproducible only up to floating-point summation order.
On CPU, repeated runs with the same seed and thread count are bit-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dftgnn.config import Config, load_config
from dftgnn.graphs import collate_sites
from dftgnn.graphs import store as G
from dftgnn.models import StagedP, Standardiser, build_model, hparams_for
from dftgnn.split import budget_train, load_split, val_split
from dftgnn.split.cv import CV_NAME, CV_PREFIX
from dftgnn.split.cv import fold as cv_fold
from dftgnn.split.cv import parse_name as cv_parse_name
from dftgnn.split.cv import val_r as cv_val_r
from dftgnn.stats.metrics import point_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = REPO_ROOT / "checkpoints"
TRAINED = ("S", "D-state", "D-late", "P1", "cgcnn-S", "cgcnn-D")
LOCO_PREFIX = "loco_v1_f"      # split name of LOCO fold k: "loco_v1_f<k>" (splits/loco_v1.json)
# Diagnostic ablations, applied identically in training, validation and test. "vacancy_flag" zeroes the flag.
# "desc_host" / "desc_site" set the host- or site-electronic descriptors of a D model to their training mean
# (0 after standardisation), which removes that class's information.
ABLATIONS = ("vacancy_flag", "desc_host", "desc_site")


def apply_ablation(batch, ablate: str | None):
    """Apply ``ablate`` in place to a collated batch and return it."""
    if ablate is None:
        return batch
    if ablate == "vacancy_flag":
        batch.vac_flag.zero_()
    elif ablate == "desc_host":
        batch.desc_host.zero_()
    elif ablate == "desc_site":
        batch.desc_site.zero_()
    else:
        raise ValueError(f"unknown ablation {ablate!r}")
    return batch


def val_seed(r: int | str, budget: int, seed: int) -> int:
    """Clarified validation seed; LOCO passes r = f"loco{fold}" (2026-10-10)."""
    return int.from_bytes(hashlib.sha256(f"val|{r}|{budget}|{seed}".encode()).digest()[:4], "big")


def pick_device() -> torch.device:
    """cuda if available, else cpu. MPS is never used for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_cpus() -> float:
    """CPUs this process may use: the cgroup quota when one is set (containers report the host's
    cores through ``os.cpu_count``), else the scheduler affinity."""
    for quota, period in (("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),):
        try:
            q, per = int(Path(quota).read_text()), int(Path(period).read_text())
            if q > 0:
                return q / per
        except (OSError, ValueError):
            pass
    try:
        mx = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if mx[0] != "max":
            return int(mx[0]) / int(mx[1])
    except (OSError, ValueError, IndexError):
        pass
    return float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))


def code_sha(root: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


CODE_REF_PATHS = ("src", "scripts", "configs", "splits", "specs")
# Successor tags whose code may run the specs pinned to a code reference (deviations rows of 2026-10-14,
# 2026-10-18 and 2026-10-19), oldest first: v2 bug fixes, v4 the moment_v2 stage, v5 the stages p1tune_v2,
# dclass, p1v2b and pcompv2b, v6 the tuned P1 settings and the specs, v7 the pcompv2b dependency check on the
# ingested p1v2b records, v8 the stage dtune_v2 and the dretrain_v2 code, v9 the tuned D settings and the
# dretrain_v2 spec. The run ids keep the original reference string; only the code check accepts the successor.
CODE_REF_SUCCESSORS = {"nrp-code-v1": ("nrp-code-v2", "nrp-code-v4", "nrp-code-v5", "nrp-code-v6", "nrp-code-v7",
                                       "nrp-code-v8", "nrp-code-v9")}


def accepted_code_commits(ref: str, root: Path = REPO_ROOT) -> set[str]:
    """Commits of the tag ``ref`` and of its ``CODE_REF_SUCCESSORS`` that exist in this clone: the code commits
    whose records ``check_code_ref`` lets stand for ``ref``."""
    out = set()
    for tag in (ref, *CODE_REF_SUCCESSORS.get(ref, ())):
        rev = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"], cwd=root,
                             capture_output=True, text=True, check=False)
        if not rev.returncode:
            out.add(rev.stdout.strip())
    return out


def check_code_ref(ref: str, root: Path = REPO_ROOT) -> str:
    """Commit of the tag whose code the checkout runs, for the pinned code reference ``ref``.

    A spec with ``code_ref`` has its run id computed with ``ref`` in place of the commit SHA, so it may run only
    on that code: ``ref`` must resolve to an ancestor of HEAD and nothing under ``CODE_REF_PATHS`` may differ
    between it and HEAD (commits that add results on top of the tag are allowed). A successor of ``ref``
    (``CODE_REF_SUCCESSORS``; nrp-code-v2, v4 to v8 for nrp-code-v1) is accepted in its place: it must exist, be an
    ancestor of HEAD, and nothing under ``CODE_REF_PATHS`` may differ between it and HEAD. Raises RuntimeError
    otherwise.
    """
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)

    def resolve(tag: str) -> str | None:
        rev = git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
        return None if rev.returncode else rev.stdout.strip()

    commit = resolve(ref)
    if commit is None:
        raise RuntimeError(f"code reference {ref!r} does not exist in this clone (git fetch --tags)")
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise RuntimeError(f"code reference {ref!r} ({commit[:12]}) is not an ancestor of HEAD")
    tried = []
    for tag in (*reversed(CODE_REF_SUCCESSORS.get(ref, ())), ref):          # newest fix first
        c = commit if tag == ref else resolve(tag)
        if c is None or git("merge-base", "--is-ancestor", c, "HEAD").returncode:
            continue
        diff = git("diff", "--name-only", c, "HEAD", "--", *CODE_REF_PATHS)
        if not diff.returncode and not diff.stdout.strip():
            return c
        tried.append(f"{tag}: {diff.stdout.split()[:5] or diff.stderr.strip()}")
    raise RuntimeError(f"code under {', '.join(CODE_REF_PATHS)} differs from {ref!r} and its bug-fix tags "
                       f"({'; '.join(tried)})")


@dataclass
class RunSpec:
    """One training run. ``split`` is ``outer_r<r>`` (r = 10..29: EXTENSION resamples), ``kiyohara``,
    ``loco_v1_f<k>`` (LOCO fold k, r = k) or ``cv_v1_k<K>_f<f>`` (EXTENSION K-fold CV, r = f).

    ``hosts`` (smoke only) overrides the split with explicit train/val/test host lists. ``max_epochs`` /
    ``patience`` are test-only overrides of the config values. ``p1_run`` / ``d_run`` are set for kind ``P``
    (evaluation of the staged model, no training). ``code_ref`` (cluster specs,
    ``scripts/queue/make_specs.py``) replaces the commit SHA in the run id; it does not enter the key
    otherwise, so every spec without it keeps its id. The run then requires ``check_code_ref``.
    """
    model: str
    hp: dict
    lr: float
    weight_decay: float
    batch_size: int
    split: str
    r: int
    budget: int
    seed: int
    hosts: dict | None = None
    max_epochs: int | None = None
    patience: int | None = None
    smoke: bool = False
    results_subdir: str | None = None     # results/<subdir>/ instead of results/ (pilots)
    eval_test: bool = True                # False: the test hosts are never loaded or predicted (epoch pilot)
    ablate: str | None = None             # see ABLATIONS (diagnostics only)
    p1_run: str | None = None
    d_run: str | None = None
    tags: dict = field(default_factory=dict)
    exclude_sites: list | None = None     # site ids removed from train, val and test (sensitivity a)
    code_ref: str | None = None           # pinned code reference used in the run id instead of the commit SHA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunSpec:
        return cls(**d)


def run_id(spec: RunSpec, code: str, graphs_sha: str) -> str:
    key = {"model": spec.model, "hyperparams": {**spec.hp, "lr": spec.lr,
                                                "weight_decay": spec.weight_decay,
                                                "batch_size": spec.batch_size},
           "split": spec.split, "r": spec.r, "B": spec.budget, "seed": spec.seed,
           "code_sha": code, "graphs_manifest_sha256": graphs_sha}
    if spec.hosts is not None:
        key["hosts"] = spec.hosts
    if not spec.eval_test:
        key["eval_test"] = False
    if spec.ablate is not None:
        key["ablate"] = spec.ablate
    if spec.results_subdir is not None:
        key["results_subdir"] = spec.results_subdir
    if spec.max_epochs is not None or spec.patience is not None:
        key["epoch_override"] = [spec.max_epochs, spec.patience]
    if spec.model == "P":
        key["components"] = [spec.p1_run, spec.d_run]
    if spec.exclude_sites is not None:
        key["exclude_sites"] = sorted(spec.exclude_sites)
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_hosts(spec: RunSpec, cfg: Config) -> dict[str, list[str]]:
    """Train / val / test host ids of a run."""
    if spec.hosts is not None:
        return {k: sorted(v) for k, v in spec.hosts.items()}
    if spec.split.startswith(LOCO_PREFIX):
        fold = int(spec.split.removeprefix(LOCO_PREFIX))
        f = load_split("loco_v1")["folds"][fold]
        if spec.budget != f["n_train"] or spec.r != fold:
            raise ValueError(f"LOCO fold {fold}: budget must be {f['n_train']} and r must be {fold}")
        v = cfg.training.validation
        tr, va = val_split(f["train"], frac=v.frac, min_hosts=v.min_hosts,
                           seed=val_seed(f"loco{fold}", spec.budget, spec.seed))
        return {"train": tr, "val": va, "test": f["test"] if spec.eval_test else []}
    if spec.split.startswith(CV_PREFIX):
        k, fold = cv_parse_name(spec.split)
        f = cv_fold(load_split(CV_NAME), k, fold)
        if spec.budget != f["n_train"] or spec.r != fold:
            raise ValueError(f"CV K={k} fold {fold}: budget must be {f['n_train']} and r must be {fold}")
        v = cfg.training.validation
        tr, va = val_split(f["train"], frac=v.frac, min_hosts=v.min_hosts,
                           seed=val_seed(cv_val_r(k, fold), spec.budget, spec.seed))
        return {"train": tr, "val": va, "test": f["test"] if spec.eval_test else []}
    sp = load_split(spec.split)
    if spec.split == "kiyohara":
        return {"train": sp["train"], "val": sp["val"], "test": sp["test"] if spec.eval_test else []}
    train = budget_train(sp, spec.budget)
    v = cfg.training.validation
    tr, va = val_split(train, frac=v.frac, min_hosts=v.min_hosts, seed=val_seed(spec.r, spec.budget, spec.seed))
    return {"train": tr, "val": va, "test": sp["test"] if spec.eval_test else []}


def epoch_limits(spec: RunSpec, cfg: Config) -> tuple[int, int]:
    me = spec.max_epochs if spec.max_epochs is not None else cfg.training.max_epochs
    pa = spec.patience if spec.patience is not None else cfg.training.early_stopping.patience
    if not isinstance(me, int) or not isinstance(pa, int):
        raise ValueError("max_epochs / patience are TBD-S07 in the config; pass a test override")  # noqa: TRY004
    return me, pa


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)   # bytes on macOS, KiB on Linux


class Store:
    """The graph store held in memory, with host -> site-position lookup."""

    def __init__(self, store_dir: Path = G.STORE_DIR, manifest: Path = G.MANIFEST, verify: bool = True):
        self.graphs, self.sites, self.meta = G.load_store(store_dir, verify=verify, manifest=manifest)
        self.manifest_sha = G.manifest_sha(manifest)
        hid = np.array(self.meta["host_ids"])[self.sites["host_idx"].numpy()]
        self.site_host = hid
        self.n_host = len(self.sites["desc_host_names"])

    def positions(self, hosts) -> np.ndarray:
        return np.nonzero(np.isin(self.site_host, list(hosts)))[0]


def _batches(positions: np.ndarray, size: int, gen: torch.Generator | None):
    order = positions if gen is None else positions[torch.randperm(len(positions), generator=gen).numpy()]
    for k in range(0, len(order), size):
        yield order[k:k + size]


def _predict(net, data: Store, pos, y, desc, batch_size, device, ablate=None) -> torch.Tensor:
    net.eval()
    out = []
    with torch.no_grad():
        for idx in _batches(pos, batch_size, None):
            b = apply_ablation(collate_sites(data.graphs, data.sites, idx, y=y, desc=desc).to(device), ablate)
            out.append(net(b).cpu())
    return torch.cat(out)


def train_run(spec: RunSpec, data: Store, cfg: Config | None = None, *, device: torch.device | None = None,
              ckpt_path: Path | None = None, log=print) -> dict:
    """Train one S / D / P1 model. Returns the result payload (predictions as a DataFrame)."""
    cfg = cfg if cfg is not None else load_config()
    if spec.model not in TRAINED:
        raise ValueError(f"{spec.model} is not trained by train_run")
    if spec.ablate not in (None, *ABLATIONS):
        raise ValueError(f"unknown ablation {spec.ablate!r}")
    if spec.ablate in ("desc_host", "desc_site") and spec.model not in ("D-state", "D-late"):
        raise ValueError(f"{spec.ablate} ablation applies to D models only")
    kept = (spec.tags or {}).get("descriptors_kept")   # dclass: the class the ablation keeps must be the one named
    if kept is not None and kept != data.sites["desc_site_names" if spec.ablate == "desc_host" else "desc_host_names"]:
        raise ValueError(f"descriptors_kept {kept} is not the class kept by ablation {spec.ablate!r}")
    device = device or pick_device()
    seed_everything(spec.seed)
    t0 = time.time()
    hosts = resolve_hosts(spec, cfg)
    pos = {k: data.positions(v) for k, v in hosts.items()}
    if spec.exclude_sites is not None:        # sensitivity (a): drop the listed sites from every split
        drop = set(spec.exclude_sites)
        pos = {k: np.array([p for p in v if data.sites["site_id"][p] not in drop], dtype=np.int64)
               for k, v in pos.items()}
        hosts = {k: sorted(set(data.site_host[pos[k]])) for k in pos}
    max_epochs, patience = epoch_limits(spec, cfg)

    target = data.sites["target"]
    desc_raw = data.sites["desc"]
    y_st = Standardiser.fit(target[pos["train"]].view(-1, 1))
    d_st = Standardiser.fit(desc_raw[pos["train"]])
    y = y_st(target.view(-1, 1)).view(-1).float()
    desc = d_st(desc_raw).float()
    is_p1 = spec.model == "P1"

    hp = hparams_for(spec.model, spec.hp)
    net = build_model(spec.model, hp, n_host=data.n_host, n_site=desc_raw.shape[1] - data.n_host,
                      cutoff=data.meta["cutoff_A"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.training.scheduler.factor,
                                                     patience=cfg.training.scheduler.patience)
    min_delta = cfg.training.early_stopping.min_delta
    gen = torch.Generator().manual_seed(spec.seed)

    def val_metric() -> float:
        p = _predict(net, data, pos["val"], y, desc, spec.batch_size, device, spec.ablate)
        if is_p1:
            return float((p - desc[pos["val"]]).abs().mean())
        return float((y_st.inverse(p.double().view(-1, 1)).view(-1) - target[pos["val"]]).abs().mean())

    best, best_state, best_epoch, wait, history, steps = float("inf"), None, -1, 0, [], 0
    for epoch in range(max_epochs):
        net.train()
        tot, n = 0.0, 0
        for idx in _batches(pos["train"], spec.batch_size, gen):
            b = apply_ablation(collate_sites(data.graphs, data.sites, idx, y=y, desc=desc).to(device), spec.ablate)
            out = net(b)
            if is_p1:
                loss = torch.nn.functional.mse_loss(out, torch.cat([b.desc_host, b.desc_site], -1))
            else:
                loss = torch.nn.functional.l1_loss(out, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
            n += len(idx)
            steps += 1
        vm = val_metric()
        sch.step(vm)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_metric": vm,
                        "lr": opt.param_groups[0]["lr"]})
        if vm < best - min_delta:
            best, best_epoch, wait = vm, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        log(f"{spec.model} epoch {epoch} train {tot / n:.4f} val {vm:.4f}")
        if wait >= patience:
            break
    if best_state is not None:                # None: the validation metric was never finite
        net.load_state_dict(best_state)
    val_metrics = None
    if not is_p1 and len(pos["val"]):         # restored best weights on the validation hosts (design screens)
        pv = _predict(net, data, pos["val"], y, desc, spec.batch_size, device, spec.ablate)
        val_metrics = point_metrics(target[pos["val"]].numpy(),
                                    y_st.inverse(pv.double().view(-1, 1)).view(-1).numpy(),
                                    data.site_host[pos["val"]])

    te = pos["test"]
    p = _predict(net, data, te, y, desc, spec.batch_size, device, spec.ablate) if len(te) else None
    frame = pd.DataFrame({"site_id": [data.sites["site_id"][i] for i in te], "host_id": data.site_host[te]})
    names = data.sites["desc_host_names"] + data.sites["desc_site_names"]
    if p is None:                             # epoch pilot: the test hosts are never evaluated
        frame = pd.DataFrame({"site_id": [], "host_id": []})
        metrics = {}
    elif is_p1:
        pr = d_st.inverse(p.double()).numpy()
        tr = desc_raw[te].numpy()
        for j, nm in enumerate(names):
            frame[f"true_{nm}"], frame[f"pred_{nm}"] = tr[:, j], pr[:, j]
        sst = ((tr - tr.mean(0)) ** 2).sum(0)
        metrics = {"per_descriptor": {nm: {
            "mae": float(np.abs(pr[:, j] - tr[:, j]).mean()),
            "r2": float(1 - ((pr[:, j] - tr[:, j]) ** 2).sum() / sst[j]) if sst[j] > 0 else None}
            for j, nm in enumerate(names)},
            "mean_standardised_mae": float((p - desc[te]).abs().mean())}
    else:
        frame["y_true"] = target[te].numpy()
        frame["y_pred"] = y_st.inverse(p.double().view(-1, 1)).view(-1).numpy()
        metrics = point_metrics(frame.y_true, frame.y_pred, frame.host_id)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "model_config": net.config, "spec": spec.to_dict(),
                    "target_mean": y_st.mean, "target_sd": y_st.sd,
                    "desc_mean": d_st.mean, "desc_sd": d_st.sd, "desc_names": names}, ckpt_path)
    return {
        "metrics": metrics, "predictions": frame,
        "n_hosts": {k: len(v) for k, v in hosts.items()},
        "n_sites": {k: len(v) for k, v in pos.items()},
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_metric": best, "val_metrics": val_metrics,
        "max_epochs": max_epochs, "patience": patience, "history": history, "train_steps": steps,
        "wall_time_s": time.time() - t0, "peak_memory_bytes": peak_memory_bytes(device),
        "device": str(device), "torch_threads": torch.get_num_threads(),
    }


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[torch.nn.Module, dict]:
    """(model in eval mode on ``device``, checkpoint dict). The model is moved to ``device`` before its weights are
    loaded, so every parameter, including the distance-basis centres of the bond expansion, is on the device of the
    batches it will see (nrp-code-v1 left it on the CPU; deviations row of 2026-10-14)."""
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    mc = ck["model_config"]
    net = build_model(mc["kind"], hparams_for(mc["kind"], mc["hp"]), n_host=mc["n_host"], n_site=mc["n_site"],
                      cutoff=mc["cutoff"]).to(device or "cpu")
    net.load_state_dict(ck["state_dict"])
    return net.eval(), ck


def component_checkpoint(rid: str, spec: RunSpec, ckpt_dir: Path = CKPT_DIR) -> Path:
    """Checkpoint file of a P component: ``<ckpt_dir>/<rid>.pt``. A component pinned in ``spec.tags["checkpoints"]``
    (pcompv2b: a D checkpoint of another checkout) is looked up in ``ckpt_dir``, the directories of
    ``$DFTGNN_CKPT_DIRS`` (separated by ``os.pathsep``) and the pinned ``dirs``, in that order, and the first file
    found must have the pinned sha256."""
    pin = ((spec.tags or {}).get("checkpoints") or {}).get(rid)
    if pin is None:
        return ckpt_dir / f"{rid}.pt"
    from dftgnn.graphs.store import sha256_file

    extra = [d for d in os.environ.get("DFTGNN_CKPT_DIRS", "").split(os.pathsep) if d]
    dirs = [Path(ckpt_dir), *map(Path, extra), *map(Path, pin["dirs"])]
    for d in dirs:
        f = d / f"{rid}.pt"
        if f.is_file():
            if sha256_file(f) != pin["sha256"]:
                raise RuntimeError(f"{f}: sha256 differs from {pin.get('record', 'the pinned hash')}")
            return f
    raise FileNotFoundError(f"checkpoint {rid}.pt not in {[str(d) for d in dirs]}; set DFTGNN_CKPT_DIRS")


def evaluate_p(spec: RunSpec, data: Store, ckpt_dir: Path = CKPT_DIR, cfg: Config | None = None,
               device: torch.device | None = None) -> dict:
    """Staged model P = P1 then D on the test hosts of ``spec`` (no training)."""
    cfg = cfg if cfg is not None else load_config()
    device = device or pick_device()
    t0 = time.time()
    p1, ck1 = load_checkpoint(component_checkpoint(spec.p1_run, spec, ckpt_dir), device)
    d, ckd = load_checkpoint(component_checkpoint(spec.d_run, spec, ckpt_dir), device)
    p = StagedP(p1, d, Standardiser(ck1["desc_mean"], ck1["desc_sd"]),
                Standardiser(ckd["desc_mean"], ckd["desc_sd"])).to(device).eval()
    te = data.positions(resolve_hosts(spec, cfg)["test"])
    target = data.sites["target"]
    y_st = Standardiser(ckd["target_mean"].cpu(), ckd["target_sd"].cpu())   # predictions are on the CPU here
    out = _predict(p, data, te, None, None, spec.batch_size, device)
    frame = pd.DataFrame({"site_id": [data.sites["site_id"][i] for i in te], "host_id": data.site_host[te],
                          "y_true": target[te].numpy(),
                          "y_pred": y_st.inverse(out.double().view(-1, 1)).view(-1).numpy()})
    return {"metrics": point_metrics(frame.y_true, frame.y_pred, frame.host_id), "predictions": frame,
            "components": {"P1": spec.p1_run, "D": spec.d_run}, "n_sites": {"test": len(te)},
            "wall_time_s": time.time() - t0, "peak_memory_bytes": peak_memory_bytes(device),
            "device": str(device)}
