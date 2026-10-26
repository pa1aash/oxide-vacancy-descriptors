"""Hyperparameter tuning.

One Optuna study per (model, anchor): models S, D-state, D-late and P1, anchors 50, 200 and 654 hosts of
resample 0's nested training sets. Each study runs until it holds ``tuning.optuna_trials_per_model_per_anchor``
(30) COMPLETE trials, with ``TPESampler(seed=tuning.optuna.sampler_seed)`` at Optuna defaults otherwise and
no pruning. Studies are stored in SQLite, one file per study, and are resumable: a trial left RUNNING by a
killed process is marked FAIL when the study is reopened, and the study continues to 30 complete trials.

Objective. Its validation hosts are ``val_split`` of the anchor's training hosts with the seed of (r = 0, B,
seed 0), so all four models share one validation set per anchor. The objective is the best validation metric
of the trial: the MAE of E_f in eV for S, D-state and D-late, and the MAE over standardised descriptors for
P1. A non-finite value is scored ``tuning.nonfinite_objective`` (1000.0) and counted.

Test hosts. ``TuningStore`` removes the test hosts of the anchor resample from memory after the store is
read (their graphs become None and their targets and descriptors NaN), and its ``positions`` raises if a
test host is ever requested. Trials run with ``eval_test=False``.

GPU sharing. Four studies run at once, one process each, with trials sequential within a study. Before a trial
a process takes a VRAM admission slot (``VramLedger``) sized by the benchmark peak of its (hidden width, batch
size, MEGNet blocks) cell; the benchmark timed set2set pooling, and that value is used for mean pooling as
well. A trial that runs out of memory is retried once with the device to itself (``needs_solo``).

Extended search spaces. ``run_study(space=...)`` replaces the configured space for one study. Parameters
outside ``ARCH_KEYS`` and ``OPT_KEYS`` go into the architecture (``hp``), except ``MODEL_PARAM``, which picks
the trained variant (D-state or D-late) of each trial; such a study is named after the model label ``D``.
Without ``space`` the sampled parameters, specs, trial CSV and record are those of before.
"""
from __future__ import annotations

import csv
import fcntl
import gc
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch

from dftgnn.config import Config, load_config
from dftgnn.graphs.store import sha256_file
from dftgnn.jobqueue import ADMIT_FRACTION, is_oom
from dftgnn.split import load_split
from dftgnn.train import REPO_ROOT, RunSpec, Store, code_sha, resolve_hosts, train_run
from dftgnn.train import admission as A

MODELS = ("S", "D-state", "D-late", "P1")
ARCH_KEYS = ("hidden_width", "megnet_blocks", "dropout", "readout_mlp_width", "pooling")
OPT_KEYS = ("learning_rate", "weight_decay", "batch_size")
MODEL_PARAM = "injection"     # extended spaces only: the D variant a trial trains
D_VARIANTS = ("D-state", "D-late")
RESULTS_DIR = REPO_ROOT / "results" / "tuning"
STORAGE_DIR = Path("/workspace/tuning")
BOUNDARY_FRACTION = 0.05      # a continuous value within 5% of its (log) range of an edge is a boundary hit
CSV_FIELDS = ("study", "trial", "state", *ARCH_KEYS, *OPT_KEYS, "objective", "val_metric_raw", "nonfinite",
              "best_epoch", "epochs_run", "wall_s", "peak_gb", "est_peak_gb", "solo", "oom_retries",
              "n_train_hosts", "n_val_hosts", "n_train_sites", "n_val_sites", "device", "pod_commit",
              "graphs_manifest_sha256", "finished_utc")


def study_name(model: str, anchor: int, prefix: str = "") -> str:
    return f"{prefix}{model}_a{anchor}"


def storage_url(storage_dir: Path, name: str) -> str:
    return f"sqlite:///{Path(storage_dir).resolve() / f'{name}.db'}"


# ---------------------------------------------------------------- search space

def suggest(trial: optuna.trial.BaseTrial, space: dict) -> dict:
    """Sample the section 6 search space, read from ``tuning.search_space`` (identical for all models)."""
    out = {}
    for name, p in space.items():
        if p.dist == "log_uniform":
            out[name] = trial.suggest_float(name, p.low, p.high, log=True)
        elif p.dist == "uniform":
            out[name] = trial.suggest_float(name, p.low, p.high)
        else:
            out[name] = trial.suggest_categorical(name, list(p.choices))
    return out


def split_params(params: dict) -> dict:
    """Search-space parameters -> {"hp", "lr", "weight_decay", "batch_size"} as a RunSpec takes them. Parameters of an
    extended space outside ``ARCH_KEYS`` and ``OPT_KEYS`` follow the architecture keys in ``hp``; ``MODEL_PARAM`` is
    not a hyperparameter of the network and is left out."""
    extra = {k: v for k, v in params.items() if k not in ARCH_KEYS and k not in OPT_KEYS and k != MODEL_PARAM}
    return {"hp": {**{k: params[k] for k in ARCH_KEYS}, **extra}, "lr": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]), "batch_size": int(params["batch_size"])}


def trial_model(model: str, params: dict) -> str:
    """The variant a trial trains: ``params[MODEL_PARAM]`` when the space has it, else ``model``."""
    return params.get(MODEL_PARAM, model)


def space_from_dict(space: dict) -> dict:
    """A search space written as JSON (``{name: {"dist", "low", "high", "choices"}}``, as in a job spec) in the
    attribute form of ``tuning.search_space``."""
    return {k: SimpleNamespace(dist=v["dist"], low=v.get("low"), high=v.get("high"), choices=v.get("choices"))
            for k, v in space.items()}


def csv_fields(space: dict | None) -> tuple[str, ...]:
    """Trial CSV columns: ``CSV_FIELDS``, with the extra parameters of an extended space after ``OPT_KEYS``."""
    if space is None:
        return CSV_FIELDS
    extra = tuple(k for k in space if k not in ARCH_KEYS and k not in OPT_KEYS)
    i = CSV_FIELDS.index(OPT_KEYS[-1]) + 1
    return (*CSV_FIELDS[:i], *extra, *CSV_FIELDS[i:])


def boundary_hits(params: dict, space: dict, frac: float = BOUNDARY_FRACTION) -> dict[str, str]:
    """Parameters at a search-space edge (informational only).

    Continuous: within ``frac`` of the range (of log10 for log-uniform) from the low or high end.
    Ordered categorical with at least three choices: the smallest or largest choice. Two-choice and
    unordered parameters (pooling) are not reported: every value is an edge.
    """
    hits = {}
    for name, p in space.items():
        v = params[name]
        if p.dist in ("log_uniform", "uniform"):
            f = (lambda x: math.log10(x)) if p.dist == "log_uniform" else (lambda x: x)
            pos = (f(v) - f(p.low)) / (f(p.high) - f(p.low))
            if pos <= frac:
                hits[name] = "low"
            elif pos >= 1 - frac:
                hits[name] = "high"
        elif len(p.choices) >= 3 and all(isinstance(c, (int, float)) for c in p.choices):
            if v == min(p.choices):
                hits[name] = "low"
            elif v == max(p.choices):
                hits[name] = "high"
    return hits


# ---------------------------------------------------------------- data

class TuningStore(Store):
    """The graph store with the anchor resample's test hosts removed from memory."""

    def __init__(self, resample: int = 0, **kw):
        super().__init__(**kw)
        self.excluded_hosts = frozenset(load_split(f"outer_r{resample}")["test"])
        host_ids = list(self.meta["host_ids"])
        for i, h in enumerate(host_ids):
            if h in self.excluded_hosts:
                self.graphs[i] = None
        mask = torch.from_numpy(np.isin(self.site_host, list(self.excluded_hosts)))
        self.n_excluded_sites = int(mask.sum())
        for key in ("target", "desc", "p1_target"):
            t = self.sites[key].clone()
            t[mask] = float("nan")
            self.sites[key] = t

    def positions(self, hosts) -> np.ndarray:
        bad = set(hosts) & self.excluded_hosts
        if bad:
            raise AssertionError(f"tuning requested {len(bad)} test hosts of the anchor resample")
        return super().positions(hosts)


def trial_spec(model: str, anchor: int, params: dict, cfg: Config, study: str = "",
               arch: dict | None = None) -> RunSpec:
    """``arch`` holds fixed backbone options, merged into the sampled architecture; it is not searched."""
    t = cfg.tuning
    sp = split_params(params)
    sp["hp"] = {**sp["hp"], **(arch or {})}
    return RunSpec(model=model, **sp, split=f"outer_r{t.anchor_resample}",
                   r=t.anchor_resample, budget=anchor, seed=t.trial_seed, eval_test=False,
                   tags={"tuning": study or study_name(model, anchor)})


def anchor_hosts(anchor: int, cfg: Config) -> dict[str, list[str]]:
    """Train / val hosts of an anchor (identical for every model: the seed depends on r, B and seed 0)."""
    t = cfg.tuning
    spec = RunSpec(model="S", hp={}, lr=0.0, weight_decay=0.0, batch_size=1, split=f"outer_r{t.anchor_resample}",
                   r=t.anchor_resample, budget=anchor, seed=t.trial_seed, eval_test=False)
    return resolve_hosts(spec, cfg)


# ---------------------------------------------------------------- VRAM admission across study processes

class VramLedger:
    """Cross-process VRAM admission for the study processes sharing one GPU.

    ``<dir>/<pid>.json`` holds ``{"gb", "solo"}`` of a process's running trial. Under an exclusive
    ``fcntl`` lock a trial is admitted if no solo trial runs, no solo trial waits, and the summed
    estimates stay within ``ADMIT_FRACTION`` of the device memory (a trial larger than that runs only
    on an idle device). A solo trial waits for an empty ledger. Entries of dead processes are dropped.
    """

    def __init__(self, root: Path, vram_gb: float | None, poll: float = 5.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = None if vram_gb is None else ADMIT_FRACTION * vram_gb
        self.poll = poll
        self.me = self.root / f"{os.getpid()}.json"
        self.wait_solo = self.root / f"{os.getpid()}.solo_wait"

    def _entries(self) -> list[dict]:
        out = []
        for f in self.root.glob("*.json"):
            try:
                pid = int(f.stem)
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                f.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            if f == self.me:
                continue
            try:
                out.append(json.loads(f.read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return out

    def _solo_waiters(self) -> bool:
        for f in self.root.glob("*.solo_wait"):
            if f == self.wait_solo:
                continue
            try:
                os.kill(int(f.stem), 0)
                return True
            except (ValueError, ProcessLookupError):
                f.unlink(missing_ok=True)
            except PermissionError:
                return True
        return False

    def try_acquire(self, gb: float, solo: bool = False) -> bool:
        if self.budget is None:
            return True
        with open(self.root / ".lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                others = self._entries()
                if solo:
                    ok = not others
                else:
                    used = sum(e["gb"] for e in others)
                    ok = (not any(e.get("solo") for e in others) and not self._solo_waiters()
                          and (not others or used + gb <= self.budget + 1e-9))
                if ok:
                    self.me.write_text(json.dumps({"gb": gb, "solo": solo}))
                    self.wait_solo.unlink(missing_ok=True)
                elif solo:
                    self.wait_solo.write_text("")
                return ok
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def release(self) -> None:
        self.me.unlink(missing_ok=True)
        self.wait_solo.unlink(missing_ok=True)

    @contextmanager
    def slot(self, gb: float, solo: bool = False):
        while not self.try_acquire(gb, solo):
            time.sleep(self.poll)
        try:
            yield
        finally:
            self.release()


# ---------------------------------------------------------------- one study

def open_study(name: str, storage_dir: Path, cfg: Config) -> optuna.Study:
    o = cfg.tuning.optuna
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=name, storage=storage_url(storage_dir, name), direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=o.sampler_seed),
                               pruner=optuna.pruners.NopPruner(), load_if_exists=True)


def fail_stale(study: optuna.Study) -> list[int]:
    """Mark trials left RUNNING by a killed process as FAIL (one process owns a study at a time)."""
    stale = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.RUNNING]
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    return [t.number for t in stale]


def n_complete(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))


def _append_csv(path: Path, row: dict, fields: tuple[str, ...] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def run_study(model: str, anchor: int, data: Store, *, cfg: Config | None = None, prefix: str = "",
              storage_dir: Path = STORAGE_DIR, csv_dir: Path = RESULTS_DIR, n_trials: int | None = None,
              device: torch.device | None = None, ledger: VramLedger | None = None,
              est_gb: Callable[[dict], float] | None = None, train_fn=train_run, pod_commit: str | None = None,
              epoch_override: tuple[int, int] | None = None, arch: dict | None = None,
              space: dict | None = None, log=print) -> optuna.Study:
    """Run (or resume) the study of ``model`` at ``anchor`` until it holds ``n_trials`` COMPLETE trials.
    ``space`` (attribute form, ``space_from_dict``) replaces the configured search space; with ``MODEL_PARAM`` in it
    the model is ``D`` and each trial trains the sampled variant."""
    cfg = cfg if cfg is not None else load_config()
    if space is not None and MODEL_PARAM in space:
        if model != "D" or set(space[MODEL_PARAM].choices) - set(D_VARIANTS):
            raise ValueError(f"{MODEL_PARAM} chooses among {D_VARIANTS} in a study of model D")
    elif model not in MODELS:
        raise ValueError(f"{model} is not a tuned model")
    n_trials = cfg.tuning.optuna_trials_per_model_per_anchor if n_trials is None else n_trials
    device = device or torch.device("cpu")
    name = study_name(model, anchor, prefix)
    study = open_study(name, storage_dir, cfg)
    stale = fail_stale(study)
    if stale:
        log(f"{name}: marked interrupted trials {stale} as FAIL")
    hosts = anchor_hosts(anchor, cfg)
    if hosts["test"]:
        raise AssertionError("a tuning run resolved test hosts")
    commit = pod_commit or code_sha()
    csv_path = Path(csv_dir) / f"trials_{name}.csv"
    fields = csv_fields(space)
    while n_complete(study) < n_trials:
        trial = study.ask()
        params = suggest(trial, space if space is not None else cfg.tuning.search_space)
        spec = trial_spec(trial_model(model, params), anchor, params, cfg, name, arch)
        if epoch_override is not None:
            spec.max_epochs, spec.patience = epoch_override
        gb = est_gb(params) if est_gb is not None else 0.0
        solo, ooms, t0 = False, 0, time.time()
        while True:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                slot = ledger.slot(gb, solo) if ledger is not None else _nullslot()
                with slot:
                    res = train_fn(spec, data, cfg, device=device, ckpt_path=None, log=lambda *_: None)
                break
            except Exception as exc:
                _free(device)
                if is_oom(exc) and not solo:
                    ooms += 1
                    solo = True
                    log(f"{name} trial {trial.number}: out of memory; retrying as needs_solo")
                    continue
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
        _free(device)
        if res["n_sites"].get("test", 0) or res["n_hosts"].get("test", 0):
            raise AssertionError("a tuning trial touched test sites")
        raw = float(res["best_val_metric"])
        nonfinite = not math.isfinite(raw)
        value = cfg.tuning.nonfinite_objective if nonfinite else raw
        attrs = {"val_metric_raw": raw if not nonfinite else str(raw), "nonfinite": nonfinite,
                 "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                 "wall_s": round(time.time() - t0, 2), "peak_gb": round(res["peak_memory_bytes"] / A.GIB, 3),
                 "est_peak_gb": round(gb, 3), "solo": solo, "oom_retries": ooms,
                 "n_train_hosts": res["n_hosts"]["train"], "n_val_hosts": res["n_hosts"]["val"],
                 "n_train_sites": res["n_sites"]["train"], "n_val_sites": res["n_sites"]["val"],
                 "device": res.get("device", str(device)), "pod_commit": commit,
                 "graphs_manifest_sha256": data.manifest_sha}
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        study.tell(trial, value)
        _append_csv(csv_path, {"study": name, "trial": trial.number, "state": "COMPLETE", **params,
                               "objective": value, **attrs,
                               "finished_utc": datetime.now(UTC).isoformat(timespec="seconds")}, fields)
        log(f"{name} trial {trial.number}: objective {value:.4f} best_epoch {res['best_epoch']} "
            f"epochs {res['epochs_run']} wall {attrs['wall_s']:.0f}s ({n_complete(study)}/{n_trials})")
    return study


@contextmanager
def _nullslot():
    yield


# ---------------------------------------------------------------- summary

def summarise(study: optuna.Study, model: str, anchor: int, cfg: Config, *, top: int = 5,
              arch: dict | None = None, epochs: tuple[int, int] | None = None, space: dict | None = None,
              space_json: dict | None = None) -> dict:
    """Payload of ``write_result("tuning_<model>_a<anchor>")``. ``epochs`` = (max_epochs, patience) of a study run
    with ``epoch_override`` (p1tune_v2); by default the configured values. ``space`` / ``space_json``: the extended
    search space of a study run with ``space`` (dtune_v2), in attribute and JSON form; the payload then also holds
    ``search_space`` and ``best_model`` (the variant of the best trial)."""
    max_epochs, patience = epochs or (cfg.training.max_epochs, cfg.training.early_stopping.patience)
    done = sorted(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)),
                  key=lambda t: (t.value, t.number))
    failed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,))
    best = done[0]
    ext = space is not None
    space = space if ext else cfg.tuning.search_space
    topk = [{"trial": t.number, "value": t.value, "params": t.params,
             "best_epoch": t.user_attrs.get("best_epoch"), "epochs_run": t.user_attrs.get("epochs_run")}
            for t in done[:top]]
    ua = best.user_attrs
    walls = [t.user_attrs.get("wall_s", 0.0) for t in done]
    # stopped by the epoch cap rather than by early stopping
    capped = [t.number for t in done if t.user_attrs.get("epochs_run") == max_epochs]
    return {
        "model": model, "anchor": anchor, "study": study.study_name,
        "objective": "val_mae_Ef_eV" if model != "P1" else "val_mae_standardised_descriptors",
        "n_complete": len(done), "n_failed": len(failed),
        "n_nonfinite": sum(bool(t.user_attrs.get("nonfinite")) for t in done),
        "best_value": best.value, "best_trial": best.number, "best_params": best.params,
        "best_hparams": split_params(best.params),
        "best_trial_best_epoch": ua.get("best_epoch"), "best_trial_epochs_run": ua.get("epochs_run"),
        "top": topk, "top5_spread": topk[-1]["value"] - topk[0]["value"] if len(topk) > 1 else 0.0,
        "boundary_hits": boundary_hits(best.params, space),
        "boundary_rule": f"continuous within {BOUNDARY_FRACTION:.0%} of the (log) range of an edge; ordered "
                         "categorical with >= 3 choices at its smallest or largest choice",
        "trials_reaching_cap": capped,
        "validation": {k: ua.get(k) for k in ("n_train_hosts", "n_val_hosts", "n_train_sites", "n_val_sites")},
        "sampler": {"name": "TPESampler", "seed": cfg.tuning.optuna.sampler_seed, "n_startup_trials": 10},
        "pruner": "NopPruner", "trial_seed": cfg.tuning.trial_seed,
        "max_epochs": max_epochs, "patience": patience,
        "sum_trial_wall_s": sum(walls),
        "pod_commit": ua.get("pod_commit"), "graphs_manifest_sha256": ua.get("graphs_manifest_sha256"),
        "test_hosts_loaded": False, "arch": arch or {},
        **({"search_space": space_json, "best_model": trial_model(model, best.params)} if ext else {}),
    }


def archive_study(study_db: Path, csv_path: Path, out_dir: Path) -> dict:
    """Copy the study's SQLite file next to the trial CSV; return both paths with their sha256."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / study_db.name
    shutil.copy2(study_db, dst)

    def rel(p: Path) -> str:
        r = p.resolve()
        return str(r.relative_to(REPO_ROOT)) if r.is_relative_to(REPO_ROOT) else str(p)

    return {"trials_csv": {"path": rel(csv_path), "sha256": sha256_file(csv_path)},
            "optuna_db": {"path": rel(dst), "sha256": sha256_file(dst)}}
