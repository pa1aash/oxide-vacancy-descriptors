"""Non-training queue tasks: ``relax``, ``embed``, ``geomeval``, ``p1tune_v2`` and ``dtune_v2``.

A task job is ``{"run_id", "stage", "spec": {"model": <stage>, "r", "budget", "seed", ...}, "after", "est_peak_gb",
"priority", "tranche"}``; the run id is ``infer.task_id`` of a key that includes the code SHA. Every task writes a
result JSON via write_result under ``results/<subdir>/<run_id>.json`` whose payload lists its artefacts
(``artifacts``: path and sha256), so the outbox packs them like training runs. Every task is idempotent: a verified
result is not recomputed (``verified``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from dftgnn import infer
from dftgnn.graphs.store import sha256_file
from dftgnn.hardware import hardware_info
from dftgnn.io.results import write_result

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_STAGES = ("relax", "embed", "geomeval", "p1tune_v2", "dtune_v2")
SUBDIR = {"relax": "mlip_v1", "embed": "embeddings_v1", "geomeval": "geomeval", "p1tune_v2": "p1tune_v2",
          "dtune_v2": "dtune_v2"}
TUNING_STAGES = {"p1tune_v2": "P1", "dtune_v2": "D"}     # tuning task -> the model label of its studies


def mace_sha() -> str:
    """sha256 of the MACE-MP-0 medium checkpoint as recorded in the tracked results/mace_model.json."""
    return json.loads((REPO_ROOT / "results" / "mace_model.json").read_text())["payload"]["sha256"]


def result_path(stage: str, rid: str) -> Path:
    return REPO_ROOT / "results" / SUBDIR[stage] / f"{rid}.json"


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def artifact(p: Path) -> dict:
    return {"path": _rel(p), "sha256": sha256_file(p)}


def verified(path: Path, rid: str) -> bool:
    try:
        pay = json.loads(path.read_text())["payload"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    if pay.get("run_id") != rid:
        return False
    arts = list(pay.get("artifacts", [])) + ([pay["predictions"]] if pay.get("predictions") else [])
    return all((REPO_ROOT / a["path"]).is_file() and sha256_file(REPO_ROOT / a["path"]) == a["sha256"] for a in arts)


# ----------------------------------------------------------------------------------------------- relax
def relax_key(host_id: str, code: str) -> dict:
    return {"stage": "relax", "host_id": host_id, "code_sha": code, "mace_model_sha256": mace_sha()}


class RelaxContext:
    """Per-process cache: unit cells, tiling map, universe, MACE calculator."""

    def __init__(self, device: str):
        from dftgnn.mlip import mace_model, release

        self.device = device
        self.cells = release.unit_cells()
        self.tmap = pd.read_parquet(REPO_ROOT / "data/processed/tiling_map_v1.parquet").set_index("host_id")
        self.u = pd.read_parquet(REPO_ROOT / "data/processed/universe_v1.parquet")
        self.hosts = release.host_table(self.u).set_index("host_id")
        path = mace_model.model_path()
        self.msha = sha256_file(path)
        if self.msha != mace_sha():
            raise RuntimeError("MACE checkpoint differs from results/mace_model.json")
        self.calc = mace_model.calculator(device, path)


def run_relax(rid: str, spec: dict, ctx: RelaxContext, out_dir: Path | None = None) -> Path:
    from dftgnn.mlip import release
    from dftgnn.mlip.pipeline import host_job
    from dftgnn.mlip.relax import OUT_DIR

    out_dir = out_dir or OUT_DIR
    h = spec["host_id"]
    vac = sorted(int(v) for v in ctx.u.vacancy_atom_index[ctx.u.host_id == h])
    rec = host_job(h, ctx.cells[h], release.read_supercell(ctx.hosts.loc[h, "cif"]), vac, ctx.tmap.loc[h].to_dict(),
                   ctx.calc, model_sha256=ctx.msha, out_dir=out_dir)
    arts = [artifact(out_dir / f"{h}.json")] + [artifact(out_dir / a["path"]) for a in rec["files"].values()]
    pay = {"run_id": rid, "spec": spec, **{k: v for k, v in rec.items() if k not in ("files", "skipped")},
           "artifacts": arts, "hardware": hardware_info(ctx.device)}
    return write_result(rid, pay, results_dir=result_path("relax", rid).parent)


# ----------------------------------------------------------------------------------------------- embed
def embed_key(r: int, budget: int, kind: str, s_run: str, code: str) -> dict:
    return {"stage": "embed", "r": r, "budget": budget, "kind": kind, "checkpoint_run_id": s_run, "code_sha": code}


def run_embed(rid: str, spec: dict, sites: infer.Sites, prim: dict, device=None) -> Path:
    from dftgnn import embeddings as E

    r, b, kind = spec["r"], spec["budget"], spec["kind"]
    rec = prim[("S", f"outer_r{r}", b, 0)]
    name = f"r{r}_B{b}_{kind}"
    if not E.done(name):
        arrays = E.extract(kind, rec, sites, r, b, device=device)
        E.write(arrays, name, {"r": r, "budget": b, "kind": kind, "checkpoint_run_id": rec["run_id"],
                               "registered_control": kind in ("trained", "init0"),
                               "label": "DESCRIPTIVE variance of the random-init control"
                               if kind in ("init1", "init2") else None})
    side = json.loads((E.OUT_DIR / f"{name}.json").read_text())
    pay = {"run_id": rid, "spec": spec, **side, "hardware": hardware_info(device),
           "artifacts": [artifact(E.OUT_DIR / f"{name}.npz"), artifact(E.OUT_DIR / f"{name}.json")]}
    return write_result(rid, pay, results_dir=result_path("embed", rid).parent)


# -------------------------------------------------------------------------------------------- geomeval
def run_geomeval(rid: str, spec: dict, sites: infer.Sites, prim: dict, key: dict, device=None,
                 mlip_dir: Path | None = None) -> Path:
    from dftgnn.mlip import geomeval as GE
    from dftgnn.mlip.relax import OUT_DIR

    comps = GE.components(prim, spec["geo_model"], spec["r"], spec["budget"], spec["seed"])
    frame, info = GE.run(spec["geo_model"], spec["r"], spec["budget"], spec["seed"], comps, sites,
                         mlip_dir=mlip_dir or OUT_DIR, device=device)
    rdir = result_path("geomeval", rid).parent
    (rdir / "predictions").mkdir(parents=True, exist_ok=True)
    pp = rdir / "predictions" / f"{rid}.parquet"
    frame.to_parquet(pp, index=False)
    pay = {"run_id": rid, "task_id": rid, "spec": spec, "key": key, **info, "hardware": hardware_info(device),
           "predictions": {**artifact(pp), "rows": len(frame)}}
    return write_result(rid, pay, results_dir=rdir)


def geomeval_key(geo_model: str, r: int, budget: int, seed: int, comps: dict, code: str) -> dict:
    from dftgnn.mlip import geomeval as GE

    return {"stage": "geomeval", "model": geo_model, "r": r, "budget": budget, "seed": seed,
            "components": {k: v["run_id"] for k, v in comps.items()}, "code_sha": code,
            "tiling_summary_sha256": sha256_file(GE.TILING_SUMMARY), "mace_model_sha256": mace_sha()}


# ------------------------------------------------------------------------------------------- p1tune_v2
def tuning_storage(stage: str = "p1tune_v2") -> Path:
    """Optuna SQLite directory of a tuning stage's studies, outside the checkout so that a resumed job (or a new pod)
    continues the same study: ``$DFTGNN_TUNING_DIR``, else ``<checkout>/../tuning_<stage>``."""
    return Path(os.environ.get("DFTGNN_TUNING_DIR") or REPO_ROOT.parent / f"tuning_{stage}")


def run_p1tune(rid: str, job: dict, data, device, storage: Path | None = None, log=print) -> Path:
    """Run or resume the P1 study of the job until it holds ``n_trials`` COMPLETE trials (``dftgnn.tune.run_study``
    with the spec's arch and epoch limit), write ``results/tuning/tuning_<study>.json`` as ``scripts/tune/run_study.py``
    writes the S and D records (trial CSV and SQLite copy beside it), then the task record listing those three files.
    ``data`` is a ``TuningStore`` of the anchor resample (its test hosts removed from memory)."""
    return run_tuning(rid, job, data, device, storage=storage, log=log)


def run_tuning(rid: str, job: dict, data, device, storage: Path | None = None, log=print) -> Path:
    """``run_p1tune`` for any stage of ``TUNING_STAGES``. A dtune_v2 spec carries its extended ``search_space``
    (``dftgnn.tune.run_study(space=...)``); a p1tune_v2 spec has none and uses the configured space."""
    from dftgnn.config import load_config
    from dftgnn.train import admission, code_sha
    from dftgnn.tune import RESULTS_DIR, archive_study, run_study, space_from_dict, summarise

    spec, cfg = job["spec"], load_config()
    stage = job.get("stage", "p1tune_v2")
    model = TUNING_STAGES[stage]
    t = cfg.tuning
    if (spec["r"], spec["seed"], spec["n_trials"]) != (t.anchor_resample, t.trial_seed,
                                                      t.optuna_trials_per_model_per_anchor):
        raise ValueError(f"{rid}: spec differs from the configured tuning protocol")
    if spec["model"] != model:
        raise ValueError(f"{rid}: stage {stage} tunes {model}, the spec names {spec['model']}")
    name, anchor = spec["study"], spec["anchor"]
    prefix = name.removesuffix(f"{model}_a{anchor}")
    space_json = spec.get("search_space")
    space = space_from_dict(space_json) if space_json is not None else None
    storage = Path(storage or tuning_storage(stage))
    try:
        table = admission.load_benchmark()
        est = lambda p: admission.est_peak_for_spec(table, p, int(p["batch_size"]))
    except (OSError, KeyError):
        est = None
    epochs = (spec["max_epochs"], spec["patience"])
    commit = code_sha()
    study = run_study(model, anchor, data, cfg=cfg, prefix=prefix, storage_dir=storage, csv_dir=RESULTS_DIR,
                      n_trials=spec["n_trials"], device=device, est_gb=est, pod_commit=commit,
                      epoch_override=epochs, arch=spec["arch"], space=space, log=log)
    payload = summarise(study, model, anchor, cfg, arch=spec["arch"], epochs=epochs, space=space,
                        space_json=space_json)
    payload["artefacts"] = archive_study(storage / f"{name}.db", RESULTS_DIR / f"trials_{name}.csv",
                                         RESULTS_DIR / "optuna")
    rec = write_result(f"tuning_{name}", payload, config=cfg, results_dir=RESULTS_DIR)
    arts = [artifact(rec)] + [artifact(REPO_ROOT / a["path"]) for a in payload["artefacts"].values()]
    pay = {"run_id": rid, "spec": spec, "key": job.get("key"), "code_ref": job.get("code_ref"), "study": name,
           "best_value": payload["best_value"], "best_trial": payload["best_trial"],
           "n_complete": payload["n_complete"], "trials_reaching_cap": len(payload["trials_reaching_cap"]),
           "pod_commit": commit, "artifacts": arts, "hardware": hardware_info(device)}
    return write_result(rid, pay, config=cfg, results_dir=result_path(stage, rid).parent)


# ---------------------------------------------------------------------------------------------- dispatch
class TaskRunner:
    """Executes task jobs inside a queue worker, with lazily built per-process state."""

    def __init__(self, store, device):
        self.store, self.device = store, device
        self._sites = self._prim = self._relax = self._tuning = None

    @property
    def sites(self) -> infer.Sites:
        if self._sites is None:
            self._sites = infer.Sites(self.store)
        return self._sites

    @property
    def prim(self) -> dict:
        if self._prim is None:
            from dftgnn.train import runindex

            self._prim = runindex.primary(runindex.load())
        return self._prim

    def __call__(self, job: dict) -> dict:
        stage, rid, spec = job["stage"], job["run_id"], job["spec"]
        out = result_path(stage, rid)
        if verified(out, rid):
            return {"status": "skipped", "result": str(out)}
        if stage == "relax":
            if self._relax is None:
                self._relax = RelaxContext(str(self.device))
            path = run_relax(rid, spec, self._relax)
        elif stage == "embed":
            path = run_embed(rid, spec, self.sites, self.prim, self.device)
        elif stage == "geomeval":
            path = run_geomeval(rid, spec, self.sites, self.prim, job["key"], self.device)
        elif stage in TUNING_STAGES:
            if self._tuning is None:
                from dftgnn.tune import TuningStore

                self._tuning = TuningStore(resample=spec["r"])
            path = run_tuning(rid, job, self._tuning, self.device, log=lambda m: print(m, flush=True))
        else:
            raise ValueError(f"unknown task stage {stage!r}")
        return {"status": "done", "result": str(path)}
