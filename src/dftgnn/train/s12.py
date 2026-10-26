"""Stages and counts:
    relax     818  MLIP relaxation, tiling, geometry metrics and condition graphs per host (section 9)
    peval     183  P = P1 then D on the test hosts, 60 (r, B) x 3 seeds + 3 Kiyohara seeds (section 8)
    embed     240  S readout-input embeddings: 60 (r, B) x {trained, init0, init1, init2} (section 10)
    geomeval  180  {S, D, P} x 10 resamples x 3 seeds x B in {654, 200 (descriptive extra)}; after relax
    loco       30  5 LOCO folds x {S, D} x 3 seeds (section 11)
    moment     60  {S, D} x 10 resamples x 3 seeds at B = 654 without the 10 high-moment sites (section 12a)
    cgcnn      18  {cgcnn-S, cgcnn-D} x B in {50, 200, 654} x 3 seeds, resample 0 (section 12c)
Priority: the inference tranche (relax, peval, embed, geomeval) first, then loco, moment, cgcnn, capped
. Tranches: I (inference), L, M, C, X.

Inference stages reuse the checkpoints through ``dftgnn.train.runindex``; their run ids are task ids or P run
ids that include the current code SHA, so they are computed at enqueue time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dftgnn import infer, tasks
from dftgnn.config import Config, load_config
from dftgnn.models.cgcnn import CGCNNParams
from dftgnn.split import load_split
from dftgnn.train import LOCO_PREFIX, RunSpec, admission, runindex
from dftgnn.train.stages import _job, tuned_hparams

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ORDER = ("relax", "peval", "embed", "geomeval", "loco", "moment", "cgcnn", "capped")
TRANCHE = {"relax": "I", "peval": "I", "embed": "I", "geomeval": "I", "loco": "L", "moment": "M", "cgcnn": "C",
           "capped": "X"}
CGCNN_OPT = {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32}    # fixed, untuned (clarification 11)
GEO_BUDGETS = (654, 200)                                            # 200: descriptive extra (clarification 6)
CAPPED_SPEC = REPO_ROOT / "scripts" / "queue" / "capped_rerun_v1.json"
INFER_BATCH = 64


def _task(stage: str, key: dict, spec: dict, after=(), est=None) -> dict:
    j = {"run_id": infer.task_id(key), "stage": stage, "spec": spec, "after": list(after), "key": key}
    if est is not None:
        j["est_peak_gb"] = round(est, 3)
    return j


def _est_infer(table: dict | None, hp_list: list[dict]) -> float | None:
    if table is None:
        return None
    return sum(admission.est_peak_for_spec(table, hp, INFER_BATCH) for hp in hp_list)


# ----------------------------------------------------------------------------------------------- relax
def mace_smoke() -> dict:
    return json.loads((REPO_ROOT / "results" / "smoke" / "mace_smoke.json").read_text())["payload"]


def relax_est_gb(n_atoms: int, smoke: dict) -> float:
    """Smoke peak memory scaled by atom count (relative to the largest relaxed smoke cell) x 1.5."""
    n_ref = max(r["n_atoms_uc"] for r in smoke["relaxed"])
    return smoke["peak_rss_mb"] / 1024 * max(1.0, n_atoms / n_ref) * 1.5


def build_relax(code: str) -> list[dict]:
    from dftgnn.mlip import release

    smoke = mace_smoke()
    t = pd.read_csv(REPO_ROOT / "data" / "tiling_summary_v1.csv").set_index("host_id")
    jobs = []
    for h in release.host_table().host_id:
        n = int(t.loc[h, "n_uc"])
        spec = {"model": "relax", "host_id": h, "n_atoms_uc": n, "r": -1, "budget": -1, "seed": -1}
        jobs.append(_task("relax", tasks.relax_key(h, code), spec, est=relax_est_gb(n, smoke)))
    return jobs


# ----------------------------------------------------------------------------------------------- peval
def build_peval(code: str, gsha: str, tuned: dict, prim: dict, cfg: Config, table: dict | None) -> list[dict]:
    d = tuned["d_variant"]
    b2a = cfg.tuning.budget_to_anchor
    cells = [(f"outer_r{r}", r, b, s) for r in range(cfg.split.n_outer_resamples) for b in cfg.budgets.hosts
             for s in cfg.training.seeds]
    n_kiy = len(load_split("kiyohara")["train"])
    cells += [("kiyohara", -1, n_kiy, s) for s in cfg.training.seeds]
    jobs = []
    for split, r, b, s in cells:
        p1, dd = prim[("P1", split, b, s)], prim[(d, split, b, s)]
        anchor = 654 if split == "kiyohara" else b2a[b]
        spec = RunSpec(model="P", **tuned_hparams(tuned, d, anchor), split=split, r=r, budget=b, seed=s,
                       p1_run=p1["run_id"], d_run=dd["run_id"], results_subdir="peval", tags={"stage": "peval"})
        j = _job(spec, "peval", code, gsha, table=table)
        if table is not None:
            j["est_peak_gb"] = round(_est_infer(table, [p1["spec"]["hp"], dd["spec"]["hp"]]), 3)
        jobs.append(j)
    return jobs


# ----------------------------------------------------------------------------------------------- embed
def build_embed(code: str, prim: dict, cfg: Config, table: dict | None) -> list[dict]:
    from dftgnn.embeddings import KINDS

    jobs = []
    for r in range(cfg.split.n_outer_resamples):
        for b in cfg.budgets.hosts:
            rec = prim[("S", f"outer_r{r}", b, 0)]
            for kind in KINDS:
                spec = {"model": "embed", "r": r, "budget": b, "seed": 0, "kind": kind}
                jobs.append(_task("embed", tasks.embed_key(r, b, kind, rec["run_id"], code), spec,
                                  est=_est_infer(table, [rec["spec"]["hp"]])))
    return jobs


# -------------------------------------------------------------------------------------------- geomeval
def build_geomeval(code: str, prim: dict, relax_jobs: list[dict], cfg: Config, table: dict | None) -> list[dict]:
    from dftgnn.mlip import geomeval as GE

    excluded = set(pd.read_csv(GE.TILING_SUMMARY).query("excluded_from_C3b").host_id)
    relax_id = {j["spec"]["host_id"]: j["run_id"] for j in relax_jobs}
    jobs = []
    for m in ("S", "D", "P"):
        for r in range(cfg.split.n_outer_resamples):
            after = sorted(relax_id[h] for h in load_split(f"outer_r{r}")["test"] if h not in excluded)
            for b in GEO_BUDGETS:
                for s in cfg.training.seeds:
                    comps = GE.components(prim, m, r, b, s)
                    spec = {"model": "geomeval", "geo_model": m, "r": r, "budget": b, "seed": s,
                            "label": None if b in GE.REGISTERED_BUDGETS else "DESCRIPTIVE EXTRA"}
                    jobs.append(_task("geomeval", tasks.geomeval_key(m, r, b, s, comps, code), spec, after=after,
                                      est=_est_infer(table, [c["spec"]["hp"] for c in comps.values()])))
    return jobs


# ------------------------------------------------------------------------------------------- training
def build_loco(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    d = tuned["d_variant"]
    folds = load_split("loco_v1")["folds"]
    jobs = []
    for f in folds:
        for s in cfg.training.seeds:
            for m in ("S", d):
                spec = RunSpec(model=m, **tuned_hparams(tuned, m, 654), split=f"{LOCO_PREFIX}{f['fold']}",
                               r=f["fold"], budget=f["n_train"], seed=s, results_subdir="loco", tags={"loco": "v1"})
                jobs.append(_job(spec, "loco", code, gsha, table=table))
    return jobs


def high_moment_sites(cfg: Config) -> list[str]:
    u = pd.read_parquet(REPO_ROOT / "data" / "processed" / "universe_v1.parquet")
    thr = cfg.sensitivity.exclude_high_moment.threshold_muB
    return sorted(u.site_id[u.defect_moment_muB.abs() > thr])


def build_moment(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None, *, sites: list[str] | None = None,
                 subdir: str = "moment", stage: str = "moment", tags: dict | None = None) -> list[dict]:
    """``sites`` (default: from the universe), ``subdir``, ``stage`` and extra ``tags`` serve the v2 rerun
    (``dftgnn.train.nrp``."""
    d = tuned["d_variant"]
    sites = high_moment_sites(cfg) if sites is None else sorted(sites)
    if len(sites) != 10:
        raise ValueError(f"expected 10 high-moment entries, found {len(sites)}")
    b = cfg.sensitivity.exclude_high_moment.budget
    jobs = []
    for r in range(cfg.split.n_outer_resamples):
        for s in cfg.training.seeds:
            for m in ("S", d):
                spec = RunSpec(model=m, **tuned_hparams(tuned, m, cfg.tuning.budget_to_anchor[b]), split=f"outer_r{r}",
                               r=r, budget=b, seed=s, exclude_sites=sites, results_subdir=subdir,
                               tags={"sensitivity": "high_moment", **(tags or {})})
                jobs.append(_job(spec, stage, code, gsha, table=table))
    return jobs


def build_cgcnn(code: str, gsha: str, cfg: Config, table: dict | None) -> list[dict]:
    cc = cfg.sensitivity.cgcnn_check
    jobs = []
    for r in cc.resamples:
        for b in cc.budgets:
            for s in cc.seeds:
                for m in ("cgcnn-S", "cgcnn-D"):
                    spec = RunSpec(model=m, hp=CGCNNParams().to_dict(), **CGCNN_OPT, split=f"outer_r{r}", r=r,
                                   budget=b, seed=s, results_subdir="cgcnn", tags={"sensitivity": "cgcnn"})
                    jobs.append(_job(spec, "cgcnn", code, gsha, table=table))
    return jobs


def capped_twins() -> tuple[list[dict], list[dict]]:
    """(the 33 twin specs to run, the 3 skipped because a capsens run duplicates them)."""
    spec = json.loads(CAPPED_SPEC.read_text())
    run = [j for j in spec["jobs"] if not j.get("existing_600_epoch_twin")]
    skip = [j for j in spec["jobs"] if j.get("existing_600_epoch_twin")]
    return run, skip


def build_capped(code: str, gsha: str, table: dict | None) -> list[dict]:
    run, _ = capped_twins()
    jobs = []
    for j in run:
        spec = RunSpec.from_dict({**j["twin_spec"], "tags": {**j["twin_spec"].get("tags", {}),
                                                             "original_run_id": j["original_run_id"]}})
        jobs.append(_job(spec, "capped", code, gsha, table=table))
    return jobs


def schedule(jobs: list[dict]) -> list[dict]:
    """Attach ``priority`` (0 first) and ``tranche``. Within a stage: deterministic order of the builder."""
    out = []
    for i, j in enumerate(sorted(jobs, key=lambda j: STAGE_ORDER.index(j["stage"]))):
        out.append({**j, "priority": i, "tranche": TRANCHE[j["stage"]]})
    return out


def build_s12(code: str, gsha: str, tuned: dict, cfg: Config | None = None, table: dict | None = None,
              stages=STAGE_ORDER) -> list[dict]:
    cfg = cfg if cfg is not None else load_config()
    prim = runindex.primary(runindex.load())
    relax = build_relax(code)
    built = {"relax": relax,
             "peval": lambda: build_peval(code, gsha, tuned, prim, cfg, table),
             "embed": lambda: build_embed(code, prim, cfg, table),
             "geomeval": lambda: build_geomeval(code, prim, relax, cfg, table),
             "loco": lambda: build_loco(code, gsha, tuned, cfg, table),
             "moment": lambda: build_moment(code, gsha, tuned, cfg, table),
             "cgcnn": lambda: build_cgcnn(code, gsha, cfg, table),
             "capped": lambda: build_capped(code, gsha, table)}
    jobs = []
    for s in stages:
        jobs += built[s] if s == "relax" else built[s]()
    if "geomeval" in stages and "relax" not in stages:
        raise ValueError("geomeval depends on relax")
    return schedule(jobs)


# ------------------------------------------------------------------------------------------- cost model
def _interp_e_conv(budget: int, conv: dict[int, int]) -> float:
    bs = sorted(conv)
    return float(np.interp(math.log(budget), [math.log(b) for b in bs], [conv[b] for b in bs]))


def _scale(budget: int, scale: dict[str, float]) -> float:
    keys = sorted(int(k) for k in scale if k.isdigit())
    if budget in keys:
        return scale[str(budget)]
    return float(np.interp(math.log(budget), [math.log(k) for k in keys], [scale[str(k)] for k in keys]))


def training_gpu_h(budget: int, max_epochs: int, cost: dict) -> tuple[float, float]:
    """(central, upper) GPU-hours of one training run scaling."""
    conv = {int(k): v for k, v in cost["e_conv"].items()}
    sc = _scale(budget, cost["epoch_scale_vs_654_train_epoch"])
    epochs_c = min(max_epochs, _interp_e_conv(budget, conv) + cost["patience"])
    cells = cost["s_per_epoch_cells"]
    return (epochs_c * cells["mean"] * sc / cost["central_speedup_used"] / 3600,
            max_epochs * cells["slowest"] * sc / 3600)


def _timing() -> dict:
    return json.loads((REPO_ROOT / "results" / "smoke" / "inference_timing.json").read_text())["payload"]


def estimate(jobs: list[dict], workers: int = 4) -> dict:
    """GPU-hours (central, upper) per stage and in total, the cost at the cost-table rate and the guard.

    Training stages (loco, moment, cgcnn): scaling per run (central: mean cell, expected epochs, 1.23x MPS
    concurrency; upper: slowest cell, max_epochs, no concurrency); cgcnn is timed with the MEGNet cell, as in
    the cost table. capped: the per-job estimates of scripts/queue/capped_rerun_v1.json. Inference stages:
    measured CPU (3 threads) seconds per site-forward from the timing smoke, central with the work spread over
    ``workers`` processes, upper serial with the slowest measured model. relax: the MACE smoke step-time
    model, central 30 FIRE steps per host, upper 500 steps per host (the cap), both over ``workers``
    processes. Pod wall time is billed, so a pod-hour counts as one GPU-hour.
    """
    cost = json.loads((REPO_ROOT / "results" / "cost_table.json").read_text())["payload"]
    tm = _timing()
    t_c, t_u = tm["s_per_site_forward"]["mean"], tm["s_per_site_forward"]["max"]
    u = pd.read_parquet(REPO_ROOT / "data" / "processed" / "universe_v1.parquet")
    n_sites = u.groupby("host_id").size()
    excluded = set(pd.read_csv(REPO_ROOT / "data" / "tiling_summary_v1.csv").query("excluded_from_C3b").host_id)
    capped = {j["original_run_id"]: j["est_gpu_h"] for j in json.loads(CAPPED_SPEC.read_text())["jobs"]}
    st: dict[str, dict] = {}

    def add(stage, c, up):
        e = st.setdefault(stage, {"jobs": 0, "central_gpu_h": 0.0, "upper_gpu_h": 0.0})
        e["jobs"] += 1
        e["central_gpu_h"] += c
        e["upper_gpu_h"] += up

    for j in jobs:
        s, sp = j["stage"], j["spec"]
        if s in ("loco", "moment", "cgcnn"):
            add(s, *training_gpu_h(sp["budget"], sp.get("max_epochs") or cost["max_epochs"], cost))
        elif s == "capped":
            e = capped[sp["tags"]["original_run_id"]]
            add(s, e["central_600_epochs"], e["upper_600_epochs_slowest_cell"])
        elif s == "relax":
            per = 0.068 + 0.0111 * sp["n_atoms_uc"]
            add(s, 30 * per / workers / 3600, 500 * per / workers / 3600)
        else:
            if s == "peval":
                test = load_split(sp["split"])["test"]
                n, k = int(n_sites[test].sum()), 2
            elif s == "embed":
                from dftgnn.split import budget_train

                spl = load_split(f"outer_r{sp['r']}")
                n, k = int(n_sites[budget_train(spl, sp["budget"])].sum() + n_sites[spl["test"]].sum()), 1
            else:                                            # geomeval: 3 conditions, P = P1 + D
                test = [h for h in load_split(f"outer_r{sp['r']}")["test"] if h not in excluded]
                n, k = 3 * int(n_sites[test].sum()), 2 if sp["geo_model"] == "P" else 1
            add(s, n * k * t_c / workers / 3600, n * k * t_u / 3600)
    c = sum(e["central_gpu_h"] for e in st.values())
    up = sum(e["upper_gpu_h"] for e in st.values())
    rate = cost["rate_usd_per_h"]
    return {"stages": st, "central_gpu_h": c, "upper_gpu_h": up, "rate_usd_per_h": rate, "central_usd": c * rate,
            "upper_usd": up * rate, "guard_gpu_h": 1.1 * up,
            "notes": ["inference stages from CPU timings (3 threads); the L40S is expected to be faster",
                      "relax upper assumes every host runs the 500-step cap",
                      f"timing basis: {tm['basis']}"]}
