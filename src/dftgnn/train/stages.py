"""Job-spec builders for the queue stages: ``smoke`` and ``kiyohara``, ``pilot_epochs``, ``c0_pilot`` and
``c0_ablate``, ``sweep``.

Every job is ``{"run_id", "stage", "spec", "after", "est_peak_gb"}``. The run id is computed at enqueue
time from the code SHA and the graphs manifest sha256, so the pod computes the same id from the same
commit. ``est_peak_gb`` is the admission estimate (``dftgnn.train.admission``) when a benchmark table
is passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dftgnn.config import Config, load_config
from dftgnn.models import HParams
from dftgnn.split import budget_train, load_split
from dftgnn.train import RunSpec, admission, run_id, val_seed

STAGES = ("tune", "sweep", "kiyohara", "loco", "sensitivity", "smoke", "pilot_epochs", "c0_pilot", "c0_ablate",
          "c0_capcheck", "c0_official", "capsens", "session",
          "s12", "relax", "peval", "embed", "geomeval", "moment", "cgcnn", "capped",    # Dftgnn.train.s12
          "diag_cross", "d_ablation", "dlate", "review", "v2_screen", "v2", "p_v2")    # collaborator NRP stages
# smoke only: fixed, untuned optimiser settings for a pipeline check (not a hyperparameter choice)
SMOKE_OPT = {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 16}
SMOKE_EPOCHS = 3
SMOKE_SIZES = {"train": 20, "val": 5, "test": 10}


def _job(spec: RunSpec, stage: str, code: str, gsha: str, after=(), table: dict | None = None) -> dict:
    job = {"run_id": run_id(spec, code, gsha), "stage": stage, "spec": spec.to_dict(), "after": list(after)}
    if table is not None:
        job["est_peak_gb"] = round(admission.est_peak_for_spec(table, spec.hp, spec.batch_size), 3)
    return job


def smoke_hosts(r: int = 0, budget: int = 25) -> dict[str, list[str]]:
    """20 train + 5 val hosts from resample r's budget-25 set, 10 test hosts from its test set."""
    sp = load_split(f"outer_r{r}")
    pool = sorted(budget_train(sp, budget))
    rng = np.random.default_rng(val_seed(r, budget, 0))
    perm = rng.permutation(len(pool))
    val = sorted(pool[i] for i in perm[: SMOKE_SIZES["val"]])
    train = sorted(set(pool) - set(val))
    test = sorted(rng.choice(sorted(sp["test"]), size=SMOKE_SIZES["test"], replace=False).tolist())
    return {"train": train, "val": val, "test": test}


def build_smoke(code: str, gsha: str, table: dict | None = None) -> list[dict]:
    hosts = smoke_hosts()
    base = {"hp": HParams().to_dict(), **SMOKE_OPT, "split": "outer_r0", "r": 0, "budget": 25,
            "seed": 0, "hosts": hosts, "max_epochs": SMOKE_EPOCHS, "patience": SMOKE_EPOCHS,
            "smoke": True}
    jobs = {m: _job(RunSpec(model=m, **base), "smoke", code, gsha, table=table)
            for m in ("S", "D-state", "D-late", "P1")}
    p = RunSpec(model="P", **base, p1_run=jobs["P1"]["run_id"], d_run=jobs["D-state"]["run_id"])
    return [*jobs.values(), _job(p, "smoke", code, gsha, after=[jobs["P1"]["run_id"],
                                                               jobs["D-state"]["run_id"]], table=table)]


def build_kiyohara(code: str, gsha: str, hparams: dict, cfg: Config | None = None,
                   d_variant: str | None = None, table: dict | None = None) -> list[dict]:
    """S, D and P on the Kiyohara split, seeds ``training.seeds``.

    ``hparams[model]`` = {"hp": {...}, "lr", "weight_decay", "batch_size"} from tuning, for S, the D variant
    and P1; the budget's anchor is 654. D is ``models.injection_mode`` unless ``d_variant`` is given.
    ``max_epochs`` and the early-stopping patience must be set in the config.
    """
    cfg = cfg if cfg is not None else load_config()
    if "TBD-S07" in (cfg.training.max_epochs, cfg.training.early_stopping.patience):
        raise ValueError("training.max_epochs / early_stopping.patience are still TBD-S07")
    d = d_variant or cfg.models.injection_mode
    if d not in ("D-state", "D-late"):
        raise ValueError("D variant undecided: set models.injection_mode or pass d_variant")
    models = []
    for m in cfg.sensitivity.kiyohara_split.models:
        models += {"S": ["S"], "D": [d], "P": ["P1"]}[m]
    missing = [m for m in models if m not in hparams]
    if missing:
        raise ValueError(f"no tuned hyperparameters for {missing}")
    n_train = len(load_split("kiyohara")["train"])
    jobs = []
    for seed in cfg.training.seeds:
        made = {}
        for m in models:
            h = hparams[m]
            spec = RunSpec(model=m, hp=h["hp"], lr=h["lr"], weight_decay=h["weight_decay"],
                           batch_size=h["batch_size"], split="kiyohara", r=-1, budget=n_train, seed=seed)
            made[m] = _job(spec, "kiyohara", code, gsha, table=table)
        jobs += made.values()
        if "P" in cfg.sensitivity.kiyohara_split.models:
            h = hparams[d]
            p = RunSpec(model="P", hp=h["hp"], lr=h["lr"], weight_decay=h["weight_decay"],
                        batch_size=h["batch_size"], split="kiyohara", r=-1, budget=n_train, seed=seed,
                        p1_run=made["P1"]["run_id"], d_run=made[d]["run_id"])
            jobs.append(_job(p, "kiyohara", code, gsha, after=[made["P1"]["run_id"], made[d]["run_id"]],
                             table=table))
    return jobs


def load_hparams(path: Path) -> dict:
    return json.loads(Path(path).read_text())


TUNED = Path(__file__).resolve().parents[3] / "configs" / "tuned_v1.yaml"
OPT_KEYS = ("learning_rate", "weight_decay", "batch_size")


def load_tuned(path: Path = TUNED) -> dict:
    """``configs/tuned_v1.yaml``: ``{"d_variant", "models": {model: {anchor: search-space params}}}``."""
    import yaml

    t = yaml.safe_load(Path(path).read_text())
    t["models"] = {m: {int(a): p for a, p in by.items()} for m, by in t["models"].items()}
    return t


def tuned_hparams(tuned: dict, model: str, anchor: int) -> dict:
    """Tuned search-space parameters -> {"hp", "lr", "weight_decay", "batch_size"} for a RunSpec."""
    p = tuned["models"][model][anchor]
    hp = {k: v for k, v in p.items() if k not in OPT_KEYS}
    hp.update(tuned.get("arch") or {})          # v2: fixed backbone options (configs/tuned_v2.yaml)
    return {"hp": hp, "lr": float(p["learning_rate"]),
            "weight_decay": float(p["weight_decay"]), "batch_size": int(p["batch_size"])}


def build_sweep(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                table: dict | None = None) -> list[dict]:
    """The primary sweep: S, D (the selected injection variant) and P1 for every outer resample, budget and seed,
    plus one P evaluation per (r, B, seed) that depends on its P1 and D runs. Each run takes the tuned
    hyperparameters of its budget's anchor (``tuning.budget_to_anchor``). Validation hosts and run ids follow
    the same rules as every other stage (``val_seed``, ``run_id``)."""
    cfg = cfg if cfg is not None else load_config()
    if "TBD-S07" in (cfg.training.max_epochs, cfg.training.early_stopping.patience):
        raise ValueError("training.max_epochs / early_stopping.patience are still TBD-S07")
    d = tuned["d_variant"]
    if d not in ("D-state", "D-late") or cfg.models.injection_mode != d:
        raise ValueError(f"D variant {d!r} does not match models.injection_mode {cfg.models.injection_mode!r}")
    b2a = cfg.tuning.budget_to_anchor
    jobs = []
    for r in range(cfg.split.n_outer_resamples):
        for b in cfg.budgets.hosts:
            for seed in cfg.training.seeds:
                made = {}
                for m in ("S", d, "P1"):
                    spec = RunSpec(model=m, **tuned_hparams(tuned, m, b2a[b]), split=f"outer_r{r}", r=r,
                                   budget=b, seed=seed)
                    made[m] = _job(spec, "sweep", code, gsha, table=table)
                jobs += made.values()
                p = RunSpec(model="P", **tuned_hparams(tuned, d, b2a[b]), split=f"outer_r{r}", r=r, budget=b,
                            seed=seed, p1_run=made["P1"]["run_id"], d_run=made[d]["run_id"])
                jobs.append(_job(p, "sweep", code, gsha, after=[made["P1"]["run_id"], made[d]["run_id"]],
                                 table=table))
    return jobs


def build_capsens(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                  table: dict | None = None) -> list[dict]:
    """Epoch-cap sensitivity: S and D at B = 654, resamples and seeds of ``sensitivity.epoch_cap``,
    ``max_epochs`` 600 and the same patience, with the a654 tuned hyperparameters. The epoch override enters
    the run id, so none of these is a twin of a 200-epoch run; results go to ``results/capsens/``."""
    cfg = cfg if cfg is not None else load_config()
    ec = cfg.sensitivity.epoch_cap
    d = tuned["d_variant"]
    anchor = cfg.tuning.budget_to_anchor[ec.budget]
    jobs = []
    for r in ec.resamples:
        for seed in ec.seeds:
            for m in ec.models:
                model = d if m == "D" else m
                spec = RunSpec(model=model, **tuned_hparams(tuned, model, anchor), split=f"outer_r{r}", r=r,
                               budget=ec.budget, seed=seed, max_epochs=ec.max_epochs, patience=ec.patience,
                               results_subdir="capsens", tags={"capsens": f"max_epochs_{ec.max_epochs}"})
                jobs.append(_job(spec, "capsens", code, gsha, table=table))
    return jobs


def build_diag_cross(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                     table: dict | None = None) -> list[dict]:
    """Item 1 diagnostic: S at each (budget, anchor) cell of ``diagnostics.hparam_cross``, so the within-host
    collapse can be attributed to the budget or to the tuned values. Cells with budget == anchor repeat sweep
    runs under this code SHA and give checkpoints."""
    cfg = cfg if cfg is not None else load_config()
    hc = cfg.diagnostics.hparam_cross
    jobs = []
    for b, a in hc.cells:
        for r in hc.resamples:
            for seed in hc.seeds:
                spec = RunSpec(model=hc.model, **tuned_hparams(tuned, hc.model, a), split=f"outer_r{r}", r=r,
                               budget=b, seed=seed, results_subdir="diag_cross",
                               tags={"diagnostic": "hparam_cross", "anchor": a})
                jobs.append(_job(spec, "diag_cross", code, gsha, table=table))
    return jobs


def build_d_ablation(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                     table: dict | None = None) -> list[dict]:
    """Item 3 diagnostic: the selected D at ``diagnostics.descriptor_ablation.budget`` with no ablation
    (checkpoints for the permutation importances) and with each descriptor class at its training mean."""
    cfg = cfg if cfg is not None else load_config()
    da = cfg.diagnostics.descriptor_ablation
    d = tuned["d_variant"]
    anchor = cfg.tuning.budget_to_anchor[da.budget]
    jobs = []
    for ab in da.ablations:
        for r in da.resamples:
            for seed in da.seeds:
                spec = RunSpec(model=d, **tuned_hparams(tuned, d, anchor), split=f"outer_r{r}", r=r,
                               budget=da.budget, seed=seed, ablate=None if ab == "none" else ab,
                               results_subdir="d_ablation", tags={"diagnostic": "descriptor_ablation"})
                jobs.append(_job(spec, "d_ablation", code, gsha, table=table))
    return jobs


def build_dlate(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                table: dict | None = None) -> list[dict]:
    """Sensitivity to the D-variant tie: the unselected variant trained exactly as the sweep trains D (its own
    tuned values at the budget's anchor), at the budgets, resamples and seeds of
    ``sensitivity.unselected_variant_sweep``. Results go to ``results/dlate/``."""
    cfg = cfg if cfg is not None else load_config()
    us = cfg.sensitivity.unselected_variant_sweep
    if us.variant == tuned["d_variant"]:
        raise ValueError(f"{us.variant} is the selected D, not the unselected variant")
    b2a = cfg.tuning.budget_to_anchor
    jobs = []
    for b in us.budgets:
        for r in us.resamples:
            for seed in us.seeds:
                spec = RunSpec(model=us.variant, **tuned_hparams(tuned, us.variant, b2a[b]), split=f"outer_r{r}",
                               r=r, budget=b, seed=seed, results_subdir="dlate",
                               tags={"sensitivity": "unselected_variant"})
                jobs.append(_job(spec, "dlate", code, gsha, table=table))
    return jobs


def build_review(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                 table: dict | None = None) -> list[dict]:
    """Every job requested by the 2026-10-07 review, prioritised: the two diagnostics (largest budget
    first, so the longest runs start early), then the D-late sensitivity by budget descending."""
    diag = build_diag_cross(code, gsha, tuned, cfg, table) + build_d_ablation(code, gsha, tuned, cfg, table)
    order = sorted(diag, key=lambda j: (-j["spec"]["budget"], j["stage"], j["spec"]["r"], j["spec"]["seed"]))
    order += sorted(build_dlate(code, gsha, tuned, cfg, table),
                    key=lambda j: (-j["spec"]["budget"], j["spec"]["r"], j["spec"]["seed"]))
    for i, j in enumerate(order):
        j["priority"] = i
    return order


def build_loco(code: str, gsha: str, tuned: dict, cfg: Config | None = None, table: dict | None = None,
               arch: dict | None = None, subdir: str = "loco") -> list[dict]:
    """Leave-chemistry-out: S and the selected D on every fold of ``splits/loco/``, all non-held-out hosts as
    training hosts, seeds ``robustness.loco.seeds``, the 654-host anchor's tuned values (the folds train on
    654-655 hosts). ``arch`` adds backbone options (the v2 model); results go to ``results/<subdir>/``."""
    cfg = cfg if cfg is not None else load_config()
    lc = cfg.robustness.loco
    d = tuned["d_variant"]
    jobs = []
    for k in range(lc.k):
        sp = load_split(f"loco/loco_f{k}")
        for seed in lc.seeds:
            for m in lc.models:
                model = d if m == "D" else m
                h = tuned_hparams(tuned, model, 654)
                h["hp"] = {**h["hp"], **(arch or {})}
                spec = RunSpec(model=model, **h, split=f"loco/loco_f{k}", r=k, budget=len(sp["budget_order"]),
                               seed=seed, results_subdir=subdir, tags={"loco_fold": k})
                jobs.append(_job(spec, "loco", code, gsha, table=table))
    return jobs


def build_v2_screen(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                    table: dict | None = None) -> list[dict]:
    """Backbone design screen: model S with each variant of ``v2.variants`` at the tuning anchors of resample 0,
    seeds ``v2.screen_seeds``, the anchor's tuned values plus the variant's options. Validation hosts only:
    ``eval_test=False``, so the test hosts are never loaded. Results go to ``results/v2_screen/``."""
    cfg = cfg if cfg is not None else load_config()
    v2 = cfg.v2
    jobs = []
    for name, opts in v2.variants.items():
        for a in v2.screen_anchors:
            for seed in v2.screen_seeds:
                h = tuned_hparams(tuned, "S", a)
                h["hp"] = {**h["hp"], **opts}
                spec = RunSpec(model="S", **h, split=f"outer_r{cfg.tuning.anchor_resample}",
                               r=cfg.tuning.anchor_resample, budget=a, seed=seed, eval_test=False,
                               results_subdir="v2_screen", tags={"v2_variant": name, "anchor": a})
                jobs.append(_job(spec, "v2_screen", code, gsha, table=table))
    return jobs


TUNED_V2 = Path(__file__).resolve().parents[3] / "configs" / "tuned_v2.yaml"


def build_v2(code: str, gsha: str, tuned: dict, cfg: Config | None = None, table: dict | None = None) -> list[dict]:
    """The v2 production runs, from ``configs/tuned_v2.yaml``: S-v2 and D-v2 over every outer resample, budget
    and seed (``results/v2_sweep/``), on the Kiyohara split (``results/v2_kiyohara/``) and on the LOCO folds
    (``results/loco_v2/``). Prioritised longest first: LOCO and Kiyohara, then the sweep by budget descending."""
    cfg = cfg if cfg is not None else load_config()
    if not tuned.get("arch"):
        raise ValueError("not a v2 tuned file (no arch)")
    d = tuned["d_variant"]
    b2a = cfg.tuning.budget_to_anchor
    sweep = []
    for r in range(cfg.split.n_outer_resamples):
        for b in cfg.budgets.hosts:
            for seed in cfg.training.seeds:
                for m in ("S", d):
                    spec = RunSpec(model=m, **tuned_hparams(tuned, m, b2a[b]), split=f"outer_r{r}", r=r, budget=b,
                                   seed=seed, results_subdir="v2_sweep", tags={"v2": tuned.get("version", "")})
                    sweep.append(_job(spec, "v2_sweep", code, gsha, table=table))
    n_train = len(load_split("kiyohara")["train"])
    kiy = [_job(RunSpec(model=m, **tuned_hparams(tuned, m, 654), split="kiyohara", r=-1, budget=n_train, seed=seed,
                        results_subdir="v2_kiyohara", tags={"v2": tuned.get("version", "")}),
                "v2_kiyohara", code, gsha, table=table)
           for seed in cfg.training.seeds for m in ("S", d)]
    loco = build_loco(code, gsha, tuned, cfg, table, subdir="loco_v2")
    for j in loco:
        j["stage"] = "loco_v2"
    jobs = loco + kiy + sorted(sweep, key=lambda j: (-j["spec"]["budget"], j["spec"]["r"], j["spec"]["seed"]))
    for i, j in enumerate(jobs):
        j["priority"] = i
    return jobs


def build_p_v2(code: str, gsha: str, tuned_v2: dict, tuned_v1: dict, cfg: Config | None = None,
               table: dict | None = None) -> list[dict]:
    """P1-v2 and the staged P-v2. P-v2: each P1-v2 run composed with the D-v2 run of the same split, budget and
    seed, taken from ``build_v2`` so the run ids match the queued v2 stage (``results/p_v2/``; runs after
    both)."""
    cfg = cfg if cfg is not None else load_config()
    d = tuned_v2["d_variant"]
    arch = tuned_v2.get("arch") or {}
    dv2 = {(j["spec"]["split"], j["spec"]["budget"], j["spec"]["seed"]): j
           for j in build_v2(code, gsha, tuned_v2, cfg, table) if j["spec"]["model"] == d
           and j["stage"] in ("v2_sweep", "v2_kiyohara")}
    b2a = cfg.tuning.budget_to_anchor
    p1_jobs, p_jobs = [], []
    for (split, b, seed), dj in sorted(dv2.items(), key=lambda kv: (-kv[0][1], kv[0][0], kv[0][2])):
        anchor = 654 if split == "kiyohara" else b2a[b]
        h = tuned_hparams(tuned_v1, "P1", anchor)
        h["hp"] = {**h["hp"], **arch}
        r = dj["spec"]["r"]
        p1 = _job(RunSpec(model="P1", **h, split=split, r=r, budget=b, seed=seed, results_subdir="p1_v2",
                          tags={"v2": tuned_v2.get("version", "")}), "p1_v2", code, gsha, table=table)
        ds = dj["spec"]
        p = _job(RunSpec(model="P", hp=ds["hp"], lr=ds["lr"], weight_decay=ds["weight_decay"],
                         batch_size=ds["batch_size"], split=split, r=r, budget=b, seed=seed, results_subdir="p_v2",
                         p1_run=p1["run_id"], d_run=dj["run_id"], tags={"v2": tuned_v2.get("version", "")}),
                 "p_v2", code, gsha, after=[p1["run_id"], dj["run_id"]], table=table)
        p1_jobs.append(p1)
        p_jobs.append(p)
    jobs = p1_jobs + p_jobs
    for i, j in enumerate(jobs):
        j["priority"] = 1000 + i                    # behind every job of the v2 stage
    return jobs


def _is_p1(j: dict) -> bool:
    return j["spec"]["model"] == "P1"


def schedule(jobs: list[dict], cfg: Config | None = None) -> list[dict]:
    """Order: (1) Kiyohara-split S and D; (2) sweep S and D by budget descending, then resample, then seed;
    (3) epoch-cap runs; (4) P1 runs (Kiyohara split first, then sweep by budget descending). Tranches: T1
    Kiyohara S and D and sweep S and D at B = 654 and 400; T2 sweep S and D at the other budgets; T3 the
    epoch-cap runs; T4 every P1 run. P evaluation jobs are not scheduled.
    """
    mo = {"S": 0, "D-state": 1, "D-late": 1}

    def key(j):
        sp, m = j["spec"], j["spec"]["model"]
        if j["stage"] == "capsens":
            return (3, sp["r"], sp["seed"], mo[m])
        kiy = sp["split"] == "kiyohara"
        if _is_p1(j):
            return (4, 0 if kiy else 1, -sp["budget"], sp["r"], sp["seed"])
        if kiy:
            return (1, sp["seed"], mo[m])
        return (2, -sp["budget"], sp["r"], sp["seed"], mo[m])

    def tranche(j):
        sp = j["spec"]
        if j["stage"] == "capsens":
            return "T3"
        if _is_p1(j):
            return "T4"
        return "T1" if sp["split"] == "kiyohara" or sp["budget"] in (654, 400) else "T2"

    out = sorted((dict(j) for j in jobs), key=key)
    for i, j in enumerate(out):
        j["priority"], j["tranche"] = i, tranche(j)
    return out


def build_session(code: str, gsha: str, tuned: dict, cfg: Config | None = None, table: dict | None = None,
                  c0_code: str | None = None) -> tuple[list[dict], list[dict]]:
    """No P evaluation. The Kiyohara-split S runs are the official runs (``build_c0_official`` at ``c0_code``,
    the code SHA that produced them): they are not rebuilt and are returned as the second list (``reused``).
    Returns ``(jobs, reused)``."""
    cfg = cfg if cfg is not None else load_config()
    jobs = [j for j in build_sweep(code, gsha, tuned, cfg, table) if j["spec"]["model"] != "P"]
    kiy = [j for j in build_kiyohara(code, gsha, kiyohara_hparams_from_tuned(tuned), cfg,
                                     d_variant=tuned["d_variant"], table=table) if j["spec"]["model"] != "P"]
    reused = []
    if c0_code is not None:
        reused = build_c0_official(c0_code, gsha, tuned, cfg, table)
        kiy = [j for j in kiy if j["spec"]["model"] != "S"]
    jobs += kiy + build_capsens(code, gsha, tuned, cfg, table)
    return schedule(jobs, cfg), reused


def kiyohara_hparams_from_tuned(tuned: dict, anchor: int = 654) -> dict:
    """``build_kiyohara``'s hparams: the 654-host anchor's values."""
    d = tuned["d_variant"]
    return {m: tuned_hparams(tuned, m, anchor) for m in ("S", d, "P1")}


# Untuned mid-space hyperparameters of the pilots. Logged in before the runs.
PILOT_HP = {"hidden_width": 64, "megnet_blocks": 3, "dropout": 0.1, "readout_mlp_width": 64,
            "pooling": "set2set"}
PILOT_OPT = {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32}
PILOT_BUDGETS = (25, 200, 654)
PILOT_MAX_EPOCHS = 1500


def build_pilot_epochs(code: str, gsha: str, table: dict | None = None) -> list[dict]:
    """Model S on resample 0's budget-25/200/654 sets, seed 0, 1500 epochs, early stopping disabled
    (patience = max_epochs), validation MAE every epoch. The test hosts are never loaded."""
    jobs = []
    for b in PILOT_BUDGETS:
        spec = RunSpec(model="S", hp=PILOT_HP, **PILOT_OPT, split="outer_r0", r=0, budget=b, seed=0,
                       max_epochs=PILOT_MAX_EPOCHS, patience=PILOT_MAX_EPOCHS, eval_test=False,
                       results_subdir="pilot_epochs", tags={"pilot": "epochs"})
        jobs.append(_job(spec, "pilot_epochs", code, gsha, table=table))
    return jobs


def build_c0_pilot(code: str, gsha: str, cfg: Config | None = None, table: dict | None = None,
                   ablate: str | None = None) -> list[dict]:
    """Model S on the Kiyohara split with the untuned mid-space hyperparameters, seeds ``training.seeds``, the
    max_epochs and patience set in the config. ``ablate='vacancy_flag'`` is the diagnostic that zeroes the
    vacancy flag."""
    cfg = cfg if cfg is not None else load_config()
    if "TBD-S07" in (cfg.training.max_epochs, cfg.training.early_stopping.patience):
        raise ValueError("training.max_epochs / early_stopping.patience are still TBD-S07")
    n_train = len(load_split("kiyohara")["train"])
    sub = "c0_pilot" if ablate is None else f"c0_pilot_ablate_{ablate}"
    jobs = []
    for seed in cfg.training.seeds:
        spec = RunSpec(model="S", hp=PILOT_HP, **PILOT_OPT, split="kiyohara", r=-1, budget=n_train,
                       seed=seed, ablate=ablate, results_subdir=sub, tags={"pilot": "c0"})
        jobs.append(_job(spec, "c0_pilot" if ablate is None else "c0_ablate", code, gsha, table=table))
    return jobs


def build_c0_capcheck(code: str, gsha: str, table: dict | None = None, seeds=(0, 2),
                      max_epochs: int = 600) -> list[dict]:
    """Excluded from every analysis."""
    cfg = load_config()
    n_train = len(load_split("kiyohara")["train"])
    return [_job(RunSpec(model="S", hp=PILOT_HP, **PILOT_OPT, split="kiyohara", r=-1, budget=n_train,
                         seed=seed, max_epochs=max_epochs, patience=cfg.training.early_stopping.patience,
                         results_subdir="c0_cap_check", tags={"pilot": "c0_cap_check"}),
                 "c0_capcheck", code, gsha, table=table) for seed in seeds]


def build_c0_official(code: str, gsha: str, tuned: dict, cfg: Config | None = None,
                      table: dict | None = None) -> list[dict]:
    """Build c0 official."""
    cfg = cfg if cfg is not None else load_config()
    n_train = len(load_split("kiyohara")["train"])
    h = tuned_hparams(tuned, "S", 654)
    return [_job(RunSpec(model="S", **h, split="kiyohara", r=-1, budget=n_train, seed=seed,
                         results_subdir="c0_official", tags={"c0": "official"}), "c0_official", code, gsha,
                 table=table) for seed in cfg.training.seeds]
