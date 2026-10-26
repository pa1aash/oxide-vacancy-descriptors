"""Job specs for the National Research Platform runs.

Every remaining stage is written once, on the Mac, as a spec file ``specs/<stage>.jsonl`` (one queue job per line)
with a manifest. Run ids are computed with a pinned code reference (``code_ref``, the git tag ``nrp-code-v1``)
instead of the generating commit, so the Mac and the cluster compute the same ids and the results merge by id.
Building specs is pure: it reads tracked files only (splits, tuned hyperparameters, committed results for the
checkpoint index, the GPU benchmark), and uses no GPU and no network.

Stages, programme group (priority order) and job counts:
    N2a  relax 818 (all hosts; ``hosts`` restricts it), peval 183, embed 240, geomeval 180          (v1)
    Moment 60, cgcnn 18, capped 33                                                   (v1)
    N2b  moment_v2 60: the moment jobs on the v2 backbone, same 10 excluded sites (specs/moment.jsonl)   (v2 only)
    Cv 180 per backbone (EXTENSION): {S, D} x K in {10, 20} x folds x seeds 0-2                  (v1 and v2)
    Precision 480 per backbone (EXTENSION): {S, D} x B in {400, 654} x (r = 10..29 x seeds 0-4
         + r = 0..9 x seeds 3, 4)                                                                     (v1 and v2)
    One Optuna study of P1 per tuning anchor (50, 200, 654), the S and D protocol of the v2
         tuning with an epoch limit of 400 (a task stage, ``dftgnn.tasks``)                          (v2 only)
    D with the host-level descriptors only (D-host) and with the site-level descriptors only
         (D-site), B in {200, 400, 654} x resamples 0-9 x seeds 0-2                                  (v2 only)
    P1 with the settings of configs/tuned_p1_v2.yaml and an epoch limit of 400, every budget x
         resamples 0-9 x seeds 0-2; pcompv2b 180: P = the new P1 then the existing v2 D of the same cell (v2 only)
    One Optuna study of D per tuning anchor in a superset of the v2 search space (wider dropout and
         weight decay, the injection variant, a descriptor encoder), epoch limit 400 (a task stage)     (v2 only)
    D with the settings of configs/tuned_d_v2.yaml and an epoch limit of 400 on the cells of
         the existing v2 D runs: B = 654 x the 30 x 5 grid (v2 sweep + precision), B = 400 and 200 x resamples
         0-9 x seeds 0-2                                                                              (v2 only)
The and stages reuse the builders (``dftgnn.train.s12``); moment_v2 passes its sites, subdir, stage
and tags to ``build_moment``, whose defaults keep the v1 ids. LOCO is not a stage: the
``configs/tuned_v2.yaml``: its a654 values plus the fixed ``arch`` options) writes to ``results/cv_v2/``,
``results/precision_v2/`` and ``results/moment_v2/``.

While that file is missing their builders refuse with a message naming it, and ``available_stages`` leaves
them out (the default output of ``make_specs.py``); so does dretrain_v2 before ``configs/tuned_d_v2.yaml``.
dclass keeps one descriptor class by setting the other to its training mean (the ``ablate`` option of
``dftgnn.train``, applied to the descriptor vector of every batch); the classes are those of
``docs/descriptor_taxonomy.csv``.
"""
from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter
from pathlib import Path

from dftgnn.config import Config, load_config
from dftgnn.split import SPLITS_DIR, budget_train, load_split
from dftgnn.split.cv import CV_KS, CV_NAME, split_name
from dftgnn.split.ext import EXT_RESAMPLES
from dftgnn.train import RunSpec, accepted_code_commits, admission, resolve_hosts, run_id
from dftgnn.train import s12 as S12
from dftgnn.train.stages import TUNED, _job, load_tuned, tuned_hparams

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS_DIR = REPO_ROOT / "specs"
CODE_REF = "nrp-code-v1"
# new stages are appended so that the priorities of the earlier stages (and their spec files) do not change
STAGES = ("relax", "peval", "embed", "geomeval", "moment", "cgcnn", "capped", "cv", "precision", "moment_v2",
          "p1tune_v2", "dclass", "p1v2b", "pcompv2b", "dtune_v2", "dretrain_v2")
GROUP = {"relax": "N2a", "peval": "N2a", "embed": "N2a", "geomeval": "N2a", "moment": "N2b", "cgcnn": "N2b",
         "capped": "N2b", "cv": "N3", "precision": "N4", "moment_v2": "N2b",
         "p1tune_v2": "N5", "dclass": "N5", "p1v2b": "N6", "pcompv2b": "N6", "dtune_v2": "N7", "dretrain_v2": "N8"}
LABEL = {"relax": "registered", "peval": "registered", "embed": "registered",
         "geomeval": "registered (B = 200 DESCRIPTIVE EXTRA)", "moment": "registered", "cgcnn": "registered",
         "capped": "POST HOC", "cv": "EXTENSION", "precision": "EXTENSION", "moment_v2": "POST HOC",
         "p1tune_v2": "POST HOC", "dclass": "POST HOC", "p1v2b": "POST HOC", "pcompv2b": "POST HOC",
         "dtune_v2": "POST HOC", "dretrain_v2": "POST HOC"}
EXPECTED = {"relax": 818, "peval": 183, "embed": 240, "geomeval": 180, "moment": 60, "cgcnn": 18, "capped": 33,
            "cv": 180, "precision": 480, "moment_v2": 60, "p1tune_v2": 3, "dclass": 180, "p1v2b": 180,
            "pcompv2b": 180, "dtune_v2": 3, "dretrain_v2": 210}
BACKBONES = ("v1", "v2")
V2_STAGES = ("cv", "precision", "moment_v2", "p1tune_v2", "dclass", "p1v2b", "pcompv2b", "dtune_v2",
             "dretrain_v2")                                                              # stages with a v2 version
V2_ONLY = ("moment_v2", "p1tune_v2", "dclass", "p1v2b", "pcompv2b", "dtune_v2", "dretrain_v2")   # no v1 version
TUNED_V2 = REPO_ROOT / "configs" / "tuned_v2.yaml"
TUNED_P1_V2 = REPO_ROOT / "configs" / "tuned_p1_v2.yaml"   # written on the Mac from the p1tune_v2 records
TUNED_D_V2 = REPO_ROOT / "configs" / "tuned_d_v2.yaml"     # written on the Mac from the dtune_v2 records
NEEDS_FILE = {"p1v2b": TUNED_P1_V2, "pcompv2b": TUNED_P1_V2, "dretrain_v2": TUNED_D_V2}
TAXONOMY = REPO_ROOT / "docs" / "descriptor_taxonomy.csv"
V2_SWEEP_DIR = REPO_ROOT / "results" / "v2_sweep"          # records of the v2 D runs that pcompv2b reuses
P_V2_DIR = REPO_ROOT / "results" / "p_v2"
P1V2B_DIR = REPO_ROOT / "results" / "p1v2b"                # ingested p1v2b records: the dependencies of pcompv2b
P1V2B_CHECKPOINTS = REPO_ROOT / "results" / "nrp" / "artefacts" / "p1v2b.sha256"   # "<sha256>  <size>  <path>" lines
V2_CKPT_DIRS = ("/workspace/dft-gnn-v2/checkpoints",)      # NRP volume holding the v2 sweep checkpoints
P1_EPOCHS = 400                                            # epoch limit of P1 in p1tune_v2 and p1v2b
DCLASS_BUDGETS = (654, 400, 200)
DCLASS = {"D-host": "desc_site", "D-site": "desc_host"}    # variant -> the class set to its training mean
V2_LABEL = "EXTENSION, v2 backbone (POST HOC)"
D_EPOCHS = 400                                             # epoch limit of D in dtune_v2 and dretrain_v2
# dtune_v2: changes to the configured search space (the space of every earlier study). Wider ranges contain the old
# ones; the new parameters contain the old network (injection D-state, desc_layers 0), so the space is a superset.
DTUNE_SPACE = {
    "dropout": {"dist": "uniform", "low": 0.0, "high": 0.5, "choices": None},
    "weight_decay": {"dist": "log_uniform", "low": 1e-7, "high": 1e-2, "choices": None},
    "injection": {"dist": "categorical", "low": None, "high": None, "choices": ["D-state", "D-late"]},
    "desc_layers": {"dist": "categorical", "low": None, "high": None, "choices": [0, 1, 2]},
    "desc_width": {"dist": "categorical", "low": None, "high": None, "choices": [16, 32, 64]},
}
DRETRAIN_SMALL = (400, 200)                                # dretrain_v2 budgets on the v2 sweep grid; 654 on 30 x 5
MOMENT_SPEC = SPECS_DIR / "moment.jsonl"       # source of the excluded sites of moment_v2
V2_PRIORITY_OFFSET = 50_000                 # v2 jobs follow the v1 jobs of the same stage
PRIORITY_STRIDE = 100_000
PRECISION_BUDGETS = (400, 654)
PRECISION_NEW = (EXT_RESAMPLES, range(5))   # r = 10..29, seeds 0-4
PRECISION_REG = (range(10), (3, 4))         # r = 0..9, seeds 3 and 4
CV_SEEDS = (0, 1, 2)


def available_stages() -> tuple[str, ...]:
    """Stages whose inputs exist: every stage except those whose file in ``NEEDS_FILE`` is missing."""
    return tuple(s for s in STAGES if s not in NEEDS_FILE or NEEDS_FILE[s].is_file())


def spec_targets() -> list[tuple[str, str]]:
    """(stage, backbone) of every spec file the available stages define, in priority order."""
    out = []
    for s in available_stages():
        out += [(s, b) for b in BACKBONES if (b == "v1" and s not in V2_ONLY) or (b == "v2" and s in V2_STAGES)]
    return sorted(out, key=lambda t: (STAGES.index(t[0]), t[1]))


def spec_name(stage: str, backbone: str) -> str:
    """File name of a stage's spec: ``<stage>.jsonl``, ``<stage>_v2.jsonl`` for a v2 version of a v1 stage."""
    return f"{stage}{'_v2' if backbone == 'v2' and stage not in V2_ONLY else ''}.jsonl"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_hosts_file(path: Path) -> list[str]:
    """One host_id per line; blank lines and lines starting with '#' are ignored."""
    hosts = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"{path}: duplicate host ids")
    return sorted(hosts)


# ----------------------------------------------------------------------------------------- new stages
def build_cv(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None, subdir: str = "cv") -> list[dict]:
    """S and D x K in {10, 20} x folds x seeds 0-2, a654 hyperparameters, results/<subdir>/."""
    d = tuned["d_variant"]
    tags = {"v2": tuned["version"]} if tuned.get("arch") else {}
    cv = load_split(CV_NAME)
    jobs = []
    for k in CV_KS:
        for f in cv["by_k"][str(k)]["folds"]:
            for s in CV_SEEDS:
                for m in ("S", d):
                    spec = RunSpec(model=m, **tuned_hparams(tuned, m, 654), split=split_name(k, f["fold"]),
                                   r=f["fold"], budget=f["n_train"], seed=s, results_subdir=subdir,
                                   tags={"extension": "N3_cv", "k": k, **tags})
                    jobs.append(_job(spec, "cv", code, gsha, table=table))
    return jobs


def precision_cells() -> list[tuple[int, int, int]]:
    """(r, B, seed) of the precision extension: new resamples x seeds 0-4."""
    out = []
    for rs, seeds in (PRECISION_NEW, PRECISION_REG):
        out += [(r, b, s) for r in rs for b in PRECISION_BUDGETS for s in seeds]
    return out


def build_precision(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None,
                    subdir: str = "precision") -> list[dict]:
    """S and D at B = 400 and 654, results/<subdir>/."""
    d = tuned["d_variant"]
    tags = {"v2": tuned["version"]} if tuned.get("arch") else {}
    b2a = cfg.tuning.budget_to_anchor
    jobs = []
    for r, b, s in precision_cells():
        for m in ("S", d):
            spec = RunSpec(model=m, **tuned_hparams(tuned, m, b2a[b]), split=f"outer_r{r}", r=r, budget=b, seed=s,
                           results_subdir=subdir, tags={"extension": "N4_precision", **tags})
            jobs.append(_job(spec, "precision", code, gsha, table=table))
    return jobs


def moment_sites(path: Path = MOMENT_SPEC) -> list[str]:
    """The excluded site ids of the v1 moment spec (identical in all its jobs), read, not re-derived."""
    lists = {tuple(sorted(j["spec"]["exclude_sites"])) for j in load_spec_file(path)}
    if len(lists) != 1:
        raise ValueError(f"{path}: {len(lists)} different exclusion lists")
    return list(lists.pop())


def build_moment_v2(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """N2b, v2 only: the 60 moment jobs on the v2 backbone without the sites of specs/moment.jsonl,
    results/moment_v2/."""
    return S12.build_moment(code, gsha, tuned, cfg, table, sites=moment_sites(), subdir="moment_v2",
                            stage="moment_v2", tags={"v2": tuned["version"]})


def build_p1tune_v2(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """One P1 study per tuning anchor under the v2 backbone (``tuned["arch"]``), with the search space, sampler,
    trial seed, anchor resample, validation hosts and trial count of the S and D studies of
    configs/tuned_v2.yaml, and an epoch limit of ``P1_EPOCHS`` with the configured patience. Each job is a
    task (``dftgnn.tasks``) that runs or resumes the study and writes
    ``results/tuning/tuning_v2_P1_a<anchor>.json`` in the format of the S and D records, plus its task record
    in ``results/p1tune_v2/``."""
    t = cfg.tuning
    space = {k: v.model_dump(mode="json") for k, v in t.search_space.items()}
    est = None
    if table is not None:     # the largest trial of the search space
        est = max(admission.est_peak_for_spec(table, {"hidden_width": h, "megnet_blocks": m}, b)
                  for h in space["hidden_width"]["choices"] for m in space["megnet_blocks"]["choices"]
                  for b in space["batch_size"]["choices"])
    jobs = []
    for a in t.anchors:
        spec = {"model": "P1", "study": f"v2_P1_a{a}", "anchor": a, "split": f"outer_r{t.anchor_resample}",
                "r": t.anchor_resample, "budget": a, "seed": t.trial_seed,
                "n_trials": t.optuna_trials_per_model_per_anchor, "max_epochs": P1_EPOCHS,
                "patience": cfg.training.early_stopping.patience, "arch": dict(tuned["arch"]),
                "results_subdir": "tuning", "tags": {"v2": tuned["version"]}}
        key = {"stage": "p1tune_v2", **spec, "search_space": space, "sampler_seed": t.optuna.sampler_seed,
               "code_sha": code, "graphs_manifest_sha256": gsha}
        jobs.append(S12._task("p1tune_v2", key, spec, est=est))
    return jobs


def descriptor_classes(path: Path = TAXONOMY) -> dict[str, list[str]]:
    """{"host": the 12 host-level, "site": the 10 site-level descriptor columns} of the taxonomy (``desc_`` names,
    sorted, as in the graph store)."""
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {k: sorted(f"desc_{r['name']}" for r in rows if r["class"] == c)
           for k, c in (("host", "host-electronic DFT"), ("site", "site-electronic DFT"))}
    if (len(out["host"]), len(out["site"])) != (12, 10):
        raise ValueError(f"{path}: {len(out['host'])} host-level and {len(out['site'])} site-level descriptors")
    return out


def build_dclass(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """D-host and D-site, each D trained as the v2 sweep trains D (its tuned values at the budget's anchor, the
    same split, validation hosts and seed), with the other descriptor class at its training mean,
    results/dclass/. The kept names enter the tags, and the trainer checks them against the store."""
    d = tuned["d_variant"]
    b2a = cfg.tuning.budget_to_anchor
    kept = descriptor_classes()
    jobs = []
    for b in DCLASS_BUDGETS:
        for r in range(cfg.split.n_outer_resamples):
            for s in cfg.training.seeds:
                for variant, ablate in DCLASS.items():
                    names = kept["host" if variant == "D-host" else "site"]
                    spec = RunSpec(model=d, **tuned_hparams(tuned, d, b2a[b]), split=f"outer_r{r}", r=r, budget=b,
                                   seed=s, ablate=ablate, results_subdir="dclass",
                                   tags={"dclass": variant, "descriptors_kept": names, "v2": tuned["version"]})
                    jobs.append(_job(spec, "dclass", code, gsha, table=table))
    return jobs


def load_tuned_p1(path: Path | None = None) -> dict:
    """``configs/tuned_p1_v2.yaml``: {"version", "arch", "max_epochs", "patience", "models": {"P1": {anchor: ...}}}."""
    path = Path(path or TUNED_P1_V2)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: the stages p1v2b and pcompv2b need the P1 settings frozen "
                                "from the p1tune_v2 records (scripts/tune/freeze_p1_v2.py, Mac session C6)")
    return load_tuned(path)


def build_p1v2b(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """P1 under the v2 backbone with the p1tune_v2 settings of the budget's anchor and an epoch limit of
    ``P1_EPOCHS``, at every budget x resamples 0-9 x seeds 0-2 (the cells of the P1 sweep, results/p1_v2/),
    results/p1v2b/."""
    tp = load_tuned_p1(NEEDS_FILE["p1v2b"])
    if tp.get("arch") != tuned["arch"]:
        raise ValueError(f"{NEEDS_FILE['p1v2b']}: arch {tp.get('arch')} differs from {tuned['arch']} of {TUNED_V2}")
    pat = cfg.training.early_stopping.patience
    if (tp.get("max_epochs"), tp.get("patience")) != (P1_EPOCHS, pat):
        raise ValueError(f"{NEEDS_FILE['p1v2b']}: tuned with max_epochs {tp.get('max_epochs')}, patience {tp.get('patience')}; "
                         f"expected {P1_EPOCHS}, {pat}")
    b2a = cfg.tuning.budget_to_anchor
    jobs = []
    for b in sorted(cfg.budgets.hosts, reverse=True):
        for r in range(cfg.split.n_outer_resamples):
            for s in cfg.training.seeds:
                spec = RunSpec(model="P1", **tuned_hparams(tp, "P1", b2a[b]), split=f"outer_r{r}", r=r, budget=b,
                               seed=s, max_epochs=P1_EPOCHS, patience=pat, results_subdir="p1v2b",
                               tags={"v2": tuned["version"], "p1_tuned": tp["version"]})
                jobs.append(_job(spec, "p1v2b", code, gsha, table=table))
    return jobs


def v2_d_records(d: str) -> dict[tuple[str, int, int], dict]:
    """(split, budget, seed) -> payload of the v2 sweep run of model ``d`` (results/v2_sweep/), each checked against
    the D component of the P-v2 record of the same cell (results/p_v2/)."""
    out = {}
    for f in sorted(V2_SWEEP_DIR.glob("*.json")):
        pay = json.loads(f.read_text())["payload"]
        sp = pay["spec"]
        if sp["model"] == d:
            k = (sp["split"], sp["budget"], sp["seed"])
            if k in out:
                raise ValueError(f"two v2 sweep runs of {d} for {k}")
            out[k] = {**pay, "record": str(f.relative_to(REPO_ROOT))}
    for f in sorted(P_V2_DIR.glob("*.json")):
        sp = json.loads(f.read_text())["payload"]["spec"]
        k = (sp["split"], sp["budget"], sp["seed"])
        if k in out and out[k]["run_id"] != sp["d_run"]:
            raise ValueError(f"{f.name}: D component {sp['d_run']} is not the v2 sweep run {out[k]['run_id']}")
    return out


def build_pcompv2b(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """P = the p1v2b P1 then the existing v2 D of the same (split, budget, seed), on the test hosts, as the P-v2
    runs were built (the D run's hyperparameters in the spec), results/pcompv2b/. The D checkpoint is not in
    this checkout: the spec pins its sha256 (from its v2 sweep record) and the volume directories where it
    lies, and the evaluation loads it only if the hash matches. Each job waits for its P1 job."""
    d = tuned["d_variant"]
    recs = v2_d_records(d)
    jobs = []
    for p1 in build_p1v2b(code, gsha, tuned, cfg, table):
        p = p1["spec"]
        k = (p["split"], p["budget"], p["seed"])
        if k not in recs:
            raise ValueError(f"no v2 sweep run of {d} for {k}")
        rec = recs[k]
        ds, ck = rec["spec"], rec["checkpoint"]
        pin = {rec["run_id"]: {"sha256": ck["sha256"], "record": rec["record"], "dirs": list(V2_CKPT_DIRS)}}
        spec = RunSpec(model="P", hp=ds["hp"], lr=ds["lr"], weight_decay=ds["weight_decay"],
                       batch_size=ds["batch_size"], split=p["split"], r=p["r"], budget=p["budget"], seed=p["seed"],
                       results_subdir="pcompv2b", p1_run=p1["run_id"], d_run=rec["run_id"],
                       tags={"v2": tuned["version"], "checkpoints": pin})
        j = _job(spec, "pcompv2b", code, gsha, after=[p1["run_id"]])
        if table is not None:
            j["est_peak_gb"] = round(S12._est_infer(table, [p["hp"], ds["hp"]]), 3)
        jobs.append(j)
    return jobs


def dtune_space(cfg: Config) -> dict:
    """The dtune_v2 search space (JSON form): the configured space with ``DTUNE_SPACE`` applied. Raises unless it
    contains every configured parameter with its full range or all its choices."""
    old = {k: v.model_dump(mode="json") for k, v in cfg.tuning.search_space.items()}
    new = {**old, **{k: dict(v) for k, v in DTUNE_SPACE.items()}}
    for k, o in old.items():
        n = new[k]
        if n["dist"] != o["dist"]:
            raise ValueError(f"dtune_v2 changes the distribution of {k}")
        if o["choices"] is not None and not set(o["choices"]) <= set(n["choices"]):
            raise ValueError(f"dtune_v2 drops choices of {k}")
        if o["choices"] is None and not n["low"] <= o["low"] < o["high"] <= n["high"]:
            raise ValueError(f"dtune_v2 narrows the range of {k}")
    return new


def build_dtune_v2(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """One D study per tuning anchor under the v2 backbone, as p1tune_v2 runs P1 (sampler, trial seed, anchor
    resample, validation hosts, 30 trials, inner validation MAE of E_f as objective), in the space
    ``dtune_space`` with an epoch limit of ``D_EPOCHS`` and the configured patience. Each trial trains the
    sampled injection variant. Each job writes ``results/tuning/tuning_v2_D_a<anchor>.json`` and its task
    record in ``results/dtune_v2/``."""
    t = cfg.tuning
    space = dtune_space(cfg)
    est = None
    if table is not None:     # the largest trial of the search space
        est = max(admission.est_peak_for_spec(table, {"hidden_width": h, "megnet_blocks": m}, b)
                  for h in space["hidden_width"]["choices"] for m in space["megnet_blocks"]["choices"]
                  for b in space["batch_size"]["choices"])
    jobs = []
    for a in t.anchors:
        spec = {"model": "D", "study": f"v2_D_a{a}", "anchor": a, "split": f"outer_r{t.anchor_resample}",
                "r": t.anchor_resample, "budget": a, "seed": t.trial_seed,
                "n_trials": t.optuna_trials_per_model_per_anchor, "max_epochs": D_EPOCHS,
                "patience": cfg.training.early_stopping.patience, "arch": dict(tuned["arch"]),
                "search_space": space, "results_subdir": "tuning", "tags": {"v2": tuned["version"]}}
        key = {"stage": "dtune_v2", **spec, "sampler_seed": t.optuna.sampler_seed, "code_sha": code,
               "graphs_manifest_sha256": gsha}
        jobs.append(S12._task("dtune_v2", key, spec, est=est))
    return jobs


def load_tuned_d(path: Path | None = None) -> dict:
    """``configs/tuned_d_v2.yaml``: {"version", "arch", "max_epochs", "patience", "models": {"D": {anchor: ...}}}, each
    anchor's parameters including ``injection`` (the D variant)."""
    path = Path(path or TUNED_D_V2)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: the stage dretrain_v2 needs the D settings frozen from the "
                                "dtune_v2 records (scripts/tune/freeze_d_v2.py, Mac session C8)")
    return load_tuned(path)


def tuned_d_hparams(td: dict, anchor: int) -> tuple[str, dict]:
    """(D variant, RunSpec hyperparameters) of an anchor of ``configs/tuned_d_v2.yaml``, with its ``arch``."""
    p = dict(td["models"]["D"][anchor])
    kind = p.pop("injection")
    if kind not in ("D-state", "D-late"):
        raise ValueError(f"tuned D a{anchor}: unknown injection {kind!r}")
    return kind, tuned_hparams({"models": {"D": {anchor: p}}, "arch": td.get("arch")}, "D", anchor)


def dretrain_cells(cfg: Config) -> list[tuple[int, int, int]]:
    """(r, B, seed) of dretrain_v2, the cells of the existing v2 D runs: B = 654 on the 30 x 5 headline grid (v2 sweep
    resamples 0-9 x seeds 0-2, then the precision cells), B = 400 and 200 on the v2 sweep grid."""
    sweep = [(r, s) for r in range(cfg.split.n_outer_resamples) for s in cfg.training.seeds]
    out = [(r, 654, s) for r, s in sweep] + [c for c in precision_cells() if c[1] == 654]
    for b in DRETRAIN_SMALL:
        out += [(r, b, s) for r, s in sweep]
    return out


def build_dretrain_v2(code: str, gsha: str, tuned: dict, cfg: Config, table: dict | None) -> list[dict]:
    """D under the v2 backbone with the dtune_v2 settings of the budget's anchor (its variant included) and an
    epoch limit of ``D_EPOCHS``, on ``dretrain_cells``: the split, validation hosts, test hosts and seed of
    the existing v2 D run (and S run) of each cell, results/dretrain_v2/."""
    td = load_tuned_d(NEEDS_FILE["dretrain_v2"])
    if td.get("arch") != tuned["arch"]:
        raise ValueError(f"{NEEDS_FILE['dretrain_v2']}: arch {td.get('arch')} differs from {tuned['arch']} of {TUNED_V2}")
    pat = cfg.training.early_stopping.patience
    if (td.get("max_epochs"), td.get("patience")) != (D_EPOCHS, pat):
        raise ValueError(f"{NEEDS_FILE['dretrain_v2']}: tuned with max_epochs {td.get('max_epochs')}, patience "
                         f"{td.get('patience')}; expected {D_EPOCHS}, {pat}")
    b2a = cfg.tuning.budget_to_anchor
    jobs = []
    for r, b, s in dretrain_cells(cfg):
        kind, hp = tuned_d_hparams(td, b2a[b])
        spec = RunSpec(model=kind, **hp, split=f"outer_r{r}", r=r, budget=b, seed=s, max_epochs=D_EPOCHS, patience=pat,
                       results_subdir="dretrain_v2", tags={"v2": tuned["version"], "d_tuned": td["version"]})
        jobs.append(_job(spec, "dretrain_v2", code, gsha, table=table))
    return jobs


# ------------------------------------------------------------------------------------------- generator
def _benchmark() -> dict | None:
    from dftgnn.train import admission

    try:
        return admission.load_benchmark()
    except (OSError, KeyError):
        return None


def build_stage(stage: str, *, code_ref: str = CODE_REF, backbone: str = "v1", hosts: list[str] | None = None,
                tuned_path: Path | None = None, cfg: Config | None = None) -> list[dict]:
    """Jobs of one stage, with ``code_ref``, ``backbone``, ``group``, ``label`` and a global ``priority``.

    ``hosts`` (relax and geomeval): the relax top-up list. relax builds only these hosts; geomeval drops the
    dependencies on hosts outside it (their relaxed geometry comes from the earlier, ingested runs).
    """
    from dftgnn.graphs.store import manifest_sha

    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}")
    if backbone == "v2" and stage not in V2_STAGES:
        raise ValueError(f"stage {stage!r} has a v1 version only")
    if backbone == "v1" and stage in V2_ONLY:
        raise ValueError(f"stage {stage!r} has a v2 version only")
    if stage in NEEDS_FILE and not NEEDS_FILE[stage].is_file():   # raises with the message naming the missing file
        (load_tuned_d if stage == "dretrain_v2" else load_tuned_p1)(NEEDS_FILE[stage])
    tuned_path = tuned_path if tuned_path is not None else (TUNED_V2 if backbone == "v2" else TUNED)
    cfg = cfg if cfg is not None else load_config()
    tuned, gsha, table, code = load_tuned(tuned_path), manifest_sha(), _benchmark(), code_ref
    if stage in ("peval", "embed", "geomeval"):
        prim = _primary()
    if stage == "relax":
        jobs = S12.build_relax(code)
        if hosts is not None:
            unknown = set(hosts) - {j["spec"]["host_id"] for j in jobs}
            if unknown:
                raise ValueError(f"hosts not in universe v1: {sorted(unknown)[:5]}")
            jobs = [j for j in jobs if j["spec"]["host_id"] in set(hosts)]
    elif stage == "peval":
        jobs = S12.build_peval(code, gsha, tuned, prim, cfg, table)
    elif stage == "embed":
        jobs = S12.build_embed(code, prim, cfg, table)
    elif stage == "geomeval":
        relax = S12.build_relax(code)
        keep = None if hosts is None else {j["run_id"] for j in relax if j["spec"]["host_id"] in set(hosts)}
        jobs = S12.build_geomeval(code, prim, relax, cfg, table)
        if keep is not None:
            jobs = [{**j, "after": [a for a in j["after"] if a in keep]} for j in jobs]
    elif stage == "moment":
        jobs = S12.build_moment(code, gsha, tuned, cfg, table)
    elif stage == "moment_v2":
        jobs = build_moment_v2(code, gsha, tuned, cfg, table)
    elif stage == "cgcnn":
        jobs = S12.build_cgcnn(code, gsha, cfg, table)
    elif stage == "capped":
        jobs = S12.build_capped(code, gsha, table)
    elif stage in ("p1tune_v2", "dclass", "p1v2b", "pcompv2b", "dtune_v2", "dretrain_v2"):
        jobs = globals()[f"build_{stage}"](code, gsha, tuned, cfg, table)
    elif stage == "cv":
        jobs = build_cv(code, gsha, tuned, cfg, table, subdir="cv" if backbone == "v1" else "cv_v2")
    else:
        jobs = build_precision(code, gsha, tuned, cfg, table, subdir="precision" if backbone == "v1" else "precision_v2")
    if bool(tuned.get("arch")) != (backbone == "v2"):
        raise ValueError(f"{tuned_path} does not match backbone {backbone}")
    base = STAGES.index(stage) * PRIORITY_STRIDE + (V2_PRIORITY_OFFSET if backbone == "v2" else 0)
    label = V2_LABEL if backbone == "v2" else LABEL[stage]
    out = []
    for i, j in enumerate(jobs):
        spec = dict(j["spec"])
        if "lr" in spec:                                   # RunSpec jobs: the runner reads code_ref from the spec
            spec["code_ref"] = code_ref
            if run_id(RunSpec.from_dict(spec), code_ref, gsha) != j["run_id"]:
                raise RuntimeError(f"run id of {j['run_id']} does not reproduce from its spec")
        out.append({**j, "spec": spec, "stage": stage, "code_ref": code_ref, "backbone": backbone,
                    "group": GROUP[stage], "tranche": GROUP[stage], "label": label, "priority": base + i})
    return out


def expected_count(stage: str, hosts: list[str] | None) -> int:
    return len(hosts) if stage == "relax" and hosts is not None else EXPECTED[stage]


def dumps_jsonl(jobs: list[dict]) -> bytes:
    return "".join(json.dumps(j, sort_keys=True, separators=(",", ":")) + "\n" for j in jobs).encode()


def manifest(stage: str, jobs: list[dict], body: bytes, *, jsonl_name: str, code_ref: str, backbone: str,
             hosts_file: Path | None, tuned_path: Path) -> dict:
    """Counts and hashes of a spec file; contains no time stamp, so it is reproducible."""
    from dftgnn.graphs.store import manifest_sha

    def rel(p: Path) -> str:
        p = Path(p).resolve()
        return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)

    return {"stage": stage, "group": GROUP[stage], "label": V2_LABEL if backbone == "v2" else LABEL[stage],
            "code_ref": code_ref, "backbone": backbone,
            "backbones_with_this_stage": {b: spec_name(stage, b) for b in BACKBONES
                                          if (b == "v1" and stage not in V2_ONLY) or (b == "v2" and stage in V2_STAGES)},
            "jsonl": jsonl_name, "sha256": hashlib.sha256(body).hexdigest(),
            "count": len(jobs), "counts_by_model": dict(sorted(Counter(j["spec"]["model"] for j in jobs).items())),
            "priority_range": [min(j["priority"] for j in jobs), max(j["priority"] for j in jobs)] if jobs else None,
            "graphs_manifest_sha256": manifest_sha(),
            "splits_manifest_sha256": sha256_file(SPLITS_DIR / "MANIFEST.sha256"),
            "tuned": {"path": rel(tuned_path), "sha256": sha256_file(tuned_path)},
            "hosts_file": None if hosts_file is None else {"path": rel(hosts_file), "sha256": sha256_file(hosts_file)},
            "generator": "scripts/queue/make_specs.py",
            **({"tuned_p1": {"path": rel(NEEDS_FILE[stage]), "sha256": sha256_file(NEEDS_FILE[stage])}}
               if stage in NEEDS_FILE and stage != "dretrain_v2" else {}),
            **({"tuned_d": {"path": rel(NEEDS_FILE[stage]), "sha256": sha256_file(NEEDS_FILE[stage])}}
               if stage == "dretrain_v2" else {}),
            **({"descriptor_taxonomy": {"path": rel(TAXONOMY), "sha256": sha256_file(TAXONOMY)}}
               if stage == "dclass" else {})}


# --------------------------------------------------------------------------------------------- checks
@functools.lru_cache(maxsize=1)
def _primary() -> dict:
    from dftgnn.train import runindex

    return runindex.primary(runindex.load())


def leakage_hosts(job: dict, cfg: Config | None = None) -> list[tuple[set, set]]:
    """(test hosts, train + validation hosts) of every model a job trains or evaluates; [] for relax."""
    cfg = cfg if cfg is not None else load_config()
    st, sp = job["stage"], job["spec"]
    if st == "relax":
        return []
    if st in ("p1tune_v2", "dtune_v2"):      # the study's anchor training and validation hosts
        h = resolve_hosts(RunSpec(model="P1", hp={}, lr=0.0, weight_decay=0.0, batch_size=1, split=sp["split"],
                                  r=sp["r"], budget=sp["budget"], seed=sp["seed"]), cfg)
        return [(set(h["test"]), set(h["train"]) | set(h["val"]))]
    if "lr" in sp:
        h = resolve_hosts(RunSpec.from_dict(sp), cfg)
        return [(set(h["test"]), set(h["train"]) | set(h["val"]))]
    if st == "embed":
        s = load_split(f"outer_r{sp['r']}")
        return [(set(s["test"]), set(budget_train(s, sp["budget"])))]
    if st == "geomeval":
        from dftgnn.mlip import geomeval as GE

        out = []
        for rec in GE.components(_primary(), sp["geo_model"], sp["r"], sp["budget"], sp["seed"]).values():
            h = resolve_hosts(RunSpec.from_dict(rec["spec"]), cfg)
            out.append((set(h["test"]), set(h["train"]) | set(h["val"])))
        return out
    raise ValueError(f"no leakage rule for stage {st!r}")


def check_no_leakage(job: dict, cfg: Config | None = None) -> None:
    for test, seen in leakage_hosts(job, cfg):
        if not test or test & seen:
            raise AssertionError(f"{job['run_id']}: test hosts overlap training/validation ({len(test & seen)})")


# ------------------------------------------------------------------------------------------ enqueueing
def load_spec_file(path: Path) -> list[dict]:
    """Jobs of ``specs/<stage>.jsonl`` after checking it against ``<stage>.manifest.json`` (sha256 and count)."""
    path = Path(path)
    man = json.loads(path.with_name(path.name.removesuffix(".jsonl") + ".manifest.json").read_text())
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != man["sha256"]:
        raise ValueError(f"{path}: sha256 differs from its manifest")
    jobs = [json.loads(ln) for ln in body.decode().splitlines() if ln]
    if len(jobs) != man["count"]:
        raise ValueError(f"{path}: {len(jobs)} jobs, manifest says {man['count']}")
    return jobs


def shard(jobs: list[dict], spec: str | None) -> list[dict]:
    """Jobs of shard ``"I/N"``: those with int(run_id, 16) % N == I (stable, independent of file order)."""
    if not spec:
        return jobs
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {spec!r}")
    return [j for j in jobs if int(j["run_id"], 16) % n == i]


def p1v2b_result_verified(rid: str) -> bool:
    """An ingested p1v2b record is a met dependency (pcompv2b) when ``results/p1v2b/<rid>.json`` names ``rid``, was
    written from a clean tree by an accepted code commit (``accepted_code_commits`` of its code reference), and its
    checkpoint sha256 equals the entry for that checkpoint in ``results/nrp/artefacts/p1v2b.sha256`` (the checkpoint
    itself stays on NRP storage). Anything missing or different is refused."""
    try:
        rec = json.loads((P1V2B_DIR / f"{rid}.json").read_text())
        pay = rec["payload"]
        ck = pay["checkpoint"]
        if rec["name"] != rid or pay["run_id"] != rid or rec["git_dirty"] is not False:
            return False
        if rec["git_sha"] not in accepted_code_commits(pay["code_ref"]):
            return False
        listed = {}
        for line in P1V2B_CHECKPOINTS.read_text().splitlines():
            if line.strip():
                h, _size, path = line.split("  ", 2)
                listed[path] = h
        return ck["path"] in listed and listed[ck["path"]] == ck["sha256"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def task_result_verified(rid: str) -> bool:
    from dftgnn import tasks

    return any(tasks.verified(tasks.result_path(st, rid), rid) for st in tasks.SUBDIR) or p1v2b_result_verified(rid)


def resolve_deps(jobs: list[dict], queue_root: Path, verified=task_result_verified) -> list[dict]:
    """Keep a dependency that is in this batch or already in the queue (any state); drop one whose result is
    already present and verifies (e.g. relax results ingested from an earlier run or another shard); refuse
    anything else, so a stage is never enqueued ahead of what it needs."""
    ids = {j["run_id"] for j in jobs}
    known = {f.stem for s in ("pending", "running", "done", "failed") for f in (Path(queue_root) / s).glob("*.json")}
    out = []
    for j in jobs:
        keep = []
        for a in j.get("after", []):
            if a in ids or a in known:
                keep.append(a)
            elif not verified(a):
                raise ValueError(f"{j['run_id']} ({j['stage']}) depends on {a}: not in this batch, not in the queue "
                                 "and no verified result; enqueue that stage first")
        out.append({**j, "after": keep})
    return out
