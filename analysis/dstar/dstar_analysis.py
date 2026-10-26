"""The re-tuned descriptor network D* (stage dretrain_v2) against S and the previous D, with the G1 estimator unchanged.

Nothing is trained. Physically: A = MAE_S - MAE_D measures how much the pristine-cell electronic descriptors (band
gap, oxide formation energy, dielectric response, band-edge character, Bader charge) lower the error of the neutral
oxygen-vacancy formation energy beyond what the crystal structure gives. If the descriptor network D was trained
with settings that held it back, A understates the information in the descriptors. D was therefore re-tuned (stage
dtune_v2: 30 trials at each of 50, 200 and 654 hosts, epoch limit 400, a search space that contains the previous one
and both descriptor-injection routes; configs/tuned_d_v2.yaml) and retrained with the best trial of each anchor
(stage dretrain_v2, 210 runs): D-late at 654 hosts on 30 resamples x 5 seeds, D-late at 400 hosts (654-anchor
settings) and D-state at 200 hosts on resamples 0-9 x seeds 0-2. That network is called D* here ("Dstar").

Estimator (imported unchanged): seed-ensemble mean per (model, r, B, site); per-resample host statistics; the G1
hierarchical bootstrap ``HierDraws`` (2000 replicates, seed 20261008), one draw set per design shared by every model
of the design, so every contrast is paired; point = mean over resamples; ``contrast`` for paired gaps; ``a_entry``
(analysis/c1/final_network.py) for the two-sided 95% interval, the one-sided 95% upper bound and the checks against
delta = 0.05 eV. Two designs:

    headline   654 hosts, 30 resamples x 5 seeds; S and D from results/analysis/precision_eval_long.parquet (v2);
               draws HierDraws.make([164] * 30, 2000, 20261008) as in analysis/precision_analysis.py
    sweep      200, 400, 654 hosts, 10 resamples x 3 seeds; S and D from results/analysis/staging_eval_long.parquet
               (v2); draws of analysis/staging_analysis.make_draws; at 654 hosts D* is the r = 0-9, seeds 0-2
               subset of its 150 runs; the descriptor classes D-host, D-site (results/dclass) and the staged network
               P (results/pcompv2b) are on this design only

Signs: A_old = MAE_S - MAE_D, A* = MAE_S - MAE_D*, D - D* = MAE_D - MAE_D* (positive = D* more accurate),
D* - D-host and D* - D-site (positive = the single class more accurate), P - D* (positive = P less accurate). A gap
is "the same within 0.01 eV" when its two-sided interval lies strictly inside (-0.01, +0.01) eV. P composes the
retrained P1 with the previous D checkpoints, so P - D* compares two different energy heads.

Context, default-initialisation network: results/dlate holds 90 D-late runs (200, 400, 654 hosts, r = 0-9, seeds
0-2) trained with the D-late values of configs/tuned_v1.yaml and no initialisation field (default initialisation),
epoch limit 200, i.e. the v1 network. Their paired gap to the v1 D-state runs of the sweep shows what the injection
route alone did there. No D-late run of the He-initialised network exists outside dretrain_v2 (scanned here).

Checks before a record is written (tolerance 1e-9 eV unless stated): dretrain_v2 run ids equal specs/dretrain_v2.jsonl
and each recorded spec equals its line; git_dirty false; code ref nrp-code-v1 with an accepted code commit; variant
D-late at 654 and 400 hosts, D-state at 200, with the settings of configs/tuned_d_v2.yaml at the budget's anchor;
prediction sha256; finite predictions; recorded MAE equals the stored predictions (1e-6); test sites, targets and
hosts equal those of the S run of the cell; no duplicate cell; complete grids; no run at the epoch limit. S, D and A
reproduce results/derived_c1/final_network_summary.json (rows B654_30x5, B200, B400, B654) and
results/precision_a_b400_b654.json (v2, including the within-host and host-mean metrics); S, D, D-host, D-site
reproduce results/derived_dclass/dclass_gaps.json; S, D, P reproduce results/derived_staging/staging_fair.json;
the v1 S and D reproduce results/derived_c1/default_init_ablation.json.

Writes through ``write_result`` into ``results/derived_dstar/dstar_gaps.json``.

    PYTHONPATH=$PWD/src python analysis/dstar/dstar_analysis.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import a1_common as c
from analysis import precision_analysis as pa
from analysis import staging_analysis as sa
from analysis.c1.final_network import a_entry, compare, ms
from analysis.dclass import dclass_analysis as dca
from analysis.m3 import staging_fair as sf
from dftgnn.stats.g1 import HierDraws, contrast, n_star, seed_ensemble

OUT_DIR = ROOT / "results" / "derived_dstar"
STAGE_DIR = ROOT / "results" / "dretrain_v2"
DLATE_DIR = ROOT / "results" / "dlate"
SPEC = ROOT / "specs" / "dretrain_v2.jsonl"
DELTA = c.DELTA_EV
SAME_EV = 0.01
HEADLINE_B = 654
SWEEP_B = (200, 400, 654)
VARIANT = {654: "D-late", 400: "D-late", 200: "D-state"}
GRID = {654: (30, 5), 400: (10, 3), 200: (10, 3)}
METRICS = pa.METRICS
COLS = ["model", "r", "B", "seed", "site_id", "host_id", "y_true", "y_pred"]
CORE_GAPS = (("S_minus_D", "S", "D"), ("S_minus_Dstar", "S", "Dstar"), ("D_minus_Dstar", "D", "Dstar"))
CLASS_GAPS = (("S_minus_Dhost", "S", "D-host"), ("S_minus_Dsite", "S", "D-site"),
              ("D_minus_Dhost", "D", "D-host"), ("D_minus_Dsite", "D", "D-site"),
              ("Dstar_minus_Dhost", "Dstar", "D-host"), ("Dstar_minus_Dsite", "Dstar", "D-site"))
LABEL = ("re-tuned descriptor network D*, dretrain_v2, spec sha256 "
         "c33cfa33aac3ed56c46c9802abbf70f63e7b030386043f233e5761c9b9b60f65; compared with the structure-only "
         "network S and the previous descriptor network D of the final (He-initialised) network")
SIGNS = {"A_old": "S_minus_D = MAE_S - MAE_D", "A_star": "S_minus_Dstar = MAE_S - MAE_D*",
         "D_minus_Dstar": "MAE_D - MAE_D* (positive = D* more accurate)",
         "Dstar_minus_class": "MAE_D* - MAE_class (positive = the single descriptor class more accurate)",
         "S_minus_class": "MAE_S - MAE_class (positive = the class more accurate than structure alone)",
         "P_minus_Dstar": "MAE_P - MAE_D* (positive = P less accurate)",
         "Dlate_minus_Dstate_v1": "MAE_D-late - MAE_D-state, default-initialisation network (positive = D-state "
                                  "more accurate)"}


# ---------------------------------------------------------------- D* records

def s_reference() -> tuple[dict, dict]:
    """(B, r) -> the seed-0 S run of the cell indexed by site_id: 654 hosts from the precision table (r = 0-29),
    200 and 400 hosts from the sweep table (r = 0-9); the two tables agree on the shared cells (checked)."""
    prec, f_prec = c.load_table("precision_eval_long")
    sweep, f_sweep = sa.load_table("staging_eval_long", c.record("staging_eval_table"))
    out = {}
    p = prec[(prec.backbone == "v2") & (prec.model == "S") & (prec.seed == 0) & (prec.B == HEADLINE_B)]
    for r, g in p.groupby("r"):
        out[(HEADLINE_B, int(r))] = g.set_index("site_id").sort_index()
    w = sweep[(sweep.backbone == "v2") & (sweep.model == "S") & (sweep.seed == 0) & sweep.B.isin(SWEEP_B)]
    for (B, r), g in w.groupby(["B", "r"]):
        g = g.set_index("site_id").sort_index()
        if (int(B), int(r)) in out:
            o = out[(int(B), int(r))]
            if not (o.index.equals(g.index) and np.array_equal(o.y_true.to_numpy(), g.y_true.to_numpy())
                    and (o.host_id == g.host_id).all()):
                raise AssertionError(f"S test sites at B = {B}, r = {r} differ between the precision and sweep tables")
        else:
            out[(int(B), int(r))] = g
    return out, {"precision_eval_long": f_prec, "staging_eval_long": f_sweep}


def tuned_matches(s: dict, tuned: dict, b2a: dict) -> bool:
    t = tuned["models"]["D"][b2a[int(s["budget"])]]
    hp = s["hp"]
    return bool(s["model"] == t["injection"] and s["lr"] == t["learning_rate"] and s["weight_decay"] == t["weight_decay"]
                and s["batch_size"] == t["batch_size"] and hp["init"] == tuned["arch"]["init"]
                and all(hp[k] == t[k] for k in ("hidden_width", "megnet_blocks", "dropout", "readout_mlp_width",
                                                 "pooling", "desc_layers", "desc_width"))
                and s["max_epochs"] == tuned["max_epochs"] and s["patience"] == tuned["patience"])


def load_dstar(ref: dict) -> tuple[pd.DataFrame, dict]:
    from dftgnn.config import load_config

    spec = {}
    for line in SPEC.read_text().splitlines():
        row = json.loads(line)
        spec[row["run_id"]] = row
    files = sorted(STAGE_DIR.glob("*.json"))
    if len(files) != 210 or {p.stem for p in files} != set(spec):
        raise AssertionError("dretrain_v2 records differ from specs/dretrain_v2.jsonl")
    tuned = yaml.safe_load((ROOT / "configs" / "tuned_d_v2.yaml").read_text())
    b2a = {int(k): int(v) for k, v in load_config().tuning.budget_to_anchor.items()}
    frames, cells, worst, capped, runs = [], set(), 0.0, [], []
    for p in files:
        rec = json.loads(p.read_text())
        pl = sf.check_record(rec, p.stem, spec[p.stem])
        s = pl["spec"]
        if pl["code_ref"] != "nrp-code-v1" or s["results_subdir"] != "dretrain_v2":
            raise AssertionError(f"{p.stem}: code ref or results directory")
        B, r, seed = int(s["budget"]), int(s["r"]), int(s["seed"])
        if B not in VARIANT or s["model"] != VARIANT[B]:
            raise AssertionError(f"{p.stem}: variant {s['model']} at {B} hosts")
        if not tuned_matches(s, tuned, b2a):
            raise AssertionError(f"{p.stem}: settings differ from configs/tuned_d_v2.yaml at anchor {b2a[B]}")
        n_r, n_s = GRID[B]
        if not (0 <= r < n_r and 0 <= seed < n_s):
            raise AssertionError(f"{p.stem}: cell ({B}, {r}, {seed}) outside the grid")
        if (B, r, seed) in cells:
            raise AssertionError(f"{p.stem}: duplicate cell")
        cells.add((B, r, seed))
        path = ROOT / pl["predictions"]["path"]
        if c.sha256(path) != pl["predictions"]["sha256"]:
            raise AssertionError(f"{p.stem}: prediction sha256")
        d = pd.read_parquet(path)
        if not np.isfinite(d[["y_true", "y_pred"]].to_numpy()).all():
            raise AssertionError(f"{p.stem}: non-finite predictions")
        worst = max(worst, abs(float(np.abs(d.y_true - d.y_pred).mean()) - pl["metrics"]["mae"]))
        base = ref[(B, r)]
        if len(d) != len(base) or set(d.site_id) != set(base.index) or d.site_id.duplicated().any():
            raise AssertionError(f"{p.stem}: test sites differ from the S run of the cell")
        j = d.set_index("site_id").loc[base.index]
        if not (np.array_equal(j.y_true.to_numpy(), base.y_true.to_numpy()) and (j.host_id == base.host_id).all()):
            raise AssertionError(f"{p.stem}: targets or hosts differ from the S run of the cell")
        if pl["epochs_run"] >= pl["max_epochs"]:
            capped.append(p.stem)
        runs.append({"B": B, "model": s["model"], "epochs_run": pl["epochs_run"], "best_epoch": pl["best_epoch"],
                     "max_epochs": pl["max_epochs"], "gpu": pl["hardware"]["gpu_model"], "git_sha": rec["git_sha"],
                     "wall_time_s": pl["wall_time_s"]})
        frames.append(d.assign(model="Dstar", r=r, B=B, seed=seed)[COLS])
    if worst > 1e-6:
        raise AssertionError(f"recorded MAE differs from the stored predictions by {worst}")
    if len(cells) != sum(a * b for a, b in GRID.values()):
        raise AssertionError("dretrain_v2 grid incomplete")
    if capped:
        raise AssertionError(f"dretrain_v2 runs at the epoch limit: {capped}")
    return pd.concat(frames, ignore_index=True), {"runs": pd.DataFrame(runs), "worst": worst,
                                                 "spec_sha256": c.sha256(SPEC)}


def health(info: dict) -> dict:
    df = info["runs"]
    out = {"n_runs": len(df), "n_runs_at_epoch_limit": int((df.epochs_run >= df.max_epochs).sum()),
           "max_abs_diff_recorded_mae_vs_predictions": info["worst"], "spec_sha256": info["spec_sha256"],
           "code_commits": sorted(df.git_sha.unique()), "per_budget": {}}
    summ = lambda x: {"min": int(x.min()), "median": float(x.median()), "mean": float(x.mean()), "max": int(x.max())}
    for B in sorted(df.B.unique()):
        g = df[df.B == B]
        out["per_budget"][str(B)] = {
            "variant": VARIANT[int(B)], "n_runs": len(g), "max_epochs": int(g.max_epochs.max()),
            "n_at_epoch_limit": int((g.epochs_run >= g.max_epochs).sum()), "epochs_run": summ(g.epochs_run),
            "best_epoch": summ(g.best_epoch), "gpu_models": dict(Counter(g.gpu)),
            "mean_wall_time_h": float(g.wall_time_s.mean() / 3600)}
    return out


# ---------------------------------------------------------------- estimator pieces

def gap_entry(diff: dict) -> dict:
    """a_entry plus the 0.01 eV sameness check and the per-resample spread of the gap."""
    e = a_entry(diff)
    per = np.asarray(diff["per_resample"], float)
    lo, hi = e["ci95"]
    e.update({"inside_pm_0p01": bool(lo > -SAME_EV and hi < SAME_EV), "abs_point_le_0p01": bool(abs(e["A"]) <= SAME_EV),
              "boot_sd": float(diff["boot_sd"]), "sd_over_resamples": float(diff["sd_over_resamples"]),
              "n_resamples": int(per.size), "n_resamples_above_delta": int((per > DELTA).sum()),
              "n_resamples_below_zero": int((per < 0).sum()), "min_resample": float(per.min()),
              "max_resample": float(per.max()), "per_resample": per.tolist()})
    return e


def relation(g: dict) -> str:
    """Words for MAE_D* - MAE_class: the sign of the paired gap where its interval excludes zero."""
    lo, hi = g["ci95"]
    if lo > 0:
        return "class more accurate than D*"
    if hi < 0:
        return "D* more accurate than class"
    return "no measurable difference"


def design_table(st: dict, draws, budgets, gaps, metrics=METRICS, gap_metrics=None) -> tuple[dict, dict]:
    """Per budget: for each metric the marginal of every model and the paired gaps; 'MAE' and 'gaps' hold the
    site MAE (the layout of analysis/dclass/dclass_analysis.table, so its reproduce applies unchanged)."""
    gap_metrics = gap_metrics or {}
    raw = {m: {(a, b): contrast({B: st[a][B] for B in budgets}, {B: st[b][B] for B in budgets}, draws, m,
                                ci=c.CI, upper_quantile=c.UPPER_Q)
               for k, a, b in gaps if m in gap_metrics.get(k, metrics)} for m in metrics}
    out = {}
    for B in budgets:
        row = {"x_training_hosts": B, "metrics": {}}
        for m in metrics:
            marg = {}
            for (a, b), ct in raw[m].items():
                marg.setdefault(a, ms(ct[B]["a"]))
                marg.setdefault(b, ms(ct[B]["b"]))
            row["metrics"][m] = {"MAE": marg, "gaps": {k: gap_entry(raw[m][(a, b)][B]["diff"])
                                                       for k, a, b in gaps if (a, b) in raw[m]}}
        row["MAE"], row["gaps"] = row["metrics"]["mae"]["MAE"], row["metrics"]["mae"]["gaps"]
        out[str(B)] = row
    return out, raw


def precision_view(raw: dict, B: int) -> dict:
    """The S - D contrast in the layout of results/precision_a_b400_b654.json (for the reproduction)."""
    sd = {m: raw[m][("S", "D")][B] for m in METRICS}
    return {"MAE_S": sd["mae"]["a"], "MAE_D": sd["mae"]["b"], "A": sd["mae"]["diff"],
            "within_host_S": sd["within_host_mae"]["a"], "within_host_D": sd["within_host_mae"]["b"],
            "A_within": sd["within_host_mae"]["diff"], "host_mean_S": sd["host_mean_mae"]["a"],
            "host_mean_D": sd["host_mean_mae"]["b"], "A_host_mean": sd["host_mean_mae"]["diff"]}


def c1_view(row: dict) -> dict:
    g = row["gaps"]["S_minus_D"]
    return {"MAE_S": row["MAE"]["S"], "MAE_D": row["MAE"]["D"],
            "A": {k: g[k] for k in ("A", "ci95", "upper_one_sided_95", "two_sided_within_pm_delta",
                                     "upper_below_delta", "upper_minus_delta", "ci95_excludes_zero")}}


# ---------------------------------------------------------------- the two designs

def headline(dstar: pd.DataFrame) -> tuple[dict, dict]:
    prec, _ = c.load_table("precision_eval_long")
    n_r, n_s = GRID[HEADLINE_B]
    base = prec[(prec.backbone == "v2") & (prec.B == HEADLINE_B) & (prec.r < n_r) & (prec.seed < n_s)][COLS]
    long = pd.concat([base, dstar[dstar.B == HEADLINE_B]], ignore_index=True)
    ens = seed_ensemble(long)
    if (ens.n_seeds != n_s).any():
        raise AssertionError("headline: seed ensemble over the wrong number of seeds")
    draws = HierDraws.make([pa.N_HOSTS_PER_RESAMPLE] * n_r, c.N_BOOT, seed=c.BOOT_SEED)
    st = {m: pa.resample_stats_by_budget(ens, m, n_r, (HEADLINE_B,)) for m in ("S", "D", "Dstar")}
    tab, raw = design_table(st, draws, (HEADLINE_B,), CORE_GAPS)
    row = tab[str(HEADLINE_B)]
    fin = c.record("derived_c1/final_network_summary")["payload"]["table"]["rows"]["B654_30x5"]
    prec_rec = c.record("precision_a_b400_b654")["payload"]["v2"]["30x5"][str(HEADLINE_B)]
    repro = {"final_network_B654_30x5": compare(c1_view(row), {k: fin[k] for k in ("MAE_S", "MAE_D", "A")},
                                                "results/derived_c1/final_network_summary.json payload.table.rows."
                                                "B654_30x5 (MAE_S, MAE_D with ci95; A with ci95, one-sided bound)"),
             "precision_v2_30x5_654": compare(precision_view(raw, HEADLINE_B), prec_rec,
                                              "results/precision_a_b400_b654.json payload.v2.30x5.654 (site, "
                                              "within-host and host-mean metrics of S, D and S - D)")}
    return tab, {"reproduction": repro, "n_seeds": n_s, "n_resamples": n_r}


def sweep_design(dstar: pd.DataFrame) -> tuple[dict, dict]:
    sweep, f_long = dca.load_sweep()
    dc, dc_info = dca.load_dclass(sweep)
    ds = dstar[(dstar.r < 10) & (dstar.seed < 3)]
    long = pd.concat([sweep[COLS], dc[COLS], ds[COLS]], ignore_index=True)
    ens = seed_ensemble(long)
    if (ens.n_seeds != 3).any():
        raise AssertionError("sweep: seed ensemble over the wrong number of seeds")
    ens["backbone"] = "v2"
    draws = sa.make_draws(ens)
    models = ("S", "D", "Dstar", "D-host", "D-site")
    st = {m: sa.stats_by_budget(ens, m, SWEEP_B) for m in models}
    gm = {k: ("mae",) for k, _, _ in CLASS_GAPS}
    tab, raw = design_table(st, draws, SWEEP_B, CORE_GAPS + CLASS_GAPS, gap_metrics=gm)
    for B in SWEEP_B:
        g = tab[str(B)]["gaps"]
        tab[str(B)]["Dstar_vs_class"] = {cl: relation(g[f"Dstar_minus_D{cl[2:]}"]) for cl in ("D-host", "D-site")}
    repro = {"final_network_B200_B400_B654": dca.reproduce(tab)}
    dg = c.record("derived_dclass/dclass_gaps")["payload"]["budgets"]
    keep = ("S", "D", "D-host", "D-site")
    dk = ("S_minus_D", "S_minus_Dhost", "S_minus_Dsite", "D_minus_Dhost", "D_minus_Dsite")
    repro["dclass_gaps"] = compare(
        {str(B): {"MAE": {m: tab[str(B)]["MAE"][m] for m in keep},
                  "gaps": {k: tab[str(B)]["gaps"][k] for k in dk}} for B in SWEEP_B},
        {str(B): {"MAE": dg[str(B)]["MAE"], "gaps": dg[str(B)]["gaps"]} for B in SWEEP_B},
        "results/derived_dclass/dclass_gaps.json payload.budgets.*.MAE and .gaps (S, D, D-host, D-site)")
    prec_rec = c.record("precision_a_b400_b654")["payload"]["v2"]["10x3"]
    repro["precision_v2_10x3"] = compare({str(B): precision_view(raw, B) for B in (400, 654)},
                                         {str(B): prec_rec[str(B)] for B in (400, 654)},
                                         "results/precision_a_b400_b654.json payload.v2.10x3.400/654 (site, "
                                         "within-host and host-mean metrics of S, D and S - D)")
    return tab, {"reproduction": repro, "draws": draws, "st": st, "staging_eval_long": f_long,
                 "dclass_runs": dc_info["n_runs"], "dclass_runs_at_epoch_limit": dc_info["n_runs_at_epoch_limit"]}


def staging(sw: dict) -> tuple[dict, dict]:
    """P (retrained P1 into the previous D) against D*, on the staging ensemble of analysis/m3/staging_fair.py."""
    sweep, _ = sf.load_sweep()
    p1_cell, _ = sf.load_p1()
    pc, info = sf.load_pcomp(sweep, p1_cell)
    old = sweep[sweep.model == "P"].assign(model="P_old")
    long = pd.concat([sweep[sweep.model.isin(["S", "D"])], old, pc], ignore_index=True)
    long = long[long.r >= 0]
    ens = sa.ensemble(long.drop(columns=["run_id"]))
    draws = sa.make_draws(ens)
    if not (np.array_equal(draws.pick, sw["draws"].pick) and np.array_equal(draws.w, sw["draws"].w)):
        raise AssertionError("staging draws differ from the sweep draws")
    res = sf.block(ens, draws)
    ref = c.record("derived_staging/staging_fair")["payload"]["budgets"]
    keys = ("mae", "S_minus_D", "P_minus_S", "P_minus_D")
    repro = compare({str(B): {k: res[str(B)][k] for k in keys} for B in SWEEP_B},
                    {str(B): {k: ref[str(B)][k] for k in keys} for B in SWEEP_B},
                    "results/derived_staging/staging_fair.json payload.budgets.200/400/654 (MAE of S, D, P, P_old; "
                    "S - D, P - S, P - D with every bound)")
    st_p = sa.stats_by_budget(ens, "P", SWEEP_B)
    out = {}
    for B in SWEEP_B:
        pds = sa.paired(st_p[B], sw["st"]["Dstar"][B], draws)
        pds["ci95_excludes_zero"] = bool(pds["ci95"][0] > 0 or pds["ci95"][1] < 0)
        pds["upper_below_delta"] = bool(pds["upper_one_sided_95"] < DELTA)
        r = res[str(B)]
        out[str(B)] = {"x_training_hosts": B, "MAE_P": r["mae"]["P"], "P_minus_Dstar": pds,
                       "P_minus_D": r["P_minus_D"], "P_minus_S": r["P_minus_S"]}
    note = ("P is the retrained P1 (stage p1v2b) composed with the previous D checkpoints (stage pcompv2b); "
            "P - D* therefore compares the staged network built on the previous energy head with the re-tuned "
            "network D*, two different networks, not the staging cost of D*")
    return out, {"reproduction": repro, "note": note, "pcompv2b_runs": info["n_runs"]}


# ---------------------------------------------------------------- earlier D-late runs

def scan_dlate() -> dict:
    """Every run record under results/ whose spec is D-late, grouped by directory and initialisation field."""
    found = Counter()
    for p in sorted((ROOT / "results").rglob("*.json")):
        if len(p.stem) != 16 or "predictions" in p.parts:
            continue
        try:
            sp = json.loads(p.read_text()).get("payload", {}).get("spec")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(sp, dict) and sp.get("model") == "D-late":
            init = (sp.get("hp") or {}).get("init", "default (no init field)")
            found[(str(p.parent.relative_to(ROOT)), init, int(sp.get("budget") or -1))] += 1
    return {f"{d} | init {i} | {b} hosts": n for (d, i, b), n in sorted(found.items())}


def dlate_v1(sw: dict) -> dict:
    """D-late of the default-initialisation network (results/dlate) against D-state of the same network."""
    from dftgnn.config import load_config

    cfg = load_config()
    b2a = {int(k): int(v) for k, v in cfg.tuning.budget_to_anchor.items()}
    t1 = yaml.safe_load((ROOT / "configs" / "tuned_v1.yaml").read_text())["models"]["D-late"]
    long, f_long = sa.load_table("staging_eval_long", c.record("staging_eval_table"))
    v1 = long[(long.backbone == "v1") & long.model.isin(["S", "D"]) & long.B.isin(SWEEP_B)]
    ref = {(int(B), int(r)): g.set_index("site_id") for (B, r), g in v1[(v1.model == "S") & (v1.seed == 0)]
           .groupby(["B", "r"])}
    files = sorted(DLATE_DIR.glob("*.json"))
    frames, cells, worst, capped, shas, max_ep = [], set(), 0.0, [], set(), set()
    for p in files:
        rec = json.loads(p.read_text())
        pl = rec["payload"]
        s = pl["spec"]
        if rec["git_dirty"] is not False or pl["run_id"] != p.stem or rec["name"] != p.stem or pl["smoke"]:
            raise AssertionError(f"dlate {p.stem}: dirty, misnamed or smoke")
        B, r, seed = int(s["budget"]), int(s["r"]), int(s["seed"])
        t = t1[b2a[B]]
        same = (s["model"] == "D-late" and s["lr"] == t["learning_rate"] and s["weight_decay"] == t["weight_decay"]
                and s["batch_size"] == t["batch_size"] and "init" not in s["hp"]
                and all(s["hp"][k] == t[k] for k in ("hidden_width", "megnet_blocks", "dropout", "readout_mlp_width",
                                                     "pooling"))
                and s["tags"] == {"sensitivity": "unselected_variant"})
        if not same:
            raise AssertionError(f"dlate {p.stem}: spec is not the tuned_v1 D-late setting of anchor {b2a[B]}")
        if (B, r, seed) in cells:
            raise AssertionError(f"dlate {p.stem}: duplicate cell")
        cells.add((B, r, seed))
        path = ROOT / pl["predictions"]["path"]
        if c.sha256(path) != pl["predictions"]["sha256"]:
            raise AssertionError(f"dlate {p.stem}: prediction sha256")
        d = pd.read_parquet(path)
        if not np.isfinite(d[["y_true", "y_pred"]].to_numpy()).all():
            raise AssertionError(f"dlate {p.stem}: non-finite predictions")
        worst = max(worst, abs(float(np.abs(d.y_true - d.y_pred).mean()) - pl["metrics"]["mae"]))
        base = ref[(B, r)]
        if len(d) != len(base) or set(d.site_id) != set(base.index) or d.site_id.duplicated().any():
            raise AssertionError(f"dlate {p.stem}: test sites differ from the v1 S run of the cell")
        j = d.set_index("site_id").loc[base.index]
        if not (np.array_equal(j.y_true.to_numpy(), base.y_true.to_numpy()) and (j.host_id == base.host_id).all()):
            raise AssertionError(f"dlate {p.stem}: targets or hosts differ from the v1 S run of the cell")
        if pl["epochs_run"] >= pl["max_epochs"]:
            capped.append(p.stem)
        shas.add(rec["git_sha"])
        max_ep.add(int(pl["max_epochs"]))
        frames.append(d.assign(model="D-late", r=r, B=B, seed=seed)[COLS])
    if worst > 1e-6 or len(cells) != 90 or len(files) != 90:
        raise AssertionError(f"dlate: {len(files)} records, {len(cells)} cells, MAE difference {worst}")
    allp = pd.concat([v1[COLS], *frames], ignore_index=True)
    ens = seed_ensemble(allp)
    ens["backbone"] = "v1"
    draws = sa.make_draws(ens)
    if not (np.array_equal(draws.pick, sw["draws"].pick) and np.array_equal(draws.w, sw["draws"].w)):
        raise AssertionError("dlate draws differ from the sweep draws")
    st = {m: sa.stats_by_budget(ens, m, SWEEP_B) for m in ("S", "D", "D-late")}
    gaps = (("S_minus_D", "S", "D"), ("S_minus_Dlate", "S", "D-late"), ("Dlate_minus_D", "D-late", "D"))
    tab, _ = design_table(st, draws, SWEEP_B, gaps, metrics=("mae",))
    for B in SWEEP_B:
        tab[str(B)].pop("metrics")
    abl = c.record("derived_c1/default_init_ablation")["payload"]["table"]["rows"]
    repro = compare({f"B{B}": c1_view(tab[str(B)]) for B in SWEEP_B},
                    {f"B{B}": {k: abl[f"B{B}"][k] for k in ("MAE_S", "MAE_D", "A")} for B in SWEEP_B},
                    "results/derived_c1/default_init_ablation.json payload.table.rows.B200/B400/B654 (v1 S, D, A)")
    if not repro["reproduced"]:
        raise AssertionError(f"v1 S and D do not reproduce the default-initialisation table: {repro}")
    tuned_v2 = yaml.safe_load((ROOT / "configs" / "tuned_v2.yaml").read_text())
    scan = scan_dlate()
    final_elsewhere = {k: v for k, v in scan.items() if "kaiming" in k and not k.startswith("results/dretrain_v2")}
    facts = {"directory": "results/dlate", "n_runs": len(files), "budgets": sorted({b for b, _, _ in cells}),
             "resamples": sorted({r for _, r, _ in cells}), "seeds": sorted({s for _, _, s in cells}),
             "variant": "D-late", "tags": {"sensitivity": "unselected_variant"},
             "settings": "configs/tuned_v1.yaml models.D-late at the anchor of each budget (budget_to_anchor)",
             "initialisation": "no init field in hp: default initialisation (the v1 network); configs/tuned_v2.yaml "
                               f"sets arch.init = {tuned_v2['arch']['init']} and has no D-late entry",
             "code_commits": sorted(shas), "max_epochs": sorted(max_ep), "n_at_epoch_limit": len(capped),
             "runs_at_epoch_limit": sorted(capped), "max_abs_diff_recorded_mae_vs_predictions": worst,
             "network": "default-initialisation network (v1), not the final network",
             "final_network_D_late_runs_outside_dretrain_v2": "none exist" if not final_elsewhere else final_elsewhere,
             "scan_of_results_run_records": scan}
    return {"facts": facts, "budgets": tab, "reproduction": repro,
            "note": "context only: default-initialisation network, 10 x 3 design, draws of the sweep; "
                    "Dlate_minus_D = MAE_D-late - MAE_D-state (positive = D-state more accurate)",
            "inputs": {"staging_eval_long": f_long}}


# ---------------------------------------------------------------- N*, outcome inputs, main

def nstar_block(head: dict, sw: dict) -> dict:
    def up(tab, B, k):
        return tab[str(B)]["gaps"][k]["upper_one_sided_95"]
    out = {}
    for tag, k in (("Dstar", "S_minus_Dstar"), ("D_old", "S_minus_D")):
        sweep_u = {B: up(sw, B, k) for B in SWEEP_B}
        out[tag] = {"sweep_10x3": n_star(sweep_u, DELTA),
                    "with_654_30x5": n_star({**sweep_u, HEADLINE_B: up(head, HEADLINE_B, k)}, DELTA)}
    ref = c.record("c1c2_v2")["payload"]["N_star"]
    out["paper_reference"] = {"N_star_old_D_full_curve": ref["N_star"], "source": "results/c1c2_v2.json "
                              "payload.N_star (budgets 25-654, 10 x 3)",
                              "note": "D* exists only at 200, 400 and 654 hosts, so its N* is over those three"}
    return out


def outcome_inputs(head: dict) -> dict:
    row = head[str(HEADLINE_B)]
    dd, a = row["gaps"]["D_minus_Dstar"], row["gaps"]["S_minus_Dstar"]
    return {"design": "654 hosts, 30 resamples x 5 seeds", "MAE_D": row["MAE"]["D"], "MAE_Dstar": row["MAE"]["Dstar"],
            "MAE_S": row["MAE"]["S"],
            "D_minus_Dstar": {k: dd[k] for k in ("A", "ci95", "inside_pm_0p01", "abs_point_le_0p01",
                                                 "ci95_excludes_zero")},
            "A_old": {k: row["gaps"]["S_minus_D"][k] for k in ("A", "ci95", "upper_one_sided_95", "upper_below_delta")},
            "A_star": {k: a[k] for k in ("A", "ci95", "upper_one_sided_95", "upper_below_delta",
                                         "two_sided_within_pm_delta", "ci95_excludes_zero")},
            "Dstar_beats_D_by_more_than_0p01": bool(dd["A"] > SAME_EV)}


def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    cfg = load_config()
    if float(cfg.stats.delta_eV) != DELTA:
        raise AssertionError("stats.delta_eV differs from 0.05")
    ref, f_ref = s_reference()
    dstar, info = load_dstar(ref)
    head, head_meta = headline(dstar)
    sw, sw_meta = sweep_design(dstar)
    stg, stg_meta = staging(sw_meta)
    dl = dlate_v1(sw_meta)
    repro = {**{f"headline.{k}": v for k, v in head_meta["reproduction"].items()},
             **{f"sweep.{k}": v for k, v in sw_meta["reproduction"].items()},
             "staging.staging_fair": stg_meta["reproduction"], "dlate_v1.default_init_ablation": dl["reproduction"]}
    bad = [k for k, v in repro.items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"not reproduced: {bad}")
    payload = {
        "label": LABEL, "delta_eV": DELTA, "same_within_eV": SAME_EV, "sign_convention": SIGNS,
        "designs": {"headline_654_30x5": "654 training hosts, outer resamples r = 0-29 x seeds 0-4, seed-ensemble "
                                         "mean over 5 seeds; S and D from results/analysis/precision_eval_long.parquet "
                                         "(v2); HierDraws.make([164] * 30, 2000, seed 20261008)",
                    "sweep_10x3": "200, 400, 654 training hosts, outer resamples r = 0-9 x seeds 0-2, seed-ensemble "
                                  "mean over 3 seeds; S and D from results/analysis/staging_eval_long.parquet (v2); "
                                  "draws of analysis/staging_analysis.make_draws (2000, seed 20261008); D* at 654 "
                                  "hosts is the r = 0-9, seeds 0-2 subset of its 150 runs",
                    "Dstar": "dretrain_v2: D-late at 654 hosts (30 x 5) and 400 hosts (10 x 3, 654-anchor settings), "
                             "D-state at 200 hosts (10 x 3); settings of configs/tuned_d_v2.yaml, epoch limit 400"},
        "estimator": "dftgnn.stats.g1 (seed_ensemble, resample_stats, HierDraws, contrast); interval and bounds by "
                     "analysis/c1/final_network.a_entry; inside_pm_0p01 = two-sided 95% interval strictly inside "
                     "(-0.01, +0.01) eV; n_resamples_above_delta counts per-resample gaps strictly above delta",
        "headline_654_30x5": head, "sweep_10x3": sw, "n_star": nstar_block(head, sw),
        "staging": {"budgets": stg, "note": stg_meta["note"]},
        "earlier_dlate": dl, "health": health(info),
        "outcome_inputs": outcome_inputs(head),
        "reproduction": repro,
        "reproduction_summary": {"max_abs_difference": max(v["max_abs_difference"] for v in repro.values()),
                                 "all_reproduced_below_1e-9": not bad,
                                 "n_values_compared": sum(v["n_values_compared"] for v in repro.values())},
        "inputs": {**f_ref, "dretrain_v2_spec": {"path": "specs/dretrain_v2.jsonl", "sha256": info["spec_sha256"]},
                   "dretrain_v2_manifest": "results/nrp/artefacts/dretrain_v2.sha256",
                   "dclass_manifest": "results/nrp/artefacts/dclass.sha256",
                   "tuned_d": {"path": "configs/tuned_d_v2.yaml",
                               "sha256": c.sha256(ROOT / "configs" / "tuned_d_v2.yaml")}},
    }
    payload = c.to_jsonable(payload)
    if "--dry-run" in sys.argv:
        print(json.dumps(payload, indent=1))
        return
    print(write_result("dstar_gaps", payload, config=cfg, repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
