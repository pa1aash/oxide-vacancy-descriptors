"""The descriptor gain S - D* by cation d-shell class (primary test G1), 654 hosts, 30 x 5.

Physics: if the electronic descriptors of the pristine cell help most where the two electrons left by a neutral O
vacancy go into a filled-d host's cation s-like band edge (d10: Zn2+, Cd2+, Ga3+ ...), the descriptor network should
gain more over the structure-only network on d10 hosts than on the others. The classes are those of the committed
record results/derived_dshell/dshell_classes.json (analysis/dshell/classes.py, fixed before any gain was computed;
its sha256 is asserted here).

Estimator (st1b common brief, implemented exactly):
    predictions  S, D (old D-state) and D* at 654 hosts, outer resamples r = 0-29 x seeds 0-4, loaded and checked
                 by analysis/dstar (s_reference, load_dstar, dstar_extras.frames); seed ensemble per (model, r, site)
    host stats   per resample, per test host: dftgnn.stats.metrics.host_stats on the host's tested sites, so
                 within-host residuals use host means over ALL tested sites of the host
    draws        headline HierDraws (2000 replicates, seed 20261008, 164 test hosts per resample), unchanged
    class value  pooled ratio: per replicate, (sum over slots of the draw-weighted numerator of the class's hosts)
                 / (sum over slots of the draw-weighted denominator); point = the same ratio over the 30 resamples
                 with unit weights. Site MAE: sum |p - y| / n sites; within-host MAE: sum of |residual| / n sites in
                 hosts with >= 2 tested sites. Interval: 2.5 / 97.5 percentiles (numpy linear).
    gains        S - D* and S - D per replicate (paired); contrasts gain(c) - gain(c') per replicate.

Primary (G1): gain_d10 - gain_rest of S - D* on the within-host residual MAE, rest = d0 + partial_d + main_group;
G1 passes iff the lower end of its 95% interval is > 0. Secondary (no multiplicity adjustment): the same contrast on
site MAE; each class against the other three classified classes on both metrics; mixed_f and unassigned gains;
main_group split by post_transition_d10_core (descriptive).

Checks: the headline S - D* within-host gap (results/derived_dstar/dstar_extras.json a_within.headline_654_30x5)
and the site-MAE gaps and MAEs (results/derived_dstar/dstar_gaps.json headline_654_30x5.654) are reproduced to
1e-9 eV with the mean-over-resamples estimator; the pooled machinery with the mean over slots equals
HierDraws.replicates on all hosts (1e-12).

Writes through ``write_result`` into ``results/derived_dshell/dshell_gain.json``.

    PYTHONPATH=$PWD/src python analysis/dshell/dshell_gain.py [--dry-run [PATH]]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "derived_dshell"
CLASS_RECORD = "results/derived_dshell/dshell_classes.json"
CLASS_SHA256 = "c7aa9487d22173f0e9b701a078d79bad3922c856ede149fc7c4318f02d93a86d"
B, N_R, N_S, N_TEST = 654, 30, 5, 164
TOL = 1e-9
CLASSES = ("d0", "d10", "partial_d", "main_group")
BINS = ("mixed_f", "unassigned")
REST = ("d0", "partial_d", "main_group")
MODELS = ("S", "D", "Dstar")
GAINS = (("S_minus_Dstar", "S", "Dstar"), ("S_minus_D", "S", "D"))
METRIC_COLS = {"mae": ("sabs", "n"), "within_host_mae": ("wsum", "wn")}


# ---------------------------------------------------------------- pooled class-restricted estimator

def slot_totals(stats: Sequence[np.ndarray], masks: Sequence[np.ndarray], pick: np.ndarray,
                w: np.ndarray) -> np.ndarray:
    """(n_boot, R, k) draw-weighted column sums of the class's host rows in each slot.

    ``stats[j]`` (hosts x k) and ``masks[j]`` (hosts,) belong to resample j; ``pick[b, s]`` is the resample of slot s
    and ``w[b, s, :]`` its host multiplicities (headline draws)."""
    n_boot, R = pick.shape
    tot = np.zeros((n_boot, R, stats[0].shape[1]))
    for j, (s, m) in enumerate(zip(stats, masks, strict=True)):
        sel = pick == j
        tot[sel] = w[sel][:, : len(s)] @ (np.asarray(s, float) * np.asarray(m, float)[:, None])
    return tot


def pooled(stats, masks, pick, w, num: int, den: int) -> tuple[float, np.ndarray]:
    """Point (unit weights over the resamples) and replicates of sum(numerator) / sum(denominator)."""
    pt = sum((np.asarray(s, float) * np.asarray(m, float)[:, None]).sum(axis=0) for s, m in zip(stats, masks, strict=True))
    tot = slot_totals(stats, masks, pick, w).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(pt[num] / pt[den]), tot[:, num] / tot[:, den]


def summary(point: float, reps: np.ndarray) -> dict:
    ok = np.isfinite(reps)
    return {"point": float(point), "ci95": [float(np.quantile(reps[ok], 0.025)), float(np.quantile(reps[ok], 0.975))],
            "boot_sd": float(reps[ok].std(ddof=1)), "n_replicates_undefined": int((~ok).sum())}


# ---------------------------------------------------------------- inputs

def load_classes() -> dict:
    from analysis import a1_common as c

    path = ROOT / CLASS_RECORD
    if c.sha256(path) != CLASS_SHA256:
        raise AssertionError(f"{CLASS_RECORD}: sha256 differs from the committed class record")
    hosts = json.loads(path.read_text())["payload"]["hosts"]
    return {h["host_id"]: h for h in hosts}


def headline_stats() -> tuple[dict, list[np.ndarray], dict]:
    """Per model, the 30 per-resample host-stat arrays; per resample the sorted test hosts; input fingerprints."""
    from analysis import precision_analysis as pa
    from analysis.dstar import dstar_analysis as da
    from analysis.dstar import dstar_extras as dx
    from dftgnn.stats.g1 import seed_ensemble

    ref, _ = da.s_reference()
    dstar, info = da.load_dstar(ref)
    fr = dx.frames(dstar)
    hd = fr["headline"]
    hd = hd[(hd.r < N_R) & (hd.seed < N_S)]
    ens = seed_ensemble(hd[da.COLS])
    if (ens.n_seeds != N_S).any():
        raise AssertionError("seed ensemble over the wrong number of seeds")
    hosts = []
    for r in range(N_R):
        hs = [sorted(ens[(ens.model == m) & (ens.r == r)].host_id.unique()) for m in MODELS]
        if any(h != hs[0] for h in hs) or len(hs[0]) != N_TEST:
            raise AssertionError(f"r = {r}: test hosts differ between models or from {N_TEST}")
        hosts.append(np.array(hs[0]))
    st = {m: pa.resample_stats_by_budget(ens, m, N_R, (B,))[B] for m in MODELS}
    return st, hosts, {**fr["inputs"], "dretrain_v2_spec_sha256": info["spec_sha256"], "n_dstar_runs": len(info["runs"]),
                       "ens": ens}


# ---------------------------------------------------------------- reproduction of the headline

def reproduce(st: dict, draws) -> dict:
    from analysis import a1_common as c
    from dftgnn.stats.g1 import contrast

    got, want = {}, {}
    ex = c.record("derived_dstar/dstar_extras")["payload"]["a_within"]["headline_654_30x5"]
    gp = c.record("derived_dstar/dstar_gaps")["payload"]["headline_654_30x5"][str(B)]
    for k, a, b in (("S_minus_Dstar", "S", "Dstar"), ("S_minus_D", "S", "D")):
        wd = contrast({B: st[a]}, {B: st[b]}, draws, "within_host_mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B]["diff"]
        sd = contrast({B: st[a]}, {B: st[b]}, draws, "mae", ci=c.CI, upper_quantile=c.UPPER_Q)[B]
        got[f"within.{k}"] = [wd["mean"], *wd["ci"]]
        want[f"within.{k}"] = [ex[k]["A"], *ex[k]["ci95"]]
        got[f"mae.{k}"] = [sd["diff"]["mean"], *sd["diff"]["ci"]]
        want[f"mae.{k}"] = [gp["gaps"][k]["A"], *gp["gaps"][k]["ci95"]]
        for m, s in ((a, sd["a"]), (b, sd["b"])):
            got[f"mae.MAE_{m}"] = [s["mean"], *s["ci"]]
            want[f"mae.MAE_{m}"] = [gp["MAE"][m]["mean"], *gp["MAE"][m]["ci95"]]
    diff = max(abs(x - y) for k in got for x, y in zip(got[k], want[k], strict=True))
    if diff > TOL:
        raise AssertionError(f"headline not reproduced: max |difference| {diff}")
    return {"values": {k: {"point": v[0], "ci95": v[1:]} for k, v in got.items()}, "max_abs_difference": diff,
            "reproduced_below_1e-9": True, "n_values_compared": sum(len(v) for v in got.values()),
            "sources": [("results/derived_dstar/dstar_extras.json payload.a_within.headline_654_30x5."
                         "{S_minus_Dstar, S_minus_D} (A, ci95)"),
                        ("results/derived_dstar/dstar_gaps.json payload.headline_654_30x5.654 gaps.{S_minus_Dstar, "
                         "S_minus_D} (A, ci95) and MAE.{S, D, Dstar} (mean, ci95)")]}


def check_slot_machinery(st: dict, draws) -> float:
    """Mean over slots of the slot ratio, from slot_totals, equals HierDraws.replicates on all hosts."""
    from dftgnn.stats.metrics import _COLS

    worst = 0.0
    ones = [np.ones(len(s)) for s in st["S"]]
    for metric, (nu, de) in METRIC_COLS.items():
        tot = slot_totals(st["S"], ones, draws.pick, draws.w)
        mine = (tot[..., _COLS.index(nu)] / tot[..., _COLS.index(de)]).mean(axis=1)
        worst = max(worst, float(np.abs(mine - draws.replicates(st["S"], metric)).max()))
    if worst > 1e-12:
        raise AssertionError(f"slot totals differ from HierDraws.replicates by {worst}")
    return worst


# ---------------------------------------------------------------- groups

def group_masks(classes: dict, hosts: list[np.ndarray]) -> dict:
    """Group name -> per-resample boolean host masks (rows in host_id order)."""
    cls = lambda h: classes[h]["class"]
    flag = lambda h: classes[h]["post_transition_d10_core"]
    sets = {"all_hosts": lambda h: True, **{k: (lambda h, k=k: cls(h) == k) for k in CLASSES + BINS},
            "rest": lambda h: cls(h) in REST,
            **{f"not_{k}": (lambda h, k=k: cls(h) in CLASSES and cls(h) != k) for k in CLASSES},
            "main_group_post_transition_true": lambda h: cls(h) == "main_group" and flag(h),
            "main_group_post_transition_false": lambda h: cls(h) == "main_group" and not flag(h)}
    return {g: [np.array([f(h) for h in hs]) for hs in hosts] for g, f in sets.items()}


def group_counts(classes: dict, masks: list[np.ndarray], hosts: list[np.ndarray], wn: list[np.ndarray],
                 ens) -> dict:
    members = sorted({h for m, hs in zip(masks, hosts, strict=True) for h in hs[m]})
    within = sorted({h for m, hs, n in zip(masks, hosts, wn, strict=True) for h in hs[m & (n > 0)]})
    tested_sites = ens[(ens.model == "S") & ens.host_id.isin(members)].site_id.nunique()
    return {"n_hosts": len(members), "n_hosts_within_host_metric": len(within),
            "n_sites_universe": int(sum(classes[h]["n_sites"] for h in members)), "n_sites_tested": int(tested_sites),
            "n_host_slots_over_30_resamples": int(sum(m.sum() for m in masks))}


def evaluate(st: dict, masks: dict, draws) -> tuple[dict, dict]:
    """Per group: pooled MAE of every model and the pooled gains; also the gain replicates for contrasts."""
    from dftgnn.stats.metrics import _COLS

    out, reps = {}, {}
    for g, mk in masks.items():
        row = {"models": {}, "gains": {}}
        val = {}
        for metric, (nu, de) in METRIC_COLS.items():
            for m in MODELS:
                val[(m, metric)] = pooled(st[m], mk, draws.pick, draws.w, _COLS.index(nu), _COLS.index(de))
                row["models"].setdefault(m, {})[metric] = summary(*val[(m, metric)])
            for k, a, b in GAINS:
                p, r = val[(a, metric)][0] - val[(b, metric)][0], val[(a, metric)][1] - val[(b, metric)][1]
                row["gains"].setdefault(k, {})[metric] = summary(p, r)
                reps[(g, k, metric)] = (p, r)
        out[g] = row
    return out, reps


def contrast_entry(reps: dict, ga: str, gb: str, metric: str, gain: str = "S_minus_Dstar") -> dict:
    pa_, ra = reps[(ga, gain, metric)]
    pb, rb = reps[(gb, gain, metric)]
    e = summary(pa_ - pb, ra - rb)
    e.update({"a": ga, "b": gb, "metric": metric, "gain": gain, "lower_gt_0": bool(e["ci95"][0] > 0),
              "ci95_excludes_zero": bool(e["ci95"][0] > 0 or e["ci95"][1] < 0)})
    return e


def run() -> dict:
    from analysis import a1_common as c
    from dftgnn.stats.g1 import HierDraws
    from dftgnn.stats.metrics import _COLS

    classes = load_classes()
    st, hosts, inputs = headline_stats()
    ens = inputs.pop("ens")
    missing = sorted({h for hs in hosts for h in hs} - set(classes))
    if missing:
        raise AssertionError(f"test hosts without a class: {missing[:5]}")
    draws = HierDraws.make([N_TEST] * N_R, c.N_BOOT, seed=c.BOOT_SEED)
    repro = reproduce(st, draws)
    worst = check_slot_machinery(st, draws)
    masks = group_masks(classes, hosts)
    wn = [s[:, _COLS.index("wn")] for s in st["S"]]
    groups, reps = evaluate(st, masks, draws)
    for g in groups:
        groups[g] = {"counts": group_counts(classes, masks[g], hosts, wn, ens), **groups[g]}
    primary = contrast_entry(reps, "d10", "rest", "within_host_mae")
    primary["g1_pass"] = primary.pop("lower_gt_0")
    secondary = {"d10_minus_rest_mae": contrast_entry(reps, "d10", "rest", "mae")}
    for k in CLASSES:
        for metric in METRIC_COLS:
            secondary[f"{k}_minus_others_{metric}"] = contrast_entry(reps, k, f"not_{k}", metric)
    pt = {metric: contrast_entry(reps, "main_group_post_transition_true", "main_group_post_transition_false", metric)
          for metric in METRIC_COLS}
    head = repro["values"]
    return {
        "label": "ST1b: descriptor gain S - D* by cation d-shell class, 654 hosts, 30 x 5, headline draws, "
                 "class-restricted pooled estimator",
        "sign_convention": "gain = MAE_S - MAE_model (positive = the descriptor network more accurate); contrast "
                           "= gain(a) - gain(b), per replicate on the shared draws",
        "estimator": {"draws": {"n_boot": c.N_BOOT, "seed": c.BOOT_SEED, "n_resamples": N_R, "n_seeds": N_S,
                                "n_test_hosts_per_resample": N_TEST, "budget": B},
                      "pooled": "replicate = sum over slots of weighted class numerator / sum of weighted class "
                                "denominator; point = same with unit weights over the 30 resamples; mae = sabs / n; "
                                "within_host_mae = wsum / wn (hosts with >= 2 tested sites)",
                      "interval": "2.5 / 97.5 percentiles, numpy linear"},
        "primary": {"name": "G1: gain_d10 - gain_rest, S - D*, within-host residual MAE",
                    "rest": list(REST), "flag": "primary", **primary},
        "secondary": {"flag": "secondary, no multiplicity adjustment", "contrasts": secondary,
                      "bins_reported_in_groups": list(BINS)},
        "descriptive": {"flag": "descriptive secondary: main_group split by post_transition_d10_core",
                        "contrast_true_minus_false": pt},
        "groups": groups,
        "headline_vs_pooled_all_hosts": {
            metric: {"mean_over_resamples": {k: head[f"{'within' if metric == 'within_host_mae' else 'mae'}.{k}"]
                                             for k, _, _ in GAINS},
                     "pooled": {k: groups["all_hosts"]["gains"][k][metric] for k, _, _ in GAINS}}
            for metric in METRIC_COLS},
        "reproduction": repro,
        "slot_machinery_max_abs_difference": worst,
        "inputs": {**inputs, "class_record": {"path": CLASS_RECORD, "sha256": CLASS_SHA256}},
    }


def main(argv: Sequence[str] | None = None) -> None:
    from analysis import a1_common as c

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", nargs="?", const="-", default=None, metavar="PATH",
                    help="write the payload JSON to PATH (stdout if omitted) instead of calling write_result")
    a = ap.parse_args(argv)
    payload = c.to_jsonable(run())
    if a.dry_run is not None:
        text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if a.dry_run == "-":
            print(text, end="")
        else:
            out = Path(a.dry_run).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(out)
        return
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    print(write_result("dshell_gain", payload, config=load_config(), repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
