"""Robustness of the within-oxide descriptor gain by cation d-shell configuration (654 hosts, 30 x 5).

Physics: the within-host residual MAE measures how well a network orders the inequivalent O sites of one
oxide, the part of the neutral-vacancy energy that the host mean cannot carry. Whether the pristine-cell
electronic descriptors (D*) help that ordering more in some electronic configurations than in others is a
question about where the two electrons left by the neutral vacancy go: into an empty cation d band (d0), a
filled-d host's cation s-like band edge (transition-metal d10: Zn2+, Cd2+, Hg2+, Cu+, Ag+), an open d shell
(partial d), or a main-group cation. For post-transition main-group cations the empty frontier state depends
on the oxidation state: a cation at its group oxidation state (s0: Ga3+, In3+, Sn4+, Sb5+ ...) has an empty ns
band at the conduction-band edge, while a cation two below it (s2: Sn2+, Sb3+, Pb2+, Bi3+ ...) keeps the ns
pair occupied. Within the transition-metal d10 class, PBEsol+U applies U = 5 eV only on Cu and Zn d, so Zn and
Cu(I) hosts sit on a different functional footing from Cd, Hg and Ag(I) hosts. This script tests whether the
measured contrast between transition-metal d10 hosts and the rest is robust to Monte-Carlo noise, to
resampling distinct hosts, to the earlier filled-(n-1)d class definition, and to normalisation by the spread
there is to resolve.

Estimator: the pooled class estimator of analysis/dshell/dshell_gain.py (imported: headline_stats, load_classes,
group_masks, group_counts, pooled, summary, reproduce): seed ensemble per (model, r, site); per-resample host
statistics; per replicate, class value = (sum over the 30 slots of the draw-weighted numerator of the class's hosts)
/ (sum of the draw-weighted denominator); point = the same ratio with unit weights. Within-host MAE = wsum / wn
(hosts with >= 2 tested sites, residuals about the host means over all tested sites); site MAE = sabs / n. Gains
S - D* per replicate (paired); contrasts gain(a) - gain(b) per replicate. Paper draws HierDraws.make([164] * 30,
2000, 20261008); 20000-draw draws are generated in chunks that consume the generator exactly as HierDraws.make does
(checked equal to HierDraws.make on the paper draws). Two-sided 95% interval = 2.5 / 97.5 percentiles (numpy
linear); U = 95th percentile.

Blocks
    mc_stability           primary contrast gain_d10 - gain_rest (rest = d0 + partial_d + main_group) on
                           HierDraws([164] * 30, 20000, seed) for seeds 20261008, 1, 2, 3, 4; ci95 ends and U per
                           seed; mean and SD of each end over seeds (Monte-Carlo SE); fraction of the 5 x 20000
                           replicates >= 0.
    host_cluster           bootstrap over the 818 distinct hosts (each drawn host carries all its test appearances
                           over the 30 resamples), 20000 multinomial replicates, default_rng(20261008).
    filled_d_bins          the primary contrast with the bins of data/host_dbins_v1.csv (d10 = at least one
                           filled-d cation incl. Ga3+, In3+, Sn4+, Pb2+, Bi3+ and none partial) against d0 + other
                           of that file (unassigned excluded); paper and 20000-draw intervals; cross-tabulation.
    relative               per published class: zero-skill within-host value (predictor constant within the host),
                           S and D* within-host MAE, gain, gain / MAE_S, gain / zero-skill; contrasts d10 - rest.
    post_transition_split  main-group hosts with a post-transition cation split into s0, s2 and mixed by the
                           oxidation states of results/derived_dshell/dshell_classes.json.
    d10_split              transition-metal d10: group 12 (Zn, Cd, Hg) vs group 11 (Cu+, Ag+, Au+); U-treated
                           (any Zn or Cu cation) vs untreated.

Reproduction: the stored primary contrast (point, ci95, boot_sd) and every stored S - D* group gain of
results/derived_dshell/dshell_gain.json, plus the headline gaps checked by dshell_gain.reproduce, to 1e-9.

Writes through ``write_result`` into ``results/derived_trv1_dshell/dshell_robust.json``.

    PYTHONPATH=$PWD/src python analysis/trv1/dshell_robust.py [--dry-run --out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import a1_common as c
from analysis.dshell import dshell_gain as dg

OUT_DIR = ROOT / "results" / "derived_trv1_dshell"
DBINS = "data/host_dbins_v1.csv"
GAIN_REC = "results/derived_dshell/dshell_gain.json"
N_R, N_TEST = dg.N_R, dg.N_TEST
PAPER_DRAWS, PAPER_SEED = 2000, 20261008
BIG = 20000
MC_SEEDS = (20261008, 1, 2, 3, 4)
CHUNK = 1000
SMALL_N = 10
TOL = 1e-9
QTY = ("n", "wn", "sabs_S", "sabs_Dstar", "wsum_S", "wsum_Dstar", "zsum")
U_TREATED = ("Zn", "Cu")           # row 4: U_eff = 5 eV on Cu and Zn d (and Ce f)
GROUP12, GROUP11 = ("Zn", "Cd", "Hg"), ("Cu", "Ag", "Au")


# ---------------------------------------------------------------- draws

def draw_chunks(n_hosts: Sequence[int], n_boot: int, seed: int, chunk: int = CHUNK):
    """Yield (pick, w) blocks of HierDraws.make(n_hosts, n_boot, seed), consuming the generator identically."""
    rng = np.random.default_rng(seed)
    R, H = len(n_hosts), max(n_hosts)
    for b0 in range(0, n_boot, chunk):
        m = min(chunk, n_boot - b0)
        pick = np.empty((m, R), dtype=int)
        w = np.zeros((m, R, H))
        for b in range(m):
            for k in range(R):
                j = int(rng.integers(0, R))
                n = n_hosts[j]
                pick[b, k] = j
                w[b, k, :n] = rng.multinomial(n, np.full(n, 1.0 / n))
        yield pick, w


def accumulate(X: Sequence[np.ndarray], pick: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(n_boot, K) sums over slots of w[b, k] @ X[pick[b, k]] (pooled totals of every column)."""
    tot = np.zeros((pick.shape[0], X[0].shape[1]))
    for j, x in enumerate(X):
        b_idx, k_idx = np.nonzero(pick == j)
        np.add.at(tot, b_idx, w[b_idx, k_idx, : len(x)] @ x)
    return tot


def hier_totals(X, n_boot: int, seed: int) -> np.ndarray:
    return np.concatenate([accumulate(X, p, w) for p, w in draw_chunks([N_TEST] * N_R, n_boot, seed)])


# ---------------------------------------------------------------- per-host feature columns

def zero_skill_sums(ens, hosts: list[np.ndarray]) -> list[np.ndarray]:
    """Per resample, per test host (sorted): sum over tested sites of |y - host mean y| (0 for one-site hosts)."""
    s = ens[ens.model == "S"].copy()
    s["dev"] = (s.y_true - s.groupby(["r", "host_id"]).y_true.transform("mean")).abs()
    s["ns"] = s.groupby(["r", "host_id"]).y_true.transform("size")
    s["dev"] = s.dev.where(s.ns >= 2, 0.0)
    out = []
    for r, hs in enumerate(hosts):
        z = s[s.r == r].groupby("host_id").dev.sum()
        if list(z.index) != list(hs):
            raise AssertionError(f"r = {r}: zero-skill host rows differ from the host-stat rows")
        out.append(z.to_numpy())
    return out


def base_columns(st: dict, z: list[np.ndarray]) -> list[np.ndarray]:
    """Per resample (164, 7) array of the QTY columns."""
    from dftgnn.stats.metrics import _COLS

    ix = {k: _COLS.index(k) for k in ("n", "wn", "sabs", "wsum")}
    for a, b in zip(st["S"], st["Dstar"], strict=True):
        if not (np.array_equal(a[:, ix["n"]], b[:, ix["n"]]) and np.array_equal(a[:, ix["wn"]], b[:, ix["wn"]])):
            raise AssertionError("site counts differ between S and D*")
    return [np.column_stack([s[:, ix["n"]], s[:, ix["wn"]], s[:, ix["sabs"]], d[:, ix["sabs"]], s[:, ix["wsum"]],
                             d[:, ix["wsum"]], zz])
            for s, d, zz in zip(st["S"], st["Dstar"], z, strict=True)]


def design(base: list[np.ndarray], masks: dict) -> tuple[list[np.ndarray], list[str]]:
    """Columns (group, quantity) for every group: X_j = base_j * mask_j[g] blocks."""
    names = list(masks)
    X = [np.concatenate([b * masks[g][j][:, None] for g in names], axis=1) for j, b in enumerate(base)]
    return X, names


class Totals:
    """Point and replicate totals per (group, quantity), with the derived class quantities."""

    def __init__(self, point: np.ndarray, reps: np.ndarray, names: list[str]):
        self.names, self.point, self.reps = names, point, reps

    def col(self, g: str, q: str) -> tuple[float, np.ndarray]:
        k = self.names.index(g) * len(QTY) + QTY.index(q)
        return float(self.point[k]), self.reps[:, k]

    def ratio(self, g: str, num: str, den: str) -> tuple[float, np.ndarray]:
        (pn, rn), (pd_, rd) = self.col(g, num), self.col(g, den)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (pn / pd_ if pd_ else float("nan")), rn / rd

    def quantities(self, g: str) -> dict:
        """(point, reps) of the per-class quantities."""
        q = {"within_S": self.ratio(g, "wsum_S", "wn"), "within_Dstar": self.ratio(g, "wsum_Dstar", "wn"),
             "within_zero_skill": self.ratio(g, "zsum", "wn"),
             "site_S": self.ratio(g, "sabs_S", "n"), "site_Dstar": self.ratio(g, "sabs_Dstar", "n")}
        for m in ("within", "site"):
            q[f"gain_{m}"] = tuple(a - b for a, b in zip(q[f"{m}_S"], q[f"{m}_Dstar"], strict=True))
            with np.errstate(divide="ignore", invalid="ignore"):
                q[f"gain_over_S_{m}"] = tuple(a / b for a, b in zip(q[f"gain_{m}"], q[f"{m}_S"], strict=True))
        with np.errstate(divide="ignore", invalid="ignore"):
            q["gain_over_zero_skill_within"] = tuple(a / b for a, b in zip(q["gain_within"], q["within_zero_skill"],
                                                                           strict=True))
        return q


def summ(point: float, reps: np.ndarray) -> dict:
    e = dg.summary(point, reps) if np.isfinite(reps).sum() > 1 else {
        "point": float(point), "ci95": None, "boot_sd": None, "n_replicates_undefined": int((~np.isfinite(reps)).sum())}
    ok = np.isfinite(reps)
    if not np.isfinite(e["point"]):
        e["point"] = None
    if e["ci95"] is not None:
        e["upper_one_sided_95"] = float(np.quantile(reps[ok], 0.95))
        e["frac_replicates_ge_0"] = float((reps[ok] >= 0).mean())
    return e


def contrast(tot: Totals, a: str, b: str, key: str) -> dict:
    (pa, ra), (pb, rb) = tot.quantities(a)[key], tot.quantities(b)[key]
    e = summ(pa - pb, ra - rb)
    e.update({"a": a, "b": b, "quantity": key,
              "ci95_excludes_zero": bool(e["ci95"] is not None and (e["ci95"][0] > 0 or e["ci95"][1] < 0))})
    return e


def gains(tot: Totals, g: str, keys=("gain_within", "gain_site")) -> dict:
    q = tot.quantities(g)
    return {k: summ(*q[k]) for k in keys}


# ---------------------------------------------------------------- groups

def pt_kind(cation_states: dict) -> str | None:
    """s0 / s2 / mixed for the post-transition main-group cations of a host (None if it has none)."""
    from pymatgen.core import Element

    kinds = set()
    for el, s in cation_states.items():
        e = Element(el)
        if e.Z >= 31 and 13 <= e.group <= 18 and e.block == "p":
            kinds.add("s0" if s == e.group - 10 else "s2" if s == e.group - 12 else f"other:{el}{int(s):+d}")
    if not kinds:
        return None
    if any(k.startswith("other") for k in kinds):
        return "other"
    return "mixed" if kinds == {"s0", "s2"} else kinds.pop()


def load_dbins() -> dict:
    with open(ROOT / DBINS, newline="") as f:
        return {r["host_id"]: r["bin"] for r in csv.DictReader(f)}


def extra_groups(classes: dict, dbins: dict) -> dict:
    """Host predicates of the groups beyond the dshell_gain groups."""
    cls = lambda h: classes[h]["class"]
    pt = {h: (pt_kind(v["cation_states"]) if v["class"] == "main_group" and v["post_transition_d10_core"] else None)
          for h, v in classes.items()}
    dcat = lambda h: set(classes[h]["d_counts"])
    d10 = lambda h: cls(h) == "d10"
    return {
        "bins_d10": lambda h: dbins[h] == "d10", "bins_d0": lambda h: dbins[h] == "d0",
        "bins_other": lambda h: dbins[h] == "other", "bins_unassigned": lambda h: dbins[h] == "unassigned",
        "bins_d0_plus_other": lambda h: dbins[h] in ("d0", "other"),
        "pt_s0": lambda h: pt[h] == "s0", "pt_s2": lambda h: pt[h] == "s2", "pt_mixed": lambda h: pt[h] == "mixed",
        "d10_group12": lambda h: d10(h) and dcat(h) <= set(GROUP12),
        "d10_group11": lambda h: d10(h) and dcat(h) <= set(GROUP11),
        "d10_group11_and_12": lambda h: d10(h) and bool(dcat(h) & set(GROUP12)) and bool(dcat(h) & set(GROUP11)),
        "d10_U_treated": lambda h: d10(h) and bool(set(classes[h]["cation_states"]) & set(U_TREATED)),
        "d10_U_untreated": lambda h: d10(h) and not set(classes[h]["cation_states"]) & set(U_TREATED),
    }, pt


# ---------------------------------------------------------------- blocks

def host_cluster(X: list[np.ndarray], hosts: list[np.ndarray], names: list[str]) -> tuple[Totals, int]:
    """Bootstrap over distinct hosts; a host's row = sum of its rows over all 30 resamples."""
    allh = sorted({h for hs in hosts for h in hs})
    pos = {h: i for i, h in enumerate(allh)}
    T = np.zeros((len(allh), X[0].shape[1]))
    for x, hs in zip(X, hosts, strict=True):
        np.add.at(T, [pos[h] for h in hs], x)
    rng = np.random.default_rng(PAPER_SEED)
    n = len(allh)
    reps = np.concatenate([rng.multinomial(n, np.full(n, 1.0 / n), size=CHUNK) @ T for _ in range(BIG // CHUNK)])
    return Totals(T.sum(axis=0), reps, names), n


def class_row(tot: Totals, g: str) -> dict:
    q = tot.quantities(g)
    return {k: summ(*q[k]) for k in ("within_zero_skill", "within_S", "within_Dstar", "gain_within",
                                     "gain_over_S_within", "gain_over_zero_skill_within", "site_S", "site_Dstar",
                                     "gain_site", "gain_over_S_site")}


def small_flag(counts: dict) -> dict:
    n = counts["n_hosts_within_host_metric"]
    return {"n_hosts_within_host_metric": n, "too_small_to_interpret": bool(n < SMALL_N),
            "note": (f"fewer than {SMALL_N} hosts in the within-host metric: too small to interpret"
                     if n < SMALL_N else "")}


def run() -> dict:
    from dftgnn.stats.g1 import HierDraws

    t0 = time.time()
    classes = dg.load_classes()
    st, hosts, inputs = dg.headline_stats()
    ens = inputs.pop("ens")
    dbins = load_dbins()
    if set(dbins) != set(classes):
        raise AssertionError(f"{DBINS} hosts differ from the class record")
    masks = dg.group_masks(classes, hosts)
    preds, pt = extra_groups(classes, dbins)
    masks.update({g: [np.array([f(h) for h in hs]) for hs in hosts] for g, f in preds.items()})
    from dftgnn.stats.metrics import _COLS
    wn = [s[:, _COLS.index("wn")] for s in st["S"]]
    counts = {g: dg.group_counts(classes, masks[g], hosts, wn, ens) for g in masks}
    z = zero_skill_sums(ens, hosts)
    base = base_columns(st, z)
    X, names = design(base, masks)
    point = sum(x.sum(axis=0) for x in X)
    t_load = time.time() - t0

    # paper draws: HierDraws.make, checked against the chunked generator and the imported pooled estimator
    draws = HierDraws.make([N_TEST] * N_R, PAPER_DRAWS, seed=PAPER_SEED)
    chunk_eq = all(np.array_equal(p, draws.pick[i * CHUNK:(i + 1) * CHUNK])
                   and np.array_equal(w, draws.w[i * CHUNK:(i + 1) * CHUNK])
                   for i, (p, w) in enumerate(draw_chunks([N_TEST] * N_R, PAPER_DRAWS, PAPER_SEED)))
    if not chunk_eq:
        raise AssertionError("chunked draws differ from HierDraws.make")
    paper = Totals(point, accumulate(X, draws.pick, draws.w), names)
    pooled_diff = 0.0
    for g in ("d10", "rest", "all_hosts", "pt_s0", "d10_U_treated"):
        for m, col in (("S", "wsum"), ("Dstar", "wsum"), ("S", "sabs"), ("Dstar", "sabs")):
            den = "wn" if col == "wsum" else "n"
            p0, r0 = dg.pooled(st[m], masks[g], draws.pick, draws.w, _COLS.index(col), _COLS.index(den))
            p1, r1 = paper.ratio(g, f"{col}_{m}", den)
            pooled_diff = max(pooled_diff, abs(p0 - p1), float(np.abs(r0 - r1).max()))
    if pooled_diff > 1e-12:
        raise AssertionError(f"vectorised totals differ from dshell_gain.pooled by {pooled_diff}")

    # reproduction of the stored record
    stored = json.loads((ROOT / GAIN_REC).read_text())["payload"]
    head = dg.reproduce(st, draws)
    prim = contrast(paper, "d10", "rest", "gain_within")
    diffs = {"primary.point": abs(prim["point"] - stored["primary"]["point"]),
             "primary.ci95": max(abs(a - b) for a, b in zip(prim["ci95"], stored["primary"]["ci95"], strict=True)),
             "primary.boot_sd": abs(prim["boot_sd"] - stored["primary"]["boot_sd"])}
    grp_diff, n_grp = 0.0, 0
    for g, row in stored["groups"].items():
        for metric, key in (("within_host_mae", "gain_within"), ("mae", "gain_site")):
            sv = row["gains"]["S_minus_Dstar"][metric]
            mv = summ(*paper.quantities(g)[key])
            grp_diff = max(grp_diff, abs(mv["point"] - sv["point"]),
                           *(abs(a - b) for a, b in zip(mv["ci95"], sv["ci95"], strict=True)))
            n_grp += 3
        for m in ("S", "Dstar"):
            for metric, key in (("within_host_mae", f"within_{m}"), ("mae", f"site_{m}")):
                sv = row["models"][m][metric]
                mv = summ(*paper.quantities(g)[key])
                grp_diff = max(grp_diff, abs(mv["point"] - sv["point"]),
                               *(abs(a - b) for a, b in zip(mv["ci95"], sv["ci95"], strict=True)))
                n_grp += 3
        sc = row["counts"]
        if sc != counts[g]:
            raise AssertionError(f"group {g}: counts differ from the record")
    diffs["groups.S_Dstar_MAE_and_gains"] = grp_diff
    worst = max(diffs.values())
    if worst > TOL:
        raise AssertionError(f"stored dshell_gain values not reproduced: {diffs}")
    reproduction = {"source": f"{GAIN_REC} payload.primary (point, ci95, boot_sd) and payload.groups.*.(gains."
                              "S_minus_Dstar, models.S, models.Dstar).{within_host_mae, mae} (point, ci95), counts",
                    "max_abs_difference": worst, "by_item": diffs, "n_group_values_compared": n_grp,
                    "headline_dshell_reproduce_max_abs_difference": head["max_abs_difference"],
                    "pooled_vs_dshell_gain_pooled_max_abs_difference": pooled_diff,
                    "chunked_draws_equal_HierDraws_make": True, "reproduced_below_1e-9": True}
    t_paper = time.time() - t0 - t_load

    # block 1: Monte-Carlo stability (20000 draws, five seeds); seed 20261008 totals reused below
    mc, big = {}, {}
    all_reps = []
    for seed in MC_SEEDS:
        tot = Totals(point, hier_totals(X, BIG, seed), names)
        if seed == PAPER_SEED:
            big = tot
            if not np.array_equal(tot.reps[:PAPER_DRAWS], paper.reps):
                raise AssertionError("first 2000 of the 20000 draws differ from the paper draws")
        e = contrast(tot, "d10", "rest", "gain_within")
        all_reps.append(tot.quantities("d10")["gain_within"][1] - tot.quantities("rest")["gain_within"][1])
        mc[str(seed)] = {"within": {k: e[k] for k in ("point", "ci95", "upper_one_sided_95", "boot_sd",
                                                       "frac_replicates_ge_0")},
                         "site": {k: v for k, v in contrast(tot, "d10", "rest", "gain_site").items()
                                  if k in ("point", "ci95", "upper_one_sided_95", "boot_sd", "frac_replicates_ge_0")}}
    t_mc = time.time() - t0 - t_load - t_paper

    def across(metric: str) -> dict:
        lo = np.array([mc[str(s)][metric]["ci95"][0] for s in MC_SEEDS])
        hi = np.array([mc[str(s)][metric]["ci95"][1] for s in MC_SEEDS])
        u = np.array([mc[str(s)][metric]["upper_one_sided_95"] for s in MC_SEEDS])
        f = lambda a: {"mean": float(a.mean()), "sd_over_seeds": float(a.std(ddof=1)),
                       "se_of_mean": float(a.std(ddof=1) / np.sqrt(len(a))), "min": float(a.min()),
                       "max": float(a.max())}
        return {"ci95_lower": f(lo), "ci95_upper": f(hi), "upper_one_sided_95": f(u),
                "n_seeds_upper_end_below_0": int((hi < 0).sum())}

    pooled_reps = np.concatenate(all_reps)
    paper_prim_site = contrast(paper, "d10", "rest", "gain_site")
    mc_block = {
        "contrast": "gain_d10 - gain_rest, S - D*, rest = d0 + partial_d + main_group (published classes)",
        "n_boot_per_seed": BIG, "seeds": list(MC_SEEDS),
        "paper_draws_2000_seed_20261008": {"within": {k: prim[k] for k in ("point", "ci95", "upper_one_sided_95",
                                                                            "boot_sd", "frac_replicates_ge_0")},
                                           "site": {k: paper_prim_site[k] for k in ("point", "ci95",
                                                                                     "upper_one_sided_95", "boot_sd",
                                                                                     "frac_replicates_ge_0")}},
        "per_seed": mc, "across_seeds": {"within": across("within"), "site": across("site")},
        "within_frac_replicates_ge_0_all_5x20000": float((pooled_reps >= 0).mean()),
        "within_ci95_pooled_5x20000": [float(np.quantile(pooled_reps, 0.025)), float(np.quantile(pooled_reps, 0.975))],
        "counts": {g: counts[g] for g in ("d10", "rest")},
    }

    # block 2: host-cluster bootstrap
    hc, n_distinct = host_cluster(X, hosts, names)
    hc_block = {
        "scheme": "bootstrap over the distinct hosts tested in any of the 30 resamples; multinomial weights "
                  f"(n = {n_distinct}), {BIG} replicates, numpy default_rng({PAPER_SEED}) in chunks of {CHUNK}; a "
                  "drawn host carries the sum of its host statistics over all its test appearances; class value = "
                  "pooled ratio of weighted sums (unstratified: class sizes vary between replicates)",
        "n_distinct_hosts": n_distinct, "n_boot": BIG, "seed": PAPER_SEED,
        "primary_contrast": {"within": contrast(hc, "d10", "rest", "gain_within"),
                             "site": contrast(hc, "d10", "rest", "gain_site")},
        "d10": gains(hc, "d10"), "rest": gains(hc, "rest"),
        "counts": {g: counts[g] for g in ("d10", "rest")},
    }

    # block 3: filled-(n-1)d bins of host_dbins_v1.csv
    tested = sorted({h for hs in hosts for h in hs})
    within_hosts = sorted({h for hs, n in zip(hosts, wn, strict=True) for h in hs[n > 0]})
    pub = ("d0", "d10", "partial_d", "main_group", "mixed_f", "unassigned")
    old = ("d0", "d10", "other", "unassigned")

    def xtab(hs):
        return {o: {p: sum(1 for h in hs if dbins[h] == o and classes[h]["class"] == p) for p in pub} for o in old}

    xt_pt = {o: sum(1 for h in tested if dbins[h] == o and classes[h]["class"] == "main_group"
                    and classes[h]["post_transition_d10_core"]) for o in old}
    fd = {}
    for lab, tot in (("paper_draws_2000_seed_20261008", paper), ("draws_20000_seed_20261008", big)):
        fd[lab] = {"contrast_within": contrast(tot, "bins_d10", "bins_d0_plus_other", "gain_within"),
                   "contrast_site": contrast(tot, "bins_d10", "bins_d0_plus_other", "gain_site"),
                   **{g: gains(tot, g) for g in ("bins_d10", "bins_d0", "bins_other", "bins_d0_plus_other",
                                                  "bins_unassigned")}}
    fd_block = {
        "classes": f"{DBINS} column 'bin' (docs/deviations.md 2026-10-10): d10 = at least one cation with a filled "
                   "d shell (group-11/12 and filled-(n-1)d p-block cations: Zn2+, Cd2+, Ga3+, In3+, Sn4+, Pb2+, "
                   "Bi3+, Ag+ ...) and none partially filled; d0 = no cation with d electrons (s-block, early "
                   "p-block, f-block and d0 transition-metal cations); other = any cation with 1-9 d electrons; "
                   "unassigned = no oxidation state",
        "comparator": "bins_d0_plus_other = d0 + other of that file (all classified non-d10 hosts; the 9 unassigned "
                      "excluded)",
        **fd,
        "counts": {g: counts[g] for g in ("bins_d10", "bins_d0", "bins_other", "bins_d0_plus_other",
                                           "bins_unassigned")},
        "cross_tab_rows_dbins_cols_published": {"all_818_hosts": xtab(tested),
                                                "hosts_in_within_host_metric": xtab(within_hosts),
                                                "published_main_group_post_transition_by_dbins": xt_pt},
    }

    # block 4: relative gains per published class
    rel_groups = ("d0", "d10", "partial_d", "main_group", "mixed_f", "unassigned", "rest",
                  "main_group_post_transition_true", "main_group_post_transition_false", "all_hosts")
    rel_block = {
        "definitions": "zero-skill = within-host residual MAE of a predictor constant within each host = sum |y - "
                       "host mean y| / wn over hosts with >= 2 tested sites (model independent); gain = MAE_S - "
                       "MAE_D*; gain_over_S = gain / MAE_S; gain_over_zero_skill = gain / zero-skill; ratios per "
                       "replicate on the paper draws (2000, seed 20261008), point = ratio of point values",
        "classes": {g: {"counts": counts[g], **class_row(paper, g)} for g in rel_groups},
        "contrast_d10_minus_rest": {k: contrast(paper, "d10", "rest", k)
                                    for k in ("gain_within", "gain_over_S_within", "gain_over_zero_skill_within",
                                              "gain_site", "gain_over_S_site")},
        "contrast_post_transition_true_minus_false": {
            k: contrast(paper, "main_group_post_transition_true", "main_group_post_transition_false", k)
            for k in ("gain_within", "gain_over_S_within", "gain_over_zero_skill_within")},
    }

    # block 5: post-transition split
    pt_groups = ("pt_s0", "pt_s2", "pt_mixed", "main_group_post_transition_false", "main_group_post_transition_true")
    states = {}
    for h, k in pt.items():
        if k is not None:
            states.setdefault(k, set()).update(f"{el}{int(s):+d}" for el, s in classes[h]["cation_states"].items())
    pt_block = {
        "rule": "main-group hosts with post_transition_d10_core (published flag); per post-transition cation (p "
                "block, Z >= 31, groups 13-18) from cation_states of results/derived_dshell/dshell_classes.json: "
                "s0 if state = group - 10 (Ga3+, Ge4+, As5+, In3+, Sn4+, Sb5+, Te6+, Tl3+, Pb4+, Bi5+), s2 if "
                "state = group - 12 (Ge2+, As3+, In+, Sn2+, Sb3+, Te4+, Tl+, Pb2+, Bi3+); host = s0 / s2 if all its "
                "post-transition cations are that kind, mixed if both occur. light = main group without a "
                "post-transition cation",
        "n_hosts_other_state": sum(1 for k in pt.values() if k == "other"),
        "cation_states_present": {k: sorted(v) for k, v in states.items()},
        "groups": {g: {"counts": counts[g], "small_n": small_flag(counts[g]), **class_row(paper, g)}
                   for g in pt_groups},
        "contrast_s0_minus_s2": {k: contrast(paper, "pt_s0", "pt_s2", k)
                                 for k in ("gain_within", "gain_over_S_within", "gain_over_zero_skill_within",
                                           "gain_site")},
        "contrast_s0_minus_light": contrast(paper, "pt_s0", "main_group_post_transition_false", "gain_within"),
        "contrast_s2_minus_light": contrast(paper, "pt_s2", "main_group_post_transition_false", "gain_within"),
        "draws": "paper draws (2000, seed 20261008)",
    }

    # block 6: transition-metal d10 split
    d10_groups = ("d10", "d10_group12", "d10_group11", "d10_group11_and_12", "d10_U_treated", "d10_U_untreated")
    d10_block = {
        "rule": "within published d10: group12 = all d-block cations in Zn, Cd, Hg; group11 = all in Cu, Ag, Au "
                "(all d10, so Cu+, Ag+, Au+); group11_and_12 = both present; U_treated = any Zn or Cu cation "
                "(PBEsol+U with U_eff = 5 eV on Cu and Zn d only, docs/provenance.md row 4), U_untreated = the "
                "other d10 hosts",
        "elements": {g: sorted({el for hs, m in zip(hosts, masks[g], strict=True) for h in hs[m]
                                for el in classes[h]["d_counts"]}) for g in d10_groups},
        "groups": {g: {"counts": counts[g], "small_n": small_flag(counts[g]), **class_row(paper, g)}
                   for g in d10_groups},
        "contrast_group12_minus_group11": {k: contrast(paper, "d10_group12", "d10_group11", k)
                                           for k in ("gain_within", "gain_over_S_within", "gain_site")},
        "contrast_U_treated_minus_untreated": {k: contrast(paper, "d10_U_treated", "d10_U_untreated", k)
                                               for k in ("gain_within", "gain_over_S_within", "gain_site")},
        "draws": "paper draws (2000, seed 20261008)",
    }
    t_all = time.time() - t0

    return {
        "label": "TRV1 T3/M22/M24: robustness of the within-host descriptor gain S - D* by cation d-shell "
                 "configuration, 654 hosts, 30 x 5, pooled class estimator of analysis/dshell/dshell_gain.py",
        "sign_convention": "gain = MAE_S - MAE_D* (positive = D* more accurate); contrast = gain(a) - gain(b) per "
                           "replicate on shared draws; relative gains = gain / MAE_S or gain / zero-skill",
        "estimator": {
            "pooled": "replicate = sum over the 30 slots of the draw-weighted class numerator / sum of the "
                      "draw-weighted class denominator; point = same with unit weights; within-host MAE = wsum / wn "
                      "(hosts with >= 2 tested sites); site MAE = sabs / n",
            "paper_draws": {"n_boot": PAPER_DRAWS, "seed": PAPER_SEED, "n_resamples": N_R,
                            "n_test_hosts_per_resample": N_TEST, "budget": dg.B, "n_seeds": dg.N_S},
            "big_draws": f"HierDraws.make([164] * 30, {BIG}, seed) reproduced in chunks of {CHUNK} with the same "
                         "generator consumption; its first 2000 replicates equal the paper draws",
            "interval": "2.5 / 97.5 percentiles (numpy linear); upper_one_sided_95 (U) = 95th percentile",
            "small_n": f"groups with fewer than {SMALL_N} hosts in the within-host metric are flagged too small to "
                       "interpret (reported, not suppressed)"},
        "mc_stability": mc_block,
        "host_cluster": hc_block,
        "filled_d_bins": fd_block,
        "relative": rel_block,
        "post_transition_split": pt_block,
        "d10_split": d10_block,
        "reproduction": reproduction,
        "runtime_s": {"load": t_load, "paper_draws_and_reproduction": t_paper, "mc_20000x5": t_mc, "total": t_all},
        "inputs": {**inputs, "class_record": {"path": dg.CLASS_RECORD, "sha256": dg.CLASS_SHA256},
                   "dbins": {"path": DBINS, "sha256": c.sha256(ROOT / DBINS)},
                   "dshell_gain": {"path": GAIN_REC, "sha256": c.sha256(ROOT / GAIN_REC)}},
    }


def finite_check(x, path="payload"):
    if isinstance(x, dict):
        for k, v in x.items():
            finite_check(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            finite_check(v, f"{path}/{i}")
    elif isinstance(x, float) and not np.isfinite(x):
        raise AssertionError(f"non-finite value at {path}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="write the payload to --out instead of write_result")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    payload = c.to_jsonable(run())
    finite_check(payload)
    print(f"runtime_s {payload['runtime_s']['total']:.1f}", file=sys.stderr)
    if a.dry_run:
        text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if a.out is None:
            print(text, end="")
        else:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text, encoding="utf-8")
            print(a.out)
        return
    from dftgnn.config import load_config
    from dftgnn.io.results import write_result

    print(write_result("dshell_robust", payload, config=load_config(), repo_root=ROOT, results_dir=OUT_DIR))


if __name__ == "__main__":
    main()
