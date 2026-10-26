"""Re-derive the headline quantities of the study from the stored per-site prediction tables and run records.

Nothing is trained and nothing is written. The script calls the computation functions of the shipped analysis code
(analysis/dstar, analysis/dsens, analysis/coord/site_selection, analysis/dshell/dshell_gain, analysis/m3/staging_fair),
which read the per-site prediction tables under results/, check their sha256 against the run records, form the
seed-ensemble predictions and run the paired hierarchical bootstrap (2000 replicates, seed 20261008). Each value is
then compared with the stored derived record that holds it (tolerance 1e-9 on the point value and on both ends of
the 95% interval). The exit status is 1 if any value differs.

Models: S is the structure-only graph network; D* is the descriptor-augmented network (pristine-host electronic
descriptors of the database added to the structure) after the wider hyperparameter search; P is the staged network
that first predicts the descriptors from the structure and feeds them to the descriptor network.

Quantities that need the processed universe table (data/processed, not in this repository) are not covered here,
for example the Spearman and per-group within-host breakdowns and the geometry re-relaxation check.

    python reproduce/numbers.py
"""
from __future__ import annotations

import os
import sys

# This file is named like the standard-library module ``numbers``; drop its own folder from the import path so that
# numpy and the standard library import the real one.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import functools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TOL = 1e-9

# Code commits of the tagged code that produced the cluster run records (the record field ``git_sha``). The run
# records name the pinned code reference ``nrp-code-v1``; its later bug-fix tags are accepted in its place
# (dftgnn.train.CODE_REF_SUCCESSORS). This repository has no tag history, so the commits are listed here and the
# record check compares each record's ``git_sha`` with this list instead of resolving tags with git.
CODE_COMMITS = {
    "nrp-code-v1": "ffd86cedc3cc87a6fc842b04fda0d86d6b6a58af",
    "nrp-code-v2": "fa0e97064a329089218c39f3f5435e719de1cb84",
    "nrp-code-v4": "bfbdd8ef0c32f07f2a974c19477eba1430d5b04b",
    "nrp-code-v5": "20eff9054bf84fc867909d893b66bb368d16631e",
    "nrp-code-v6": "6d577a6f34a90f2cf61b011b9da91263b18e0e81",
    "nrp-code-v7": "8f03f102579daa88f6761b7e962dcaa4972f6aff",
    "nrp-code-v8": "053d9c6f3f70a561107dc0e8f1c22a4298bb5aed",
    "nrp-code-v9": "bc286bb96529835c3509407ba5c81c8e4e8d118b",
}


def accepted_commits(ref: str, root: Path | None = None) -> set[str]:
    """Stand-in for dftgnn.train.accepted_code_commits without git tags: the listed commit of ``ref`` and of its
    successor tags."""
    import dftgnn.train as T

    return {CODE_COMMITS[t] for t in (ref, *T.CODE_REF_SUCCESSORS.get(ref, ())) if t in CODE_COMMITS}


def stored(path: str, *keys):
    """A value of a stored record: results/<path>, then ``payload`` and the keys."""
    x = json.loads((ROOT / "results" / path).read_text())["payload"]
    for k in keys:
        x = x[k]
    return x


def memo_table(fn):
    """Cache a table loader (each parquet is read and hash-checked once); callers get a copy."""
    cache = {}

    @functools.wraps(fn)
    def load(name, *args, **kw):
        if name not in cache:
            cache[name] = fn(name, *args, **kw)
        df, info = cache[name]
        return df.copy(), dict(info)
    return load


def memo(fn):
    cache = {}

    @functools.wraps(fn)
    def call(*args):
        key = tuple(id(a) for a in args)
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]
    return call


def setup():
    """Import the analysis modules and share the heavy loads between them."""
    import dftgnn.train as T
    from analysis import a1_common as c
    from analysis import staging_analysis as sa
    from analysis.dstar import dstar_analysis as da

    T.accepted_code_commits = accepted_commits
    c.load_table = memo_table(c.load_table)
    sa.load_table = memo_table(sa.load_table)
    da.s_reference = memo(da.s_reference)
    da.load_dstar = memo(da.load_dstar)


def row(name, point, ci, ref_point, ref_ci, unit="eV"):
    got = [float(point), *map(float, ci)]
    want = [float(ref_point), *map(float, ref_ci)]
    diff = max(abs(a - b) for a, b in zip(got, want, strict=True))
    return {"quantity": name, "value": got[0], "ci95": got[1:], "stored": want[0], "diff": diff,
            "match": diff <= TOL, "unit": unit}


def headline_rows(rows: list, timings: dict) -> None:
    """MAE of S and D*, A* and the relative reduction at 654 hosts (30 resamples x 5 seeds); A* at 200 and 400
    hosts (10 resamples x 3 seeds)."""
    from analysis import a1_common as c
    from analysis import precision_analysis as pa
    from analysis.dsens import dsens as ds
    from analysis.dstar import dstar_analysis as da
    from dftgnn.stats.g1 import HierDraws

    t = time.time()
    ref, _ = da.s_reference()
    dstar, _ = da.load_dstar(ref)
    timings["load stored predictions of S, D and D*"] = time.time() - t

    t = time.time()
    head, meta = da.headline(dstar)
    bad = [k for k, v in meta["reproduction"].items() if not v["reproduced"]]
    if bad:
        raise AssertionError(f"headline cross-checks failed: {bad}")
    h = head["654"]
    rec = "derived_dstar/dstar_gaps.json"
    sh = stored(rec, "headline_654_30x5", "654")
    for m, label in (("S", "MAE of S, structure only, 654 hosts (30 resamples x 5 seeds)"),
                     ("Dstar", "MAE of D*, descriptor network with wider search, 654 hosts (30 x 5)")):
        rows.append(row(label, h["MAE"][m]["mean"], h["MAE"][m]["ci95"], sh["MAE"][m]["mean"], sh["MAE"][m]["ci95"]))
    g, gs = h["gaps"]["S_minus_Dstar"], sh["gaps"]["S_minus_Dstar"]
    rows.append(row("Descriptor advantage A* = MAE_S - MAE_D*, 654 hosts (30 x 5)", g["A"], g["ci95"], gs["A"],
                    gs["ci95"]))

    long = ds.headline_long(dstar)
    ens, _ = ds.drop_cells(long, [], ds.N_R, ds.N_S)
    draws = HierDraws.make([pa.N_HOSTS_PER_RESAMPLE] * ds.N_R, c.N_BOOT, seed=c.BOOT_SEED)
    rel = ds.relative(ds.headline_reps(ens, draws), "Dstar")
    sr = stored("derived_dsens/dsens.json", "relative_reduction", "rows", "654_30x5", "Dstar")
    rows.append(row("Relative reduction (MAE_S - MAE_D*) / MAE_S, 654 hosts (30 x 5)", rel["R"], rel["ci95"],
                    sr["R"], sr["ci95"], unit="fraction"))
    timings["654-host design: MAE, A*, relative reduction"] = time.time() - t

    t = time.time()
    sw, _ = ds.sweep_reps(dstar)
    lc = stored("derived_dsens/dsens.json", "learning_curve", "rows")
    for B in (200, 400):
        g = ds.block(sw[B])["gaps"]["S_minus_Dstar"]
        s = lc[f"{B}_10x3"]
        rows.append(row(f"A* = MAE_S - MAE_D*, {B} hosts (10 resamples x 3 seeds)", g["A"], g["ci95"], s["A"],
                        s["ci95"]))
    timings["200- and 400-host design: A*"] = time.time() - t


def selection_rows(rows: list, timings: dict) -> None:
    """Fraction of test hosts whose lowest-predicted oxygen site is a site of lowest DFT energy (hosts with at
    least two tested sites; 654 hosts, 30 x 5)."""
    from analysis.coord import site_selection as ss

    t = time.time()
    res = ss.run()
    if not res["reproduction"]["reproduced"]:
        raise AssertionError("site selection: within-host cross-check failed")
    got = res["results"]["hosts_ge2_sites"]
    ref = stored("derived_coord/site_selection.json", "results", "hosts_ge2_sites")
    for k, label in (("S", "Lowest-site selection fraction, S, 654 hosts (30 x 5)"),
                     ("Dstar", "Lowest-site selection fraction, D*, 654 hosts (30 x 5)"),
                     ("Dstar_minus_S", "Lowest-site selection, D* - S (paired), 654 hosts (30 x 5)")):
        rows.append(row(label, got[k]["mean"], got[k]["ci95"], ref[k]["mean"], ref[k]["ci95"], unit="fraction"))
    timings["lowest-site selection"] = time.time() - t


def dshell_rows(rows: list, timings: dict) -> None:
    """Within-host gain of D* over S on hosts whose d-block cations are d10, minus the same gain on the other
    classified hosts (d0, partial d, main group); within-host residual MAE, 654 hosts, 30 x 5."""
    from analysis.dshell import dshell_gain as dg

    t = time.time()
    res = dg.run()
    p, s = res["primary"], stored("derived_dshell/dshell_gain.json", "primary")
    rows.append(row("d10 within-host contrast: gain(d10) - gain(other classes), S - D*, 654 hosts", p["point"],
                    p["ci95"], s["point"], s["ci95"]))
    timings["d-shell within-host contrast"] = time.time() - t


def staging_rows(rows: list, timings: dict) -> None:
    """Staged network P against S at 654 hosts (10 resamples x 3 seeds; P exists only on that design)."""
    import pandas as pd
    from analysis import staging_analysis as sa
    from analysis.m3 import staging_fair as sf

    t = time.time()
    sweep, _ = sf.load_sweep()
    p1_cell, _ = sf.load_p1()
    pc, _ = sf.load_pcomp(sweep, p1_cell)
    old = sweep[sweep.model == "P"].assign(model="P_old")
    long = pd.concat([sweep[sweep.model.isin(["S", "D"])], old, pc], ignore_index=True)
    long = long[long.r >= 0]
    ens = sa.ensemble(long.drop(columns=["run_id"]))
    res = sf.block(ens, sa.make_draws(ens))
    sf.reproduce(res)
    ref = stored("derived_staging/staging_fair.json", "budgets", "654")
    g, s = res["654"]["P_minus_S"], ref["P_minus_S"]
    rows.append(row("Staged network P - S, 654 hosts (10 x 3)", g["mean"], g["ci95"], s["mean"], s["ci95"]))
    timings["staged network"] = time.time() - t


def fmt(x: float, unit: str) -> str:
    """Four decimals (eV or fraction); the comparison itself uses the full stored precision."""
    return f"{x:.4f}"


def main() -> int:
    t0 = time.time()
    setup()
    rows, timings = [], {}
    headline_rows(rows, timings)
    selection_rows(rows, timings)
    dshell_rows(rows, timings)
    staging_rows(rows, timings)

    head = ("quantity", "value", "95% interval", "stored value", "match")
    body = [(r["quantity"] + ("" if r["unit"] == "fraction" else " [eV]"), fmt(r["value"], r["unit"]),
             f"[{fmt(r['ci95'][0], r['unit'])}, {fmt(r['ci95'][1], r['unit'])}]", fmt(r["stored"], r["unit"]),
             ("yes" if r["match"] else "NO") + f" (max diff {r['diff']:.1e})") for r in rows]
    w = [max(len(x[i]) for x in [head, *body]) for i in range(len(head))]
    line = lambda x: " | ".join(s.ljust(w[i]) for i, s in enumerate(x)).rstrip()
    print(line(head))
    print("-|-".join("-" * n for n in w))
    for b in body:
        print(line(b))
    n_bad = sum(not r["match"] for r in rows)
    print(f"\n{len(rows)} quantities, {len(rows) - n_bad} match the stored records within {TOL:g} "
          f"(point value and both interval ends); runtime {time.time() - t0:.0f} s")
    for k, v in timings.items():
        print(f"  {k}: {v:.0f} s", file=sys.stderr)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
