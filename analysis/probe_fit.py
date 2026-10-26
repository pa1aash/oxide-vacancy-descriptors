"""Physics. The structure-only model S sees the pristine crystal with the vacancy site marked; D is given the DFT
electronic descriptors of the pristine host and site. The probe asks whether S's readout input vector (the
concatenation [vacancy-node embedding | pooled node embeddings | global state] that its head sees) encodes those
descriptors linearly, and whether that encoding grows with the number of training hosts.

Rules:, row of 2026-10-14, which restates the §10 rule and fixes the alpha grid, the shuffle scheme, the
bootstrap and the class combination before any v1 probe number exists. Embeddings: row of 2026-10-10 (§10).

Per (r, B) cell, from ``data/processed/embeddings_v1/r<r>_B<B>_<kind>.npz`` (allow_pickle=False):

* trained   ridge from the trained seed-0 S embedding to each of the 22 descriptors;
* shuffled  the same with the training rows' descriptor rows permuted among training hosts (seed 20261008);

Training rows = sites flagged train and val; test rows = sites flagged test. Alpha per descriptor by 5-fold
GroupKFold by host on the training rows. Test R^2 per descriptor and, for the site-electronic descriptors, the
within-host R^2 (target and prediction mean-removed per test host with >= 2 test sites).

Checks before fitting (any failure raises): 240 records, 60 cells x 4 kinds; record, sidecar and npz sha256
and shape agree; ids and hosts equal the universe; the four kinds of a cell share rows; per cell the train,
val and test hosts equal ``dftgnn.embeddings.site_split``, the test hosts equal ``splits/outer_r<r>.json`` and
the S sites of ``results/analysis/eval_table.parquet``, the row counts equal the seed-0 S run record and no
test host is in training; every trained record points at that seed-0 S run.

Sanity checks (Task 3, descriptive): for r = 0 and B = 25, 200, 654, (a) a probe from the trained embedding to the
model's own seed-0 test prediction (5-fold grouped CV on the test rows, out-of-fold R^2); (b) the checkpoint head
replayed on the stored embedding against the stored prediction.

Writes ``results/probe_fit_v1.json`` via write_result.

    PYTHONPATH=$PWD/src python analysis/probe_fit.py [--data-root DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (25, 50, 100, 200, 400, 654)
N_RESAMPLES = 10
KINDS = ("trained", "init0", "init1", "init2")
PROBES = ("trained", "shuffled", "init0", "init1", "init2")
ALPHAS = 10.0 ** np.arange(-3.0, 6.0 + 1e-9, 0.5)      # 19 values, row of 2026-10-14 item b
FOLDS = 5
SHUFFLE_SEED = 20261008
SD_FLOOR = 1e-12
SST_FLOOR = 1e-12
CLASSES = {"host": "host-electronic DFT", "site": "site-electronic DFT"}
CLASS_SIZES = {"host": 12, "site": 10}
SANITY_CELLS = ((0, 25), (0, 200), (0, 654))
LABEL = "registered analysis (originally registered backbone v1)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- probe mathematics

def standardise(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Columns standardised with training-row mean and population SD (SD below 1e-12 replaced by 1)."""
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < SD_FLOOR, 1.0, sd)
    return [(a - mu) / sd for a in (train, *others)]


def ridge_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """Ridge predictions for every alpha and target, shape (n_alpha, n_eval, n_target).

    Closed form through the SVD of the centred training design; intercept unpenalised (centred data)."""
    xm, ym = x_tr.mean(0), y_tr.mean(0)
    u, s, vt = np.linalg.svd(x_tr - xm, full_matrices=False)
    uty = u.T @ (y_tr - ym)                                  # (k, t)
    xv = (x_ev - xm) @ vt.T                                  # (n_eval, k)
    out = np.empty((len(alphas), len(x_ev), y_tr.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = xv @ ((s / (s ** 2 + a))[:, None] * uty) + ym
    return out


def group_folds(groups: np.ndarray, folds: int = FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    idx = np.arange(len(groups))
    return list(GroupKFold(n_splits=folds).split(idx, groups=groups))


def cv_sse(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float] = ALPHAS,
           folds: int = FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """(summed squared error over held-out folds (n_alpha, n_target), out-of-fold predictions (n_alpha, n, t))."""
    sse = np.zeros((len(alphas), y.shape[1]))
    oof = np.zeros((len(alphas), *y.shape))
    for tr, va in group_folds(groups, folds):
        p = ridge_predict(x[tr], y[tr], x[va], alphas)
        oof[:, va] = p
        sse += ((p - y[va][None]) ** 2).sum(1)
    return sse, oof


def choose_alpha(sse: np.ndarray) -> np.ndarray:
    """Index of the alpha with the lowest summed CV error per target; ties to the smaller alpha (alphas ascend)."""
    return sse.argmin(0)


def fit_probe(x_tr, y_tr, groups_tr, x_te, alphas: Sequence[float] = ALPHAS, folds: int = FOLDS) -> dict:
    """Standardise, choose alpha per target by grouped CV on the training rows, refit, predict the test rows."""
    xt, xe = standardise(x_tr, x_te)
    sse, _ = cv_sse(xt, y_tr, groups_tr, alphas, folds)
    best = choose_alpha(sse)
    full = ridge_predict(xt, y_tr, xe, alphas)
    pred = full[best, :, np.arange(y_tr.shape[1])].T          # (n_test, t)
    return {"pred": pred, "alpha_index": best, "alpha": np.asarray(alphas)[best]}


def r2_scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Test R^2 per column, SS_tot about the test mean; NaN where SS_tot is zero."""
    y, p = np.atleast_2d(y.T).T, np.atleast_2d(p.T).T
    sst = ((y - y.mean(0)) ** 2).sum(0)
    ssr = ((y - p) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sst > SST_FLOOR, 1.0 - ssr / np.where(sst > SST_FLOOR, sst, 1.0), np.nan)


def host_demean(a: np.ndarray, hosts: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    for h in np.unique(hosts):
        m = hosts == h
        out[m] -= out[m].mean(0)
    return out


def within_host_r2(y: np.ndarray, p: np.ndarray, hosts: np.ndarray) -> tuple[np.ndarray, int, int]:
    """R^2 of host-mean-removed prediction against host-mean-removed target over hosts with >= 2 sites.

    Returns (R^2 per column, hosts used, sites used)."""
    hosts = np.asarray(hosts)
    hs, counts = np.unique(hosts, return_counts=True)
    keep = np.isin(hosts, hs[counts >= 2])
    if not keep.any():
        return np.full(np.atleast_2d(y.T).T.shape[1], np.nan), 0, 0
    yc = host_demean(np.atleast_2d(y.T).T[keep], hosts[keep])
    pc = host_demean(np.atleast_2d(p.T).T[keep], hosts[keep])
    sst = (yc ** 2).sum(0)
    ssr = ((yc - pc) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > SST_FLOOR, 1.0 - ssr / np.where(sst > SST_FLOOR, sst, 1.0), np.nan)
    return r2, int((counts >= 2).sum()), int(keep.sum())


def shuffle_by_host(y: np.ndarray, hosts: np.ndarray, site_ids: np.ndarray, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Descriptor rows permuted among hosts: sorted hosts permuted by ``default_rng(seed)``; each host takes the rows
    of its image host, sites in site_id order, cycled when the site counts differ."""
    rng = np.random.default_rng(seed)
    hosts, site_ids = np.asarray(hosts), np.asarray(site_ids)
    uh = np.unique(hosts)                                    # sorted
    image = dict(zip(uh, rng.permutation(uh), strict=True))
    order = {h: np.nonzero(hosts == h)[0][np.argsort(site_ids[hosts == h], kind="stable")] for h in uh}
    out = np.empty_like(y)
    for h in uh:
        dst, src = order[h], order[image[h]]
        out[dst] = y[src[np.arange(len(dst)) % len(src)]]
    return out


def class_means(values: np.ndarray, classes: Sequence[str]) -> dict:
    """Unweighted mean over the defined (non-NaN) values of each class, plus the count left out."""
    v, c = np.asarray(values, float), np.asarray(classes)
    out = {}
    for k in ("host", "site"):
        m = c == k
        if not m.any():
            continue
        ok = m & np.isfinite(v)
        out[k] = float(v[ok].mean()) if ok.any() else None
        out[f"n_undefined_{k}"] = int((m & ~np.isfinite(v)).sum())
    ok = np.isfinite(v)
    out["all"] = float(v[ok].mean()) if ok.any() else None
    return out


def _nan_list(a) -> list:
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, float)]


def probe_cell(emb: Mapping[str, np.ndarray], y: np.ndarray, split: np.ndarray, hosts: np.ndarray,
               site_ids: np.ndarray, classes: Sequence[str], alphas: Sequence[float] = ALPHAS) -> dict:
    """All five probes of one cell. ``emb`` maps kind (trained, init0, ...) to the embedding matrix; ``y`` is the raw
    descriptor matrix in the same row order."""
    tr = np.isin(split, ("train", "val"))
    te = split == "test"
    assert tr.any() and te.any()
    y_tr, y_te = standardise(y[tr], y[te])
    site_cols = np.asarray(classes) == "site"
    out = {}
    for probe in PROBES:
        kind = "trained" if probe == "shuffled" else probe
        if kind not in emb:
            continue
        target = shuffle_by_host(y_tr, hosts[tr], site_ids[tr]) if probe == "shuffled" else y_tr
        f = fit_probe(emb[kind][tr].astype(np.float64), target, hosts[tr], emb[kind][te].astype(np.float64), alphas)
        r2 = r2_scores(y_te, f["pred"])
        w, nh, ns = within_host_r2(y_te[:, site_cols], f["pred"][:, site_cols], hosts[te])
        out[probe] = {"r2": _nan_list(r2), "alpha": [float(a) for a in f["alpha"]],
                      "alpha_at_upper_edge": int((f["alpha_index"] == len(alphas) - 1).sum()),
                      "alpha_at_lower_edge": int((f["alpha_index"] == 0).sum()),
                      "class_mean": class_means(r2, classes),
                      "within_r2_site": _nan_list(w), "within_class_mean_site": class_means(w, ["site"] * len(w))["site"],
                      "within_n_hosts": nh, "within_n_sites": ns}
    return out


# ---------------------------------------------------------------- inputs and checks

def descriptor_table(path: Path) -> tuple[list[str], list[str]]:
    """(descriptor names, class keys) of the 22 D descriptors in taxonomy order, host-electronic first."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    names, cls = [], []
    for key in ("host", "site"):
        sel = [r["name"] for r in rows if r["class"] == CLASSES[key]]
        if len(sel) != CLASS_SIZES[key]:
            raise AssertionError(f"taxonomy has {len(sel)} {CLASSES[key]} descriptors, expected {CLASS_SIZES[key]}")
        names += sel
        cls += [key] * len(sel)
    return names, cls


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def embed_records(results: Path) -> dict:
    """(r, B, kind) -> payload of the embed records of ``results/embeddings_v1``."""
    out = {}
    for f in sorted(results.glob("*.json")):
        p = json.loads(f.read_text())["payload"]
        key = (int(p["r"]), int(p["budget"]), p["kind"])
        if key in out:
            raise AssertionError(f"duplicate embed record {key}")
        out[key] = p
    return out


def s_seed0_runs(results: Path) -> dict:
    """(r, B) -> payload of the v1 sweep S run of seed 0 (records directly under results/, 200-epoch cap)."""
    out = {}
    for f in sorted(results.glob("*.json")):
        try:
            p = json.loads(f.read_text()).get("payload")
        except ValueError:
            continue
        if not isinstance(p, dict) or "spec" not in p or p.get("smoke"):
            continue
        sp = p["spec"]
        if (sp.get("model") != "S" or sp.get("results_subdir") is not None or sp.get("tags", {}).get("pilot")
                or not str(sp.get("split", "")).startswith("outer_r") or int(sp.get("seed", -1)) != 0
                or int(p.get("max_epochs") or 0) != 200):
            continue
        r, b = int(sp["r"]), int(sp["budget"])
        if r < N_RESAMPLES and b in BUDGETS:
            if (r, b) in out:
                raise AssertionError(f"two seed-0 S runs for {(r, b)}")
            out[(r, b)] = p
    return out


def check_cell(r: int, b: int, recs: Mapping[str, dict], arrays: Mapping[str, dict], emb_dir: Path,
               universe: pd.DataFrame, s_run: dict, eval_sites: set, split_test: set, groups: Mapping) -> dict:
    """Integrity checks of one cell; raises on any mismatch, returns the counts."""
    ref = arrays["trained"]
    for kind in KINDS:
        rec, a = recs[kind], arrays[kind]
        side = json.loads((emb_dir / f"r{r}_B{b}_{kind}.json").read_text())
        npz = emb_dir / f"r{r}_B{b}_{kind}.npz"
        got = sha256(npz)
        if not (got == rec["sha256"] == side["sha256"]):
            raise AssertionError(f"{npz.name}: sha256 npz {got}, record {rec['sha256']}, sidecar {side['sha256']}")
        art = {x["path"]: x["sha256"] for x in rec["artifacts"]}
        if art.get(f"data/processed/embeddings_v1/{npz.name}") != got:
            raise AssertionError(f"{npz.name}: artifact sha256 differs from the npz")
        if art.get(f"data/processed/embeddings_v1/{npz.stem}.json") != sha256(npz.with_suffix(".json")):
            raise AssertionError(f"{npz.name}: sidecar sha256 differs from the record artifact")
        shape = list(a["embeddings"].shape)
        if not (shape == rec["shape"] == side["shape"]):
            raise AssertionError(f"{npz.name}: shape {shape}, record {rec['shape']}, sidecar {side['shape']}")
        nby = {s: int((a["split"] == s).sum()) for s in ("train", "val", "test")}
        if not (nby == rec["n_by_split"] == side["n_by_split"]):
            raise AssertionError(f"{npz.name}: split counts {nby} differ from record or sidecar")
        if not (a["block_bounds"].tolist() == rec["block_bounds"] == side["block_bounds"]):
            raise AssertionError(f"{npz.name}: block bounds differ")
        if a["block_bounds"][0] != 0 or a["block_bounds"][-1] != shape[1]:
            raise AssertionError(f"{npz.name}: block bounds do not span the embedding")
        if rec["checkpoint_run_id"] != s_run["run_id"]:
            raise AssertionError(f"{npz.name}: checkpoint {rec['checkpoint_run_id']} is not the seed-0 S run "
                                 f"{s_run['run_id']}")
        if not np.isfinite(a["embeddings"]).all():
            raise AssertionError(f"{npz.name}: non-finite embeddings")
        for k in ("site_ids", "host_ids", "split"):
            if not np.array_equal(a[k], ref[k]):
                raise AssertionError(f"{npz.name}: {k} differs from the trained file of the cell")
        if kind == "trained" and set(a["embeddings"].shape[1:]) != {s_run_width(s_run)}:
            raise AssertionError(f"{npz.name}: width {shape[1]} differs from the checkpoint's readout width")
    sid, hid, spl = ref["site_ids"], ref["host_ids"], ref["split"]
    if len(set(sid)) != len(sid):
        raise AssertionError(f"r{r} B{b}: duplicate site ids")
    uh = universe.loc[sid, "host_id"].to_numpy()          # KeyError if a site is not in the universe
    if not np.array_equal(uh, hid):
        raise AssertionError(f"r{r} B{b}: host ids differ from the universe")
    hosts = {s: set(hid[spl == s]) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        if hosts[s] != set(groups[s]):
            raise AssertionError(f"r{r} B{b}: {s} hosts differ from the registered split")
    if hosts["test"] != split_test:
        raise AssertionError(f"r{r} B{b}: test hosts differ from splits/outer_r{r}.json")
    if (hosts["train"] | hosts["val"]) & hosts["test"]:
        raise AssertionError(f"r{r} B{b}: a test host is in training")
    all_sites = set(universe.index[universe.host_id.isin(hosts["train"] | hosts["val"] | hosts["test"])])
    if set(sid) != all_sites:
        raise AssertionError(f"r{r} B{b}: rows are not every site of the cell's hosts")
    if set(sid[spl == "test"]) != eval_sites:
        raise AssertionError(f"r{r} B{b}: test sites differ from the eval table")
    nby = {s: int((spl == s).sum()) for s in ("train", "val", "test")}
    if nby != {k: int(v) for k, v in s_run["n_sites"].items()}:
        raise AssertionError(f"r{r} B{b}: site counts {nby} differ from the S run record {s_run['n_sites']}")
    nh = {s: len(hosts[s]) for s in hosts}
    if nh != {k: int(v) for k, v in s_run["n_hosts"].items()}:
        raise AssertionError(f"r{r} B{b}: host counts {nh} differ from the S run record {s_run['n_hosts']}")
    if nh["train"] + nh["val"] != b:
        raise AssertionError(f"r{r} B{b}: {nh['train'] + nh['val']} training hosts")
    return {"rows": len(sid), "n_sites": nby, "n_hosts": nh, "width": int(ref["embeddings"].shape[1])}


def s_run_width(s_run: dict) -> int:
    hp = s_run["spec"]["hp"]
    h2 = int(hp["hidden_width"]) // 2
    return h2 + (2 * h2 if hp["pooling"] == "set2set" else h2) + h2


# ---------------------------------------------------------------- sanity checks (Task 3 iii)

def sanity_cell(r: int, b: int, arrays: dict, s_run: dict, data_root: Path) -> dict:
    """Probe to the model's own test prediction (grouped 5-fold CV on the test rows) and head replay."""
    import torch

    from dftgnn.train import load_checkpoint

    a = arrays["trained"]
    te = a["split"] == "test"
    pred_path = ROOT / s_run["predictions"]["path"]
    if sha256(pred_path) != s_run["predictions"]["sha256"]:
        raise AssertionError(f"{pred_path}: sha256 differs from the run record")
    pr = pd.read_parquet(pred_path).set_index("site_id")
    sid = a["site_ids"][te]
    y = pr.loc[sid, "y_pred"].to_numpy()[:, None]
    hosts = a["host_ids"][te]
    x = standardise(a["embeddings"][te].astype(np.float64))[0]
    sse, oof = cv_sse(x, y, hosts)
    k = int(choose_alpha(sse)[0])
    r2_oof = float(r2_scores(y, oof[k])[0])
    ck_path = data_root / s_run["checkpoint"]["path"]
    if sha256(ck_path) != s_run["checkpoint"]["sha256"]:
        raise AssertionError(f"{ck_path}: sha256 differs from the run record")
    net, ck = load_checkpoint(ck_path, torch.device("cpu"))
    net.eval()
    with torch.no_grad():
        z = net.head(torch.from_numpy(a["embeddings"][te].astype(np.float32))).view(-1).double()
    replay = (z * ck["target_sd"].double().view(-1) + ck["target_mean"].double().view(-1)).numpy()
    diff = np.abs(replay - y[:, 0])
    return {"r": r, "B": b, "run_id": s_run["run_id"], "n_test_sites": int(te.sum()),
            "probe_to_own_prediction": {"alpha": float(ALPHAS[k]), "r2_out_of_fold": r2_oof,
                                        "passes_0p99": bool(r2_oof > 0.99)},
            "head_replay": {"max_abs_diff_eV": float(diff.max()), "mean_abs_diff_eV": float(diff.mean())}}


# ---------------------------------------------------------------- main

def main() -> None:
    from dftgnn.config import load_config
    from dftgnn.embeddings import site_split
    from dftgnn.io.results import write_result
    from dftgnn.split import load_split

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="checkout holding the gitignored data/processed and checkpoints (read only)")
    a = ap.parse_args()
    cfg = load_config()
    emb_dir = a.data_root / "data" / "processed" / "embeddings_v1"
    uni_path = a.data_root / "data" / "processed" / "universe_v1.parquet"
    tax_path = ROOT / "docs" / "descriptor_taxonomy.csv"
    eval_path = ROOT / "results" / "analysis" / "eval_table.parquet"
    names, classes = descriptor_table(tax_path)
    cols = [f"desc_{n}" for n in names]
    universe = pd.read_parquet(uni_path).set_index("site_id")
    if universe.index.duplicated().any() or universe[cols].isna().any().any():
        raise AssertionError("universe: duplicate site ids or missing descriptors")
    ev = pd.read_parquet(eval_path)
    ev = ev[ev.model == "S"]
    recs = embed_records(ROOT / "results" / "embeddings_v1")
    want = {(r, b, k) for r in range(N_RESAMPLES) for b in BUDGETS for k in KINDS}
    if set(recs) != want or len(recs) != 240:
        raise AssertionError(f"embed records: {len(recs)}, missing {sorted(want - set(recs))[:5]}, "
                             f"extra {sorted(set(recs) - want)[:5]}")
    for (r, b, k), p in recs.items():
        if p.get("registered_control") is not (k == "init0") and k != "trained":
            raise AssertionError(f"record {(r, b, k)}: registered_control flag {p.get('registered_control')}")
    s_runs = s_seed0_runs(ROOT / "results")
    if set(s_runs) != {(r, b) for r in range(N_RESAMPLES) for b in BUDGETS}:
        raise AssertionError(f"seed-0 S runs: {len(s_runs)} cells")
    n_npz = len(list(emb_dir.glob("*.npz")))
    if n_npz != 240 or len(list(emb_dir.glob("*.json"))) != 240:
        raise AssertionError(f"{emb_dir}: {n_npz} npz files")

    cells, checks, sanity = [], [], []
    for r in range(N_RESAMPLES):
        split_test = set(load_split(f"outer_r{r}")["test"])
        for b in BUDGETS:
            arrays = {k: load_npz(emb_dir / f"r{r}_B{b}_{k}.npz") for k in KINDS}
            eval_sites = set(ev.loc[(ev.r == r) & (ev.B == b), "site_id"])
            c = check_cell(r, b, {k: recs[(r, b, k)] for k in KINDS}, arrays, emb_dir, universe, s_runs[(r, b)],
                           eval_sites, split_test, site_split(r, b))
            checks.append({"r": r, "B": b, **c})
            ref = arrays["trained"]
            y = universe.loc[ref["site_ids"], cols].to_numpy(dtype=np.float64)
            probes = probe_cell({k: arrays[k]["embeddings"] for k in KINDS}, y, ref["split"], ref["host_ids"],
                                ref["site_ids"], classes)
            cells.append({"r": r, "B": b, "s_run_id": s_runs[(r, b)]["run_id"],
                          "embed_run_ids": {k: recs[(r, b, k)]["run_id"] for k in KINDS},
                          "n_sites": c["n_sites"], "n_hosts": c["n_hosts"], "width": c["width"], "probes": probes})
            if (r, b) in SANITY_CELLS:
                sanity.append(sanity_cell(r, b, arrays, s_runs[(r, b)], a.data_root))
            print(f"r{r} B{b}: rows {c['rows']}, trained host {probes['trained']['class_mean']['host']:.3f} "
                  f"site {probes['trained']['class_mean']['site']:.3f}", flush=True)

    payload = {
        "label": LABEL,
        "rule_row": "docs/deviations.md 2026-10-14 (§10, step A4)",
        "descriptors": names, "classes": classes, "alphas": ALPHAS.tolist(), "folds": FOLDS,
        "shuffle_seed": SHUFFLE_SEED, "probes": list(PROBES),
        "probe_labels": {"trained": "trained seed-0 S", "shuffled": "control (registered): shuffled labels",
                         "init0": "control (registered): random-init encoder, seed 0",
                         "init1": "DESCRIPTIVE spread of the random-init control",
                         "init2": "DESCRIPTIVE spread of the random-init control"},
        "inputs": {"universe": {"path": "data/processed/universe_v1.parquet", "sha256": sha256(uni_path)},
                   "taxonomy": {"path": "docs/descriptor_taxonomy.csv", "sha256": sha256(tax_path)},
                   "eval_table": {"path": "results/analysis/eval_table.parquet", "sha256": sha256(eval_path)},
                   "embeddings": {"dir": "data/processed/embeddings_v1", "n_npz": n_npz,
                                  "records": "results/embeddings_v1", "n_records": len(recs)}},
        "checks": {"n_records": len(recs), "n_cells": len(cells), "all_passed": True,
                   "list": ["record, sidecar, artifact and npz sha256 agree", "record, sidecar and npz shapes agree",
                            "split counts and block bounds agree", "trained records point at the seed-0 S run",
                            "site and host ids equal the universe", "the four kinds of a cell share rows",
                            "train, val and test hosts equal dftgnn.embeddings.site_split",
                            "test hosts equal splits/outer_r<r>.json", "test sites equal the eval table S sites",
                            "rows are every site of the cell's hosts", "site and host counts equal the S run record",
                            "no test host in training", "embeddings finite"],
                   "cells": checks},
        "sanity": sanity,
        "cells": cells,
    }
    print(write_result("probe_fit_v1", payload, config=cfg, repo_root=ROOT))


if __name__ == "__main__":
    main()
