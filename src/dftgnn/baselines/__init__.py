"""Non-GNN baselines on the tracked splits: shared data access, run loops and result payloads.

All splits come from ``splits/`` (``dftgnn.split.load_split``); nothing here generates a split.
A *fit function* has the signature ``fit(train_rows, test_rows, seed) -> {"pred": array, "info": dict}``
where the rows are positional indices into the universe table.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dftgnn.config import Config, load_config
from dftgnn.data import universe as U
from dftgnn.split import budget_train, derived_seed, load_split
from dftgnn.stats import metrics as M

REPO_ROOT = U.REPO_ROOT
PRED_DIR = REPO_ROOT / "results" / "predictions"
TRACK_LIMIT_BYTES = 2_000_000


@dataclass
class Data:
    uni: pd.DataFrame
    desc_class: dict[str, str]

    @classmethod
    def load(cls, cfg: Config | None = None) -> Data:
        cfg = cfg if cfg is not None else load_config()
        uni = U.read_universe(cfg)
        return cls(uni, dict(uni.attrs["descriptor_class"]))

    def features(self, classes: list[str] | None = None) -> list[str]:
        cols = sorted(c for c in self.uni.columns if c.startswith("desc_"))
        return cols if classes is None else [c for c in cols if self.desc_class[c] in classes]

    def rows(self, host_ids) -> np.ndarray:
        """Positional indices of every site of ``host_ids`` (universe order)."""
        return np.nonzero(self.uni.host_id.isin(set(host_ids)).to_numpy())[0]

    def masked(self, keep: np.ndarray) -> Data:
        out = Data(self.uni.loc[keep].reset_index(drop=True), self.desc_class)
        out.uni.attrs = self.uni.attrs
        return out


def feature_sets(cfg: Config, data: Data) -> dict[str, list[str]]:
    """Descriptor sets: all 70 (RF-Kumagai), Model D's classes (cfg), and structural/compositional."""
    return {
        "kumagai": data.features(),
        "electronic": data.features(list(cfg.models.D.descriptor_classes)),
        "structural": data.features(["structural/compositional"]),
    }


def _pred_frame(data: Data, rows: np.ndarray, pred: np.ndarray, **tags) -> pd.DataFrame:
    u = data.uni.iloc[rows]
    return pd.DataFrame({**tags, "site_id": u.site_id.to_numpy(), "host_id": u.host_id.to_numpy(),
                         "y_true": u.target_Ef_eV.to_numpy(), "y_pred": np.asarray(pred, float)})


def run_split(data: Data, fit, train_hosts, test_hosts, seed: int, **tags):
    """Fit on ``train_hosts``, predict ``test_hosts``; returns (prediction frame, info)."""
    tr, te = data.rows(train_hosts), data.rows(test_hosts)
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("empty train or test rows")
    out = fit(tr, te, seed)
    frame = _pred_frame(data, te, out["pred"], **tags)
    info = {**out["info"], "n_train_hosts": len(set(train_hosts)), "n_train_sites": len(tr),
            "n_test_hosts": len(set(test_hosts)), "n_test_sites": len(te)}
    return frame, info


def run_outer(data: Data, fit, cfg: Config, *, seed_tag: str, budgets: list[int] | None = None,
              resamples: range | None = None, log=print) -> tuple[pd.DataFrame, list[dict]]:
    """All outer resamples x budgets. Returns (predictions, per-run info records)."""
    budgets = budgets if budgets is not None else cfg.budgets.hosts
    frames, infos = [], []
    for r in resamples if resamples is not None else range(cfg.split.n_outer_resamples):
        sp = load_split(f"outer_r{r}")
        for b in budgets:
            train = budget_train(sp, b)
            seed = derived_seed(r, f"{seed_tag}_{b}") % 2**31
            frame, info = run_split(data, fit, train, sp["test"], seed, resample=r, budget=len(train))
            frames.append(frame)
            infos.append({"resample": r, "budget_hosts": len(train), "seed": seed, **info})
            log(f"{seed_tag} r={r} B={len(train)} sites={info['n_train_sites']}")
    return pd.concat(frames, ignore_index=True), infos


def run_kiyohara(data: Data, fit, *, seed_tag: str) -> tuple[pd.DataFrame, dict]:
    """Train on Kiyohara's train hosts only (no validation set needed), test on their test hosts."""
    k = load_split("kiyohara")
    seed = derived_seed(0, f"{seed_tag}_kiyohara") % 2**31
    frame, info = run_split(data, fit, k["train"], k["test"], seed, resample=-1, budget=len(k["train"]))
    return frame, {"seed": seed, **info}


def write_predictions(name: str, frame: pd.DataFrame) -> dict:
    """Write results/predictions/<name>.parquet; report sha256, size and whether it is tracked."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    size = path.stat().st_size
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": size, "rows": len(frame),
            "tracked": size < TRACK_LIMIT_BYTES,
            "tracking_rule": f"tracked only if < {TRACK_LIMIT_BYTES} bytes, else gitignored with the hash recorded"}


def host_stat_table(frame: pd.DataFrame) -> pd.DataFrame:
    return M.host_stats(frame.y_true, frame.y_pred, frame.host_id)


def metrics_block(frame: pd.DataFrame, *, n_boot: int, seed: int) -> dict:
    """Point metrics and test-host cluster CIs for one prediction frame."""
    return M.cluster_bootstrap_ci(frame.y_true, frame.y_pred, frame.host_id, n_boot=n_boot, seed=seed)


def summarize_outer(frame: pd.DataFrame, infos: list[dict], cfg: Config) -> dict:
    """Per budget: per-resample metrics (with cluster CIs) and the hierarchical aggregate."""
    n_boot = cfg.stats.cluster_bootstrap_n
    out = {}
    for b, fb in frame.groupby("budget"):
        runs, stats = [], []
        for r, fr in fb.groupby("resample"):
            info = next(i for i in infos if i["resample"] == r and i["budget_hosts"] == b)
            runs.append({**info, "metrics": metrics_block(fr, n_boot=n_boot, seed=cfg.split.seeds[0])})
            stats.append(host_stat_table(fr))
        out[str(b)] = {"aggregate": M.aggregate_resamples(stats, n_boot=n_boot, ci=cfg.stats.ci,
                                                          seed=cfg.split.seeds[0]),
                       "runs": runs}
    return out
