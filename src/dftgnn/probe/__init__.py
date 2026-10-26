"""Linear probe of the structure-only model's readout vector.

For one S checkpoint: the readout input vector [vacancy node | pooled nodes | global state] (``readout_vector``)
is extracted for the run's training sites and test sites. Features are standardised with training-site
statistics. For each of the 22 D descriptors (standardised with training-site statistics), a ridge regression
is fitted with its strength chosen from ``ALPHAS`` by 5-fold ``GroupKFold`` over training hosts (lowest mean
squared error), refitted on all training sites and scored by test R^2. Controls: (a) the same probe on training
labels shuffled across training hosts (a host's sites keep one shuffled host's labels, so the host structure of
the labels is kept), scored on the true test labels; (b) the probe on the readout vector of an untrained network
with the same architecture and seed.
"""
from __future__ import annotations

import numpy as np
import torch

ALPHAS = np.logspace(-3, 4, 15)


def features(net, data, positions, device, batch_size: int = 32) -> np.ndarray:
    from dftgnn.graphs import collate_sites

    net = net.to(device).eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(positions), batch_size):
            b = collate_sites(data.graphs, data.sites, positions[k:k + batch_size]).to(device)
            out.append(net.readout_vector(b).double().cpu().numpy())
    return np.concatenate(out)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Coefficients (with intercept as last row) of ridge on centred data, closed form."""
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return np.vstack([w, ym - xm @ w])


def _predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ coef[:-1] + coef[-1]


def probe(x_tr, y_tr, groups_tr, x_te, y_te, *, folds: int = 5) -> dict:
    """Per-target alpha by grouped CV, then test R^2 per target."""
    from sklearn.model_selection import GroupKFold

    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    xt, xe = (x_tr - mu) / sd, (x_te - mu) / sd
    cv = np.zeros((len(ALPHAS), y_tr.shape[1]))
    for tr, va in GroupKFold(n_splits=folds).split(xt, groups=groups_tr):
        for i, a in enumerate(ALPHAS):
            cv[i] += ((_predict(_ridge_fit(xt[tr], y_tr[tr], a), xt[va]) - y_tr[va]) ** 2).sum(0)
    best = ALPHAS[cv.argmin(0)]
    r2, pred = np.zeros(y_tr.shape[1]), np.zeros_like(y_te)
    for j, a in enumerate(best):
        coef = _ridge_fit(xt, y_tr[:, [j]], a)
        pred[:, j] = _predict(coef, xe)[:, 0]
        sst = ((y_te[:, j] - y_te[:, j].mean()) ** 2).sum()
        r2[j] = 1 - ((pred[:, j] - y_te[:, j]) ** 2).sum() / sst if sst > 0 else np.nan
    return {"r2": r2.tolist(), "alpha": best.tolist()}


def shuffle_by_host(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Each host's sites take the labels of one other host (host-level permutation; sites cycled if counts
    differ)."""
    rng = np.random.default_rng(seed)
    hosts = np.unique(groups)
    perm = dict(zip(hosts, rng.permutation(hosts), strict=True))
    idx = {h: np.nonzero(groups == h)[0] for h in hosts}
    out = np.empty_like(y)
    for h in hosts:
        src = idx[perm[h]]
        dst = idx[h]
        out[dst] = y[src[np.arange(len(dst)) % len(src)]]
    return out
