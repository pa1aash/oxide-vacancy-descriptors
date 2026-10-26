"""Random-forest baselines, set up as in Kumagai et al. (PRM 5, 123803), Sec. II I and the released
``vacancy_formation_energy_ml/machine_learning.py``.

Reported hyperparameters: 400 trees; ``max_features`` chosen by grid search over range(20, 45)
with fourfold ShuffleSplit cross-validation (test_size = 0.1, rows are sites, as released) and the
mean-squared-error loss; all other parameters at scikit-learn defaults. For feature sets with fewer
than 70 columns the grid keeps the same fractions of the feature count (``max_features_grid``);
the CV scheme and tree count are unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, ShuffleSplit

N_TREES = 400
CV_SPLITS = 4
CV_TEST_SIZE = 0.1
GRID_RANGE = range(20, 45)  # of 70 features
N_FEATURES_REF = 70
PERM_REPEATS = 5


def max_features_grid(n_features: int) -> list[int]:
    """range(20, 45) at 70 features; the same fractions of the feature count otherwise."""
    vals = {max(1, min(n_features, round(k / N_FEATURES_REF * n_features))) for k in GRID_RANGE}
    return sorted(vals)


def make_fit(data, features: list[str], *, importances: bool = True, n_jobs: int = 8):
    X = data.uni[features].to_numpy(float)
    y = data.uni.target_Ef_eV.to_numpy(float)
    grid = max_features_grid(len(features))

    def fit(tr, te, seed):
        gs = GridSearchCV(
            RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed),
            param_grid={"max_features": grid}, scoring="neg_mean_squared_error",
            cv=ShuffleSplit(n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=seed),
            n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        best = gs.best_estimator_
        best.set_params(n_jobs=1)
        pred = best.predict(X[te])
        info = {"best_max_features": int(gs.best_params_["max_features"]),
                "cv_rmse_eV": float(np.sqrt(-gs.best_score_))}
        imp = dict(zip(features, best.feature_importances_, strict=True))
        info["importance"] = {f: {"impurity": float(imp[f])} for f in features}
        if importances:
            pi = permutation_importance(best, X[te], y[te], scoring="neg_mean_absolute_error",
                                        n_repeats=PERM_REPEATS, random_state=seed, n_jobs=n_jobs)
            for f, m, s in zip(features, pi.importances_mean, pi.importances_std, strict=True):
                info["importance"][f].update(perm_mae_increase_eV=float(m), perm_sd=float(s))
        return {"pred": pred, "info": info}
    return fit
