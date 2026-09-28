"""Step 2 -- the meta-features X, one row per dataset.

  X_base    the 15 Atividade 1 descriptors (4 imbalance + 11 TSFEL), untouched
  X_pruned  X_base without pct_rare_08 (redundant with pct_rare_09)      -> 14
  X_ext     X_pruned + the 4 new descriptors below                        -> 18

The 4 new descriptors (Atividade 2):
  c2         feature-target correlation      (Lorena et al. 2018, complexity)
  s4         1-NN non-linearity              (Lorena et al. 2018, complexity)
  l1         linear-model error               (Lorena et al. 2018, model-based)
  theta_hat  extremal index of the rare values (Ferro & Segers 2003, tail)

c2, s4 and l1 are computed on the time-delay embedding the base experiment
trains on (V1..V9 -> V10), min-max scaled to [0, 1] per dataset so values are
comparable across series with different units (as ECoL does). theta_hat is
computed on the raw, contiguous series (see extremal_index).
"""
import time

import ImbalancedLearningRegression as iblr
import numpy as np
import pandas as pd
from imbalance_eval import _iblr_numpy_quantile_compat, compute_imbalance
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from tsfel.feature_extraction import features as tsfel

from src import data
from src.data import SEED

X_BASE = [
    "log_N", "IR", "pct_rare_09", "pct_rare_08",                          # imbalance
    "autocorr", "hurst_exponent", "slope", "skewness", "kurtosis",        # TSFEL ...
    "interq_range", "spectral_entropy", "mean_abs_diff", "lempel_ziv",
    "neighbourhood_peaks", "calc_centroid",
]
X_PRUNED = [f for f in X_BASE if f != "pct_rare_08"]
NEW_FEATURES = ["c2", "s4", "l1", "theta_hat"]
X_EXT = X_PRUNED + NEW_FEATURES

# The series have no shared time unit (daily, hourly, half-hourly), so TSFEL's
# sampling frequency is 1: the index itself is the time axis.
FS = 1.0
RARE_THRESHOLD = 0.9  # phi >= 0.9 is "rare", as in the meta-target and pct_rare_09


def base_row(y: np.ndarray) -> dict:
    """The 15 Atividade 1 descriptors of one raw series (same code path as Atividade 1).

    Imbalance features use the raw values, because phi -- and therefore the
    F1-phi meta-target -- is defined on them. TSFEL features use the z-scored
    series so they are scale-free across datasets with different units.
    """
    rare_09 = compute_imbalance(y, rel_thres=0.9)
    rare_08 = compute_imbalance(y, rel_thres=0.8)
    z = (y - y.mean()) / y.std(ddof=0)
    return {
        "log_N": np.log(rare_09["N"]),
        "IR": rare_09["IR"],
        "pct_rare_09": rare_09["%Rare"],
        "pct_rare_08": rare_08["%Rare"],
        "autocorr": tsfel.autocorr(z),
        "hurst_exponent": tsfel.hurst_exponent(z),
        "slope": tsfel.slope(z),
        "skewness": tsfel.skewness(z),
        "kurtosis": tsfel.kurtosis(z),
        "interq_range": tsfel.interq_range(z),
        "spectral_entropy": tsfel.spectral_entropy(z, FS),
        "mean_abs_diff": tsfel.mean_abs_diff(z),
        "lempel_ziv": tsfel.lempel_ziv(z),
        "neighbourhood_peaks": tsfel.neighbourhood_peaks(z),
        # Divided by the duration, otherwise the centroid just re-encodes length.
        "calc_centroid": tsfel.calc_centroid(z, FS) / (len(z) / FS),
    }


def x_base(series: dict) -> pd.DataFrame:
    """X_base: one row per dataset, the 15 columns of X_BASE."""
    return pd.DataFrame({ds: base_row(y) for ds, y in series.items()}).T[X_BASE].astype(float)


def scaled_embedding(y: np.ndarray):
    """The base experiment's embedding (V1..V9 -> V10), each column min-max
    scaled to [0, 1] so c2/s4/l1 do not depend on the series' units.
    Returns (X, t): the 9 lag columns and the target column."""
    E = data.embed(y).to_numpy()
    E = (E - E.min(axis=0)) / (E.max(axis=0) - E.min(axis=0))
    return E[:, :-1], E[:, -1]


def c2(X: np.ndarray, t: np.ndarray) -> float:
    """C2 (Lorena et al. 2018): mean |Spearman rho| between each lag and the target.
    High = the recent past still predicts the next value."""
    return float(np.mean([abs(spearmanr(X[:, j], t)[0]) for j in range(X.shape[1])]))


def s4(X: np.ndarray, t: np.ndarray) -> float:
    """S4 (Lorena et al. 2018): non-linearity of the 1-NN regressor.

    Sort the examples by target, create one synthetic example between each
    pair of neighbours in that order (random convex combination of both x and
    target), and report the MSE of a 1-NN regressor trained on the real data
    when predicting them. This is the same interpolation SMOTER performs, so a
    high S4 says interpolated examples do not look like real ones."""
    order = np.argsort(t, kind="stable")
    Xs, ts = X[order], t[order]
    lam = np.random.default_rng(SEED).uniform(size=(len(ts) - 1, 1))
    X_new = Xs[:-1] + lam * (Xs[1:] - Xs[:-1])
    t_new = ts[:-1] + lam[:, 0] * (ts[1:] - ts[:-1])
    predicted = KNeighborsRegressor(n_neighbors=1).fit(X, t).predict(X_new)
    return float(np.mean((predicted - t_new) ** 2))


def l1(X: np.ndarray, t: np.ndarray) -> float:
    """L1 (Lorena et al. 2018): mean absolute error of a linear SVR fitted to the
    data (ECoL's defaults: C = 1, epsilon = 0.1). Low = the problem is already
    close to linear, so resampling has little left to fix."""
    model = LinearSVR(C=1.0, epsilon=0.1, max_iter=10_000, random_state=SEED).fit(X, t)
    return float(np.mean(np.abs(model.predict(X) - t)))


def complexity(y: np.ndarray) -> dict:
    """c2, s4 and l1 of one raw series (convenience wrapper; x_new times them separately)."""
    X, t = scaled_embedding(y)
    return {"c2": c2(X, t), "s4": s4(X, t), "l1": l1(X, t)}


def rare_mask(y: np.ndarray) -> np.ndarray:
    """True where phi(y) >= 0.9 -- the same relevance function (boxplot
    control points, coef = 1.5, both tails) and threshold that define the rare
    cases in the F1-phi meta-target and in pct_rare_09."""
    y = pd.Series(y)
    # imbalance_eval's shim lets iblr 0.0.2 run under numpy 2 (see that package).
    with _iblr_numpy_quantile_compat():
        ctrl = iblr.phi_ctrl_pts(y, method="auto", xtrm_type="both", coef=1.5)
        phi = np.asarray(iblr.phi(y, ctrl))
    return phi >= RARE_THRESHOLD


def extremal_index(exceed: np.ndarray) -> float:
    """Extremal index theta by the intervals estimator (Ferro & Segers 2003, eq. 4).

    exceed: boolean array, True where the series is above the rarity threshold.
    With T_i the gaps between consecutive exceedance times (N exceedances,
    N-1 gaps):
        max T <= 2:  theta = 2 (sum T)^2           / ((N-1) sum T^2)
        otherwise:   theta = 2 (sum (T-1))^2       / ((N-1) sum (T-1)(T-2))
    capped at 1. theta ~ 1: rare values arrive isolated; theta -> 0: they arrive
    in bursts (mean burst size ~ 1/theta).

    The gaps are *times*, so the series must be contiguous: subsampling it
    would change every gap and silently change theta.
    """
    times = np.flatnonzero(exceed)
    if len(times) < 3:
        raise ValueError(f"extremal index needs >= 3 exceedances, got {len(times)}")
    T = np.diff(times).astype(float)
    n_gaps = len(T)
    if T.max() <= 2:
        theta = 2 * T.sum() ** 2 / (n_gaps * (T ** 2).sum())
    else:
        theta = 2 * (T - 1).sum() ** 2 / (n_gaps * ((T - 1) * (T - 2)).sum())
    return float(min(1.0, theta))


def extract(series: dict):
    """All 19 columns (X_base + the 4 new ones) for every dataset, plus the
    wall-clock cost of each feature block -- the report must state what the
    new descriptors cost to extract.

    Returns (X_all, cost): X_all is datasets x 19; cost has one row per
    (dataset, block) with the seconds that block took on that dataset.
    """
    rows, cost = {}, []

    def timed(ds, block, fn, *args):
        start = time.perf_counter()
        value = fn(*args)
        cost.append({"dataset": ds, "block": block, "n": len(series[ds]),
                     "seconds": time.perf_counter() - start})
        return value

    for ds, y in series.items():
        row = timed(ds, "X_base (imbalance + TSFEL)", base_row, y)
        X, t = timed(ds, "embedding", scaled_embedding, y)
        row["c2"] = timed(ds, "c2", c2, X, t)
        row["s4"] = timed(ds, "s4", s4, X, t)
        row["l1"] = timed(ds, "l1", l1, X, t)
        row["theta_hat"] = timed(ds, "theta_hat", lambda v: extremal_index(rare_mask(v)), y)
        rows[ds] = row

    X_all = pd.DataFrame(rows).T[X_BASE + NEW_FEATURES].astype(float)
    # Fail loudly rather than feed a NaN or an impossible value into X.
    if X_all.isna().any().any():
        raise ValueError(f"NaN meta-features:\n{X_all[X_all.isna().any(axis=1)]}")
    if not X_all["theta_hat"].between(0, 1, inclusive="right").all():
        raise ValueError("theta_hat outside (0, 1]")
    if not X_all["c2"].between(0, 1).all():
        raise ValueError("c2 outside [0, 1]")
    return X_all, pd.DataFrame(cost)


def arms(X_all: pd.DataFrame) -> dict:
    """The three feature sets compared in the experiment."""
    return {"X_base": X_all[X_BASE], "X_pruned": X_all[X_PRUNED], "X_ext": X_all[X_EXT]}
