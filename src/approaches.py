"""Step 3 -- the six algorithm-selection approaches of Atividade 1, unchanged in
behaviour (tests/test_pipeline.py checks they reproduce Atividade 1 exactly).

Every approach has the same interface:
    fit(X, P, R, P_folds)  on the training datasets
    predict(X) -> DataFrame (test datasets x 52 workflows) of predicted ranks, 1 = best

  AR, MR, SigWins        ignore X: one fixed ranking learned from the training datasets
  Approach1              random forest regressing P (F1-phi), then rank the predictions
  Approach2              random forest regressing R (the ranks) directly
  HARRIS                 forest of hybrid ranking/regression trees (Fehring et al. 2022)
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.data import SEED

HARRIS_LAMBDA = 0.5  # weight of the ranking term (1 = pure ranking); Atividade 1's main setting
HARRIS_DEPTH = 3
HARRIS_TREES = 20
WILCOXON_ALPHA = 0.05


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank each row, 1 = lowest score; ties share the average rank."""
    return np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, np.atleast_2d(scores))


def same_ranking_for_every_row(ranking: np.ndarray, X: pd.DataFrame, columns) -> pd.DataFrame:
    return pd.DataFrame(np.tile(ranking, (len(X), 1)), index=X.index, columns=columns)


class AR:
    """Average Rank: order workflows by their mean rank over the training datasets."""

    def fit(self, X, P, R, P_folds):
        self.columns, self.ranking = R.columns, ranks(R.mean().to_numpy())[0]
        return self

    def predict(self, X):
        return same_ranking_for_every_row(self.ranking, X, self.columns)


class MR(AR):
    """Median Rank: order workflows by their median rank over the training datasets."""

    def fit(self, X, P, R, P_folds):
        self.columns, self.ranking = R.columns, ranks(R.median().to_numpy())[0]
        return self


# A dataset's pairwise Wilcoxon tests do not depend on which fold it is used in,
# so they are computed once per dataset and reused (Atividade 1 recomputed them
# in every fold: same result, 24x slower).
_WINS_CACHE = {}


def significant_wins_on(dataset: str, folds: pd.DataFrame) -> pd.Series:
    """For one dataset: how many other workflows each workflow beats with a
    significant Wilcoxon signed-rank test over the per-iteration F1-phi
    (alpha = 0.05, no multiple-comparison correction)."""
    if dataset not in _WINS_CACHE:
        wins = pd.Series(0, index=folds.index)
        for a, b in combinations(folds.index, 2):
            fa, fb = folds.loc[a].to_numpy(), folds.loc[b].to_numpy()
            if np.allclose(fa - fb, 0):
                continue  # identical results: no test possible, nobody wins
            if wilcoxon(fa, fb).pvalue < WILCOXON_ALPHA:
                wins[a if np.median(fa - fb) > 0 else b] += 1
        _WINS_CACHE[dataset] = wins
    return _WINS_CACHE[dataset]


class SigWins(AR):
    """Significant Wins: order workflows by their total significant wins over
    the training datasets (more wins = better rank)."""

    def fit(self, X, P, R, P_folds):
        total = sum(significant_wins_on(ds, P_folds[ds]) for ds in P.index)
        self.columns = P.columns
        self.ranking = ranks(-total.reindex(P.columns).to_numpy())[0]
        return self


class Approach1:
    """Regress F1-phi: one random forest per workflow (52 outputs), then rank
    the predicted F1-phi (highest = rank 1)."""

    target = "P"

    def fit(self, X, P, R, P_folds):
        Y = P if self.target == "P" else R
        self.columns = Y.columns
        # n_jobs=-1 fits the 52 forests in parallel; each keeps random_state, so
        # the result is identical to fitting them one after the other.
        self.model = MultiOutputRegressor(RandomForestRegressor(random_state=SEED), n_jobs=-1)
        self.model.fit(X.to_numpy(), Y.to_numpy())
        return self

    def predict(self, X):
        predicted = self.model.predict(X.to_numpy())
        # P: higher is better, so negate before ranking; R: already lower is better.
        return pd.DataFrame(ranks(-predicted if self.target == "P" else predicted),
                            index=X.index, columns=self.columns)


class Approach2(Approach1):
    """Regress the ranks R directly, then re-rank the predicted ranks."""

    target = "R"


# ---------------------------------------------------------------- HARRIS ----
#
# A tree node holds the training datasets that reach it. Its label is the mean
# (scaled) P vector of those datasets. A split is chosen to minimise, over the
# two children (weighted by size),
#     loss = lam * ranking_loss + (1 - lam) * regression_loss
# NOTE: lam sits on the *ranking* term, as the assignment defines it -- the
# opposite of the HARRIS paper's Eq. 3. Deliberate; do not "fix" it.

def regression_loss(P: np.ndarray) -> float:
    """MSE of the node's P rows around their mean."""
    return float(((P - P.mean(axis=0)) ** 2).mean())


def ranking_loss(R: np.ndarray) -> float:
    """Mean (1 - Spearman rho) between each row of R and the node's consensus
    (Borda) ranking, halved to lie in [0, 1] like the scaled MSE.

    Rows of R and the Borda label are already rank vectors, so Spearman rho is
    just Pearson correlation on them -- computed for all rows at once. A row with
    no variation (all ties) has undefined rho and is scored as rho = 0."""
    if R.shape[0] == 1:
        return 0.0
    borda = rankdata(R.mean(axis=0), method="average")
    rows = R - R.mean(axis=1, keepdims=True)
    label = borda - borda.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = rows @ label / (np.linalg.norm(rows, axis=1) * np.linalg.norm(label))
    return float(np.mean(1.0 - np.nan_to_num(rho, nan=0.0)) / 2.0)


def node_loss(P, R, lam):
    return lam * ranking_loss(R) + (1.0 - lam) * regression_loss(P)


def grow_tree(X, P, R, rows, depth, lam, max_features, rng) -> dict:
    """Grow one hybrid tree on the datasets `rows` (indices, may repeat: bootstrap).

    Tries every midpoint between observed values of a random subset of features
    and keeps the split with the lowest size-weighted child loss. Stops at the
    depth limit, or when no split beats the node's own loss."""
    node = {"value": P[rows].mean(axis=0)}
    if depth == 0 or len(rows) < 2:
        return node
    parent_loss = node_loss(P[rows], R[rows], lam)
    candidates = rng.choice(X.shape[1], size=min(max_features, X.shape[1]), replace=False)

    best = None
    for j in candidates:
        values = np.unique(X[rows, j])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            goes_left = X[rows, j] <= threshold
            left, right = rows[goes_left], rows[~goes_left]
            score = (len(left) * node_loss(P[left], R[left], lam)
                     + len(right) * node_loss(P[right], R[right], lam)) / len(rows)
            if best is None or score < best[0]:
                best = (score, j, threshold, left, right)

    if best is None or best[0] >= parent_loss:
        return node
    _, node["feature"], node["threshold"], left, right = best
    node["left"] = grow_tree(X, P, R, left, depth - 1, lam, max_features, rng)
    node["right"] = grow_tree(X, P, R, right, depth - 1, lam, max_features, rng)
    return node


def leaf_value(node: dict, x: np.ndarray) -> np.ndarray:
    while "feature" in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["value"]


class HARRIS:
    """Forest of HARRIS_TREES hybrid trees, each on a bootstrap sample of the
    training datasets and max(2, n_features // 2) candidate features per node."""

    def __init__(self, lam=HARRIS_LAMBDA, depth=HARRIS_DEPTH, n_trees=HARRIS_TREES):
        self.lam, self.depth, self.n_trees = lam, depth, n_trees

    def fit(self, X, P, R, P_folds):
        self.columns = P.columns
        X, P, R = X.to_numpy(float), P.to_numpy(float), R.to_numpy(float)
        # Min-max scale P globally (not per workflow): per-column scaling would
        # distort the mean-P vector that the forest finally ranks.
        P = (P - P.min()) / (P.max() - P.min()) if P.max() > P.min() else np.zeros_like(P)
        max_features = max(2, X.shape[1] // 2)
        rng = np.random.default_rng(SEED)
        n = X.shape[0]
        self.trees = [grow_tree(X, P, R, rng.integers(0, n, n), self.depth, self.lam,
                                max_features, rng)
                      for _ in range(self.n_trees)]
        return self

    def predict(self, X):
        mean_p = np.array([np.mean([leaf_value(t, x) for t in self.trees], axis=0)
                           for x in X.to_numpy(float)])
        return pd.DataFrame(ranks(-mean_p), index=X.index, columns=self.columns)


ALL = {"AR": AR, "MR": MR, "SigWins": SigWins,
       "Approach1": Approach1, "Approach2": Approach2, "HARRIS": HARRIS}
X_DEPENDENT = ["Approach1", "Approach2", "HARRIS"]
