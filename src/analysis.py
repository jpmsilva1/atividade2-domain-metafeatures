"""Step 4 -- the three analyses the assignment asks for (section 2.1).

  2.1.1  relation of each meta-feature to the meta-target   -> association(), importance()
  2.1.2  redundancy between meta-features                   -> redundancy()
  2.1.3  effect on the recommender, X_base vs X_ext          -> lodo(), summarize(), critical_difference()

run() executes all of them and writes the CSVs that figures.py and the report read.
"""
import json
from functools import partial

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import friedmanchisquare, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence

from src import approaches, features
from src.data import RESULTS_DIR, SEED

REDUNDANCY_THRESHOLD = 0.95  # |Spearman r| above which two meta-features say the same thing
SENSITIVITY_SEEDS = [0, 1, 2, 3, 4]
LEARNERS = ["lm", "mars", "rf", "rpart", "svm"]  # the base learners combined with resampling  # extra forest seeds for the seed-sensitivity check

# Demsar (2006), Table 5: Nemenyi critical values q_alpha (alpha = 0.05) by number of methods k.
Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


# ------------------------------------------------------------ 2.1.3 -----

def lodo(X, P, R, P_folds, models: dict) -> pd.DataFrame:
    """Leave-one-dataset-out: for every dataset, train each approach on the other
    datasets and rank the 52 workflows for the held-out one.

    models: {name: class}; a fresh instance is fitted in every fold.
    Returns one row per (dataset, approach) with the Spearman correlation between
    predicted and true ranking and the predicted ranking itself.
    """
    rows = []
    for held_out in P.index:
        train = P.index.drop(held_out)
        for name, Model in models.items():
            model = Model().fit(X.loc[train], P.loc[train], R.loc[train],
                                {d: P_folds[d] for d in train})
            predicted = model.predict(X.loc[[held_out]]).loc[held_out]
            rho = spearmanr(predicted.to_numpy(float), R.loc[held_out].to_numpy(float))[0]
            rows.append({"dataset_id": held_out, "approach": name,
                         # all-ties prediction -> undefined rho, scored as no correlation
                         "spearman": np.nan_to_num(rho, nan=0.0),
                         "predicted_ranking": predicted})
    return pd.DataFrame(rows)


def loss_curve(P_row: pd.Series, predicted_ranking: pd.Series) -> np.ndarray:
    """loss(t) = best F1-phi of the dataset - best F1-phi among the first t
    workflows of the predicted ranking, t = 1..52. Never increases; 0 at t = 52."""
    order = predicted_ranking.sort_values(kind="stable").index
    return P_row.max() - P_row.loc[order].cummax().to_numpy(float)


def summarize(results: pd.DataFrame, P: pd.DataFrame):
    """Per approach: mean and sd Spearman, the loss curve averaged over the
    held-out datasets, its area (mean over t) and loss(1), the loss of trusting
    the top recommendation alone. Returns (summary table, {approach: mean curve})."""
    rows, curves = [], {}
    for name, group in results.groupby("approach", sort=False):
        curve = np.mean([loss_curve(P.loc[r.dataset_id], r.predicted_ranking)
                         for r in group.itertuples()], axis=0)
        curves[name] = curve
        rows.append({"approach": name, "mean_spearman": group["spearman"].mean(),
                     "sd_spearman": group["spearman"].std(ddof=1),
                     "auc": curve.mean(), "loss_at_1": curve[0]})
    return pd.DataFrame(rows), curves


def critical_difference(scores: pd.DataFrame) -> dict:
    """Friedman test + Nemenyi critical difference (Demsar 2006) on a
    datasets x methods matrix of Spearman scores (higher = better).
    k and n are read from the matrix, never assumed."""
    n, k = scores.shape
    return {"avg_rank": scores.rank(axis=1, ascending=False).mean().to_dict(),
            "friedman_p": float(friedmanchisquare(*scores.T.to_numpy()).pvalue),
            "cd": float(Q_ALPHA[k] * np.sqrt(k * (k + 1) / (6 * n))),
            "n_datasets": n, "k_methods": k}


# ------------------------------------------------------------ 2.1.2 -----

def redundancy(X: pd.DataFrame):
    """Spearman correlation between every pair of meta-features, and groups of
    features that are near-copies of each other: hierarchical clustering on the
    distance 1 - |r|, cut so that everything in a group has |r| > 0.95
    (complete linkage: every pair inside a group, not just a chain).
    Returns (correlation matrix, {feature: group id})."""
    corr = X.corr(method="spearman")
    distance = squareform(1 - corr.abs().to_numpy(), checks=False)
    groups = fcluster(linkage(distance, method="complete"), t=1 - REDUNDANCY_THRESHOLD,
                      criterion="distance")
    return corr, pd.Series(groups, index=X.columns)


# ------------------------------------------------------------ 2.1.1 -----

def model_family(workflow: str) -> str:
    """'mc.svm_SMOTEB' -> 'svm'; 'mc.arima' -> 'arima'."""
    return workflow.removeprefix("mc.").split("_")[0]


def strategy_family(workflow: str) -> str:
    """'mc.svm_SMOTEB' -> 'smote', 'mc.lm_UNDERTPhi' -> 'under'; no suffix
    (plain learner, arima, BDES) -> 'none'."""
    parts = workflow.removeprefix("mc.").split("_")
    if len(parts) == 1:
        return "none"
    return next(f for f in ["under", "over", "smote"] if parts[1].lower().startswith(f))


def association(X: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation, across the datasets, of each meta-feature with each
    workflow's F1-phi, summarised per feature as mean and sd over the 52
    workflows. A high |mean| with a small sd marks a feature that tracks how hard
    the dataset is for every workflow alike -- level, not order."""
    rho = pd.DataFrame({w: X.corrwith(P[w], method="spearman") for w in P.columns})
    return pd.DataFrame({"mean": rho.mean(axis=1), "sd": rho.std(axis=1)})


def best_labels(R: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, the base learner and the resampling family of its best workflow."""
    best = R.idxmin(axis=1)
    return pd.DataFrame({"learner": best.map(model_family), "strategy": best.map(strategy_family)})


def best_strategy_per_learner(P: pd.DataFrame) -> pd.DataFrame:
    """Datasets x base learners: the resampling family of each learner's best
    workflow (argmax F1-phi over its 10 workflows) on each dataset."""
    return pd.DataFrame({m: P[[w for w in P.columns if model_family(w) == m]]
                         .idxmax(axis=1).map(strategy_family) for m in LEARNERS})


def forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    # 1000 trees: MDI from 24 rows is noisy, and more trees are cheap here.
    return RandomForestClassifier(n_estimators=1000, random_state=SEED).fit(X, y)


def importance(X: pd.DataFrame, P: pd.DataFrame, R: pd.DataFrame):
    """Gini/MDI importances of random forest classifiers fitted on all datasets.

    meta_ir:     as in Meta-IR (Avelino et al. 2025, Fig. 11, Independent): one
                 classifier for the best learner, one for the best resampling
                 family. Features x {learner, strategy}.
    per_learner: as in Meta-Scaler (de Amorim et al. 2025, Fig. 5): one classifier
                 per base learner, predicting which resampling family is best for
                 that learner. Learners x features.
    pdp:         partial dependence (Hastie et al.; as in FFORMS and MetaFore) of the
                 best-resampling-family classifier on the new features: predicted
                 probability of each family over the observed range of the feature.
    """
    labels = best_labels(R)
    models = {t: forest(X, labels[t]) for t in labels}
    meta_ir = pd.DataFrame({t: m.feature_importances_ for t, m in models.items()}, index=X.columns)
    per_strategy = best_strategy_per_learner(P)
    per_learner = pd.DataFrame({m: forest(X, per_strategy[m]).feature_importances_
                                for m in LEARNERS}, index=X.columns).T
    rows = []
    for f in features.NEW_FEATURES:
        pd_result = partial_dependence(models["strategy"], X, [f], grid_resolution=50,
                                       percentiles=(0, 1), kind="both")
        # 95% band from the ICE curves (as in FFORMS): the partial dependence is the
        # mean of the per-dataset ICE values, so mean +- 1.96 sd / sqrt(n).
        half = 1.96 * pd_result["individual"].std(axis=1, ddof=1) / np.sqrt(len(X))
        for cls, curve, h in zip(models["strategy"].classes_, pd_result["average"], half):
            rows += [{"feature": f, "value": v, "strategy": cls, "probability": p,
                      "lower": p - hh, "upper": p + hh}
                     for v, p, hh in zip(pd_result["grid_values"][0], curve, h)]
    return meta_ir, per_learner, pd.DataFrame(rows), labels, per_strategy


def loo_accuracy(X: pd.DataFrame, y: pd.Series, seed=SEED, n_estimators=500) -> float:
    """Leave-one-dataset-out accuracy of a random forest classifier."""
    hits = [RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
            .fit(X.drop(index=d), y.drop(index=d)).predict(X.loc[[d]])[0] == y[d] for d in X.index]
    return float(np.mean(hits))


def predictive_check(X: pd.DataFrame, targets: pd.DataFrame, n_shuffles=30) -> pd.DataFrame:
    """Do the classifiers behind the importances predict anything? Per target:
    leave-one-out accuracy vs always guessing the most common class, and vs the
    same procedure on shuffled labels (chance). Importances and partial dependence
    of a classifier that does not beat chance describe noise, not the data."""
    rng = np.random.default_rng(SEED)
    rows = []
    for name, y in targets.items():
        acc = loo_accuracy(X, y)
        null = np.array([loo_accuracy(X, pd.Series(rng.permutation(y.to_numpy()), index=y.index), seed=i)
                         for i in range(n_shuffles)])
        rows.append({"target": name, "loo_accuracy": acc,
                     "majority": y.value_counts().max() / len(y),
                     "shuffled_mean": null.mean(), "shuffled_p95": np.percentile(null, 95),
                     "p_value": (np.sum(null >= acc) + 1) / (n_shuffles + 1)})
    return pd.DataFrame(rows)


def seed_sensitivity(arms: dict, P, R, P_folds, seeds=SENSITIVITY_SEEDS) -> pd.DataFrame:
    """Mean Spearman of the two random-forest approaches on each arm, repeated
    with different forest seeds. With 23 training datasets a forest is noisy:
    if the gap between arms is smaller than the spread across seeds, the arms
    are not really different -- the question 2.1.3's CD diagram cannot answer."""
    rows = []
    for seed in seeds:
        models = {name: partial(approaches.ALL[name], seed=seed) for name in ["Approach1", "Approach2"]}
        for arm, X in arms.items():
            results = lodo(X, P, R, P_folds, models)
            for name, group in results.groupby("approach"):
                rows.append({"seed": seed, "arm": arm, "approach": name,
                             "mean_spearman": group["spearman"].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ run -------

def run(X_all: pd.DataFrame, P: pd.DataFrame, R: pd.DataFrame, P_folds: dict):
    """All analyses; every result lands in results/ as CSV/JSON."""
    X_all = X_all.loc[P.index]  # same datasets, same order as the meta-target
    arms = features.arms(X_all)

    # 2.1.2 redundancy -- over all 19 columns, so the pct_rare_08/09 pair is visible.
    corr, groups = redundancy(X_all)
    corr.to_csv(RESULTS_DIR / "redundancy_corr.csv")
    groups.rename("group").to_csv(RESULTS_DIR / "redundancy_groups.csv", index_label="feature")

    # 2.1.1 relation to the meta-target -- on X_ext, the extended set.
    association(arms["X_ext"], P).to_csv(RESULTS_DIR / "association_level.csv", index_label="feature")
    meta_ir, per_learner, pdp, labels, per_strategy = importance(arms["X_ext"], P, R)
    meta_ir.to_csv(RESULTS_DIR / "importance_meta_ir.csv", index_label="feature")
    per_learner.to_csv(RESULTS_DIR / "importance_per_learner.csv")
    pdp.to_csv(RESULTS_DIR / "partial_dependence.csv", index=False)
    labels.to_csv(RESULTS_DIR / "best_labels.csv", index_label="dataset")
    per_strategy.to_csv(RESULTS_DIR / "best_strategy_per_learner.csv", index_label="dataset")
    predictive_check(arms["X_ext"], pd.concat([labels, per_strategy.add_prefix("strategy_")], axis=1)
                     ).to_csv(RESULTS_DIR / "predictive_check.csv", index=False)

    compare_arms(arms, P, R, P_folds)
    seed_sensitivity(arms, P, R, P_folds).to_csv(RESULTS_DIR / "seed_sensitivity.csv", index=False)


def compare_arms(arms: dict, P, R, P_folds):
    """2.1.3: every approach under LODO on every arm -> Spearman per dataset,
    summary table, mean loss curves, and one critical-difference test per arm.
    X-blind approaches give the same result for every arm, so they run once."""
    runs = [("X-blind", lodo(arms["X_base"], P, R, P_folds,
                             {n: approaches.ALL[n] for n in ["AR", "MR", "SigWins"]}))]
    for arm, X in arms.items():
        runs.append((arm, lodo(X, P, R, P_folds,
                               {n: approaches.ALL[n] for n in approaches.X_DEPENDENT})))

    spearman, summaries, curves = [], [], []
    for arm, results in runs:
        spearman.append(results.drop(columns="predicted_ranking").assign(arm=arm))
        summary, mean_curves = summarize(results, P)
        summaries.append(summary.assign(arm=arm))
        curves += [{"approach": a, "arm": arm, "t": t + 1, "loss": v}
                   for a, c in mean_curves.items() for t, v in enumerate(c)]
    spearman = pd.concat(spearman)
    spearman.to_csv(RESULTS_DIR / "spearman.csv", index=False)
    pd.concat(summaries).to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS_DIR / "loss_curves.csv", index=False)

    # One critical-difference test per arm: the 6 approaches on the 24 datasets,
    # the Atividade 1 comparison repeated with each X.
    blind = spearman[spearman["arm"] == "X-blind"]
    cd = {}
    for arm in arms:
        scores = pd.concat([blind, spearman[spearman["arm"] == arm]]).pivot(
            index="dataset_id", columns="approach", values="spearman")
        cd[arm] = critical_difference(scores)
    (RESULTS_DIR / "cd.json").write_text(json.dumps(cd, indent=2) + "\n")
