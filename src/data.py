"""Step 1 -- the data: file locations, the raw series, and the meta-target.

Everything the later steps need comes from two vendored folders (see
data/SNAPSHOT.md for provenance):
  data/series/          one CSV per dataset: the raw time series (column "target")
  data/raw_iterations/  one CSV per dataset: F1-phi of each of the 52 workflows
                        on each train/test iteration, as run by the 002 project

Nothing here hardcodes how many datasets exist: they are discovered on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imbalance_eval import impute_dataframe
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SERIES_DIR = DATA_DIR / "series"
RAW_ITER_DIR = DATA_DIR / "raw_iterations"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 42  # every random_state in the pipeline, so a rerun is bit-for-bit identical
EMBED_SIZE = 10  # the base experiment's embedding: 9 lags + the target (Exps.R)


def dataset_id(path: Path) -> str:
    """'DS07_bike_hourly_windspeed.csv' -> 'DS07'."""
    return path.stem.split("_")[0]


def load_series() -> dict:
    """{dataset: raw series as a numpy array}, gaps filled by KNN imputation.

    Same loading as Atividade 1: DS12, DS13, DS23 and DS24 contain missing
    values, filled with imbalance_eval's KNN imputer (5 neighbours).
    """
    series = {}
    for path in sorted(SERIES_DIR.glob("DS*.csv")):
        df = pd.read_csv(path)
        if df["target"].isna().any():
            df = impute_dataframe(df)
        series[dataset_id(path)] = df["target"].to_numpy(dtype=float)
    return series


def embed(y: np.ndarray, size: int = EMBED_SIZE) -> pd.DataFrame:
    """Time-delay embedding exactly as the base experiment builds it in R
    (Exps.R: embed(ts, 10)[, 10:1], formula V10 ~ .).

    Row t holds (y[t-9], ..., y[t]) as columns V1..V10: V1..V9 are the nine
    predictors (lags), V10 is the value to forecast. Built on the full series,
    so it has len(y) - 9 rows -- nothing is subsampled.
    """
    windows = np.lib.stride_tricks.sliding_window_view(y, size)
    return pd.DataFrame(windows, columns=[f"V{i}" for i in range(1, size + 1)])


def meta_target():
    """Build the meta-target from the per-iteration results.

    Returns (P, R, P_folds):
      P        datasets x 52 workflows, mean F1-phi over iterations (higher = better)
      R        the same shape, R[d] = rank of each workflow on dataset d (1 = best)
      P_folds  {dataset: workflows x iterations} per-iteration F1-phi, used by
               Significant Wins' paired tests

    Fails loudly on a partially written file: a missing workflow x iteration cell
    would otherwise be skipped silently by pandas' mean.
    """
    P, R, P_folds = {}, {}, {}
    for path in sorted(RAW_ITER_DIR.glob("DS*.csv")):
        ds = dataset_id(path)
        folds = pd.read_csv(path).pivot_table(index="workflow", columns="iteration", values="F1")
        if folds.isna().any().any():
            raise ValueError(f"{ds}: missing workflow x iteration cells -- file partially written?")
        P_folds[ds] = folds
        P[ds] = folds.mean(axis=1)

    P = pd.DataFrame(P).T.sort_index()
    if P.isna().any().any():
        raise ValueError("datasets do not share the same set of workflows")
    # Rank 1 = highest F1-phi; ties share the average rank.
    R = P.apply(lambda row: pd.Series(rankdata(-row.to_numpy(), method="average"), index=row.index),
                axis=1)
    return P, R, P_folds
