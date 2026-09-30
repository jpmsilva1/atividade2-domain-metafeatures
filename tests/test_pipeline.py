"""Invariants the pipeline must hold (PLAN.md, "Safety net").

The assignment freezes P and the protocol, so the first tests check this rewrite
against Atividade 1's committed results in data/atividade1_reference/ -- an
independent source of truth, not a recomputation of the same code.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import analysis, approaches, data, features, figures

REFERENCE = data.DATA_DIR / "atividade1_reference"


def test_meta_target_matches_atividade1():
    P, R, _ = data.meta_target()
    assert_frame_equal(P, pd.read_csv(REFERENCE / "P.csv", index_col=0), check_names=False)
    assert_frame_equal(R, pd.read_csv(REFERENCE / "R.csv", index_col=0), check_names=False)


def test_embedding_mirrors_r_embed_with_full_length():
    # R: embed(0:14, 10)[, 10:1] -> 6 rows, first row 0..9, last row 5..14.
    E = data.embed(np.arange(15.0))
    assert list(E.columns) == [f"V{i}" for i in range(1, 11)]
    assert E.shape == (6, 10)
    assert E.iloc[0].tolist() == list(range(10))
    assert E.iloc[-1].tolist() == list(range(5, 15))


def test_x_base_matches_atividade1():
    # Two of the short series keep the test fast; extraction is per-dataset anyway.
    series = {ds: y for ds, y in data.load_series().items() if ds in ("DS01", "DS14")}
    expected = pd.read_csv(REFERENCE / "X.csv", index_col=0).loc[["DS01", "DS14"]]
    # check_dtype=False: TSFEL's autocorr is integer-valued, so the CSV reads it back as int.
    assert_frame_equal(features.x_base(series), expected, check_names=False, check_dtype=False)


def test_complexity_features_on_known_series():
    # A straight line: every lag is perfectly rank-correlated with the target (c2 = 1),
    # a linear model fits it (l1 ~ 0) and 1-NN barely errs on interpolated points (s4 ~ 0).
    # White noise: lags carry no information, so all three are clearly worse.
    line = np.arange(2000.0)
    noise = np.random.default_rng(0).standard_normal(2000)
    smooth, rough = features.complexity(line), features.complexity(noise)
    assert smooth["c2"] == pytest.approx(1.0)
    assert smooth["l1"] < 0.05 and smooth["s4"] < 1e-4
    assert rough["c2"] < 0.1
    assert rough["l1"] > smooth["l1"] and rough["s4"] > smooth["s4"]


def test_approaches_reproduce_atividade1_spearman():
    # The whole Atividade 1 protocol (6 approaches x 24 LODO folds) on X_base must
    # give the same per-dataset Spearman as Atividade 1 -- the evidence that the
    # simplified code (incl. HARRIS's faster ranking loss) changed nothing.
    P, R, P_folds = data.meta_target()
    X = pd.read_csv(REFERENCE / "X.csv", index_col=0)
    got = analysis.lodo(X, P, R, P_folds, approaches.ALL)
    expected = pd.read_csv(REFERENCE / "spearman.csv")
    merged = expected.merge(got, on=["dataset_id", "approach"], suffixes=("_a1", ""))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged["spearman"], merged["spearman_a1"], atol=1e-10)


def test_loss_curve_is_the_gap_to_the_best_workflow_found_so_far():
    # P over 3 workflows; the ranking tries B, then A, then C. Best is C = 0.9.
    P_row = pd.Series({"A": 0.5, "B": 0.2, "C": 0.9})
    ranking = pd.Series({"A": 2, "B": 1, "C": 3})
    np.testing.assert_allclose(analysis.loss_curve(P_row, ranking), [0.7, 0.4, 0.0])


def test_cd_cliques_join_only_methods_within_cd():
    # Ranks 1.0, 1.5, 3.0 with CD 0.8: only the first two are indistinguishable.
    assert figures.cliques([1.0, 1.5, 3.0], cd=0.8) == [(0, 1)]
    # With CD 2.5 all three are; the sub-run (1, 2) is contained and not repeated.
    assert figures.cliques([1.0, 1.5, 3.0], cd=2.5) == [(0, 2)]
    # 1.0-2.0 and 2.0-3.0 each within CD 1.0, but 1.0-3.0 is not: two overlapping groups.
    assert figures.cliques([1.0, 2.0, 3.0], cd=1.0) == [(0, 1), (1, 2)]


def test_extremal_index_recovers_known_theta():
    # Theory: iid noise has theta = 1 (isolated extremes); the moving maximum
    # max(Z_t, Z_t-1) has theta = 1/2 (every extreme arrives as a pair).
    z = np.random.default_rng(0).standard_normal(200_001)
    iid, moving_max = z[1:], np.maximum(z[1:], z[:-1])
    for y, theta in [(iid, 1.0), (moving_max, 0.5)]:
        exceed = y > np.quantile(y, 0.99)
        assert abs(features.extremal_index(exceed) - theta) < 0.08
