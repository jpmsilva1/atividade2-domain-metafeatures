#!/usr/bin/env python3
"""Run the whole experiment, in order. Each stage writes CSVs to results/ and
the next stage reads them back, so any intermediate result can be opened on
its own and a later stage can be re-run without redoing the earlier ones.

    python3 run_all.py                  # every stage
    python3 run_all.py figures          # only the named stage(s)
    python3 run_all.py --tests          # the test suite
"""
import subprocess
import sys
import time

import pandas as pd

from src import analysis, data, features, figures
from src.data import RESULTS_DIR


def stage_features():
    """1. Extract the 19 meta-features and their extraction cost."""
    X_all, cost = features.extract(data.load_series())
    X_all.to_csv(RESULTS_DIR / "X_all.csv")
    cost.to_csv(RESULTS_DIR / "extraction_cost.csv", index=False)


def stage_analysis():
    """2-4. Redundancy (2.1.2), relation to the meta-target (2.1.1),
    and the effect on the recommender under LODO (2.1.3)."""
    X_all = pd.read_csv(RESULTS_DIR / "X_all.csv", index_col=0)
    P, R, P_folds = data.meta_target()
    analysis.run(X_all, P, R, P_folds)


def stage_figures():
    """5. Every figure in the report, from the CSVs above."""
    figures.run()


STAGES = {"features": stage_features, "analysis": stage_analysis, "figures": stage_figures}

if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"]).returncode)
    RESULTS_DIR.mkdir(exist_ok=True)
    for name in sys.argv[1:] or STAGES:
        start = time.perf_counter()
        print(f"[{name}] ...", flush=True)
        STAGES[name]()
        print(f"[{name}] done in {time.perf_counter() - start:.0f}s", flush=True)
