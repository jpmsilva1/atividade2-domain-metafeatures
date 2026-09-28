# Domain-specific meta-features for resampling selection in imbalanced time series

IN1097 (Tópicos Avançados em Agentes Inteligentes 2), Atividade 2. Does adding
domain-specific meta-features to the Atividade 1 meta-dataset improve the recommendation of
resampling workflows? The meta-target P (24 imbalanced time series × 52 workflows, F1φ) and
the leave-one-dataset-out protocol are **unchanged from Atividade 1**; only the meta-feature
matrix X changes.

| Feature set | Columns |
|---|---|
| `X_base` | the 15 Atividade 1 descriptors (4 imbalance + 11 TSFEL) |
| `X_pruned` | `X_base` without `pct_rare_08` (r = 0.98 with `pct_rare_09`) — 14 |
| `X_ext` | `X_pruned` + **C2**, **S4**, **L1** (Lorena et al. 2018) + **extremal index θ̂** (Ferro & Segers 2003) — 18 |

**Result:** `X_ext` does not improve any approach (mean Spearman 0.467 → 0.454, 0.457 → 0.440,
0.467 → 0.458 for Approach 1, Approach 2, HARRIS). The strongest new feature (C2) predicts how
*hard* a dataset is for every workflow alike, which does not change the *ranking* the
recommender is scored on; θ̂ is the only genuinely new information but too few series have
bursty extremes for the meta-model to use it. All executed outputs are in `results/`.

## Reproduce

```bash
pip install -r requirements.txt
python3 run_all.py --tests     # checks the rewrite reproduces Atividade 1 exactly
python3 run_all.py             # features -> analysis -> figures (writes results/)
python3 run_all.py figures     # or any single stage
```

## Code, in reading order

| File | Step |
|---|---|
| `src/data.py` | paths, series loading, time-delay embedding, meta-target P / R |
| `src/features.py` | X_base, the 4 new features (each timed), the three feature sets |
| `src/approaches.py` | AR, MR, Significant Wins, Approach 1, Approach 2, HARRIS |
| `src/analysis.py` | §2.1.1 association + importance, §2.1.2 redundancy, §2.1.3 LODO + critical difference |
| `src/figures.py` | every figure, from the CSVs in `results/` |
| `run_all.py` | runs the stages in order |
| `tests/test_pipeline.py` | P, R, X_base and all six approaches match Atividade 1; θ̂ recovers known θ; C2/S4/L1 on known series |

## Results

| File | Content |
|---|---|
| `X_all.csv` | the 19 meta-features for the 24 datasets |
| `extraction_cost.csv` | seconds per feature block per dataset |
| `redundancy_corr.csv`, `redundancy_groups.csv` | Spearman correlation between meta-features, groups with \|r\| > 0.95 |
| `association_level.csv` | Spearman of each feature with each workflow's F1φ, mean/sd over the 52 workflows |
| `importance_meta_ir.csv`, `best_labels.csv` | Gini importance of RF classifiers for the best learner and best resampling family (Meta-IR, Fig. 11), and those labels |
| `importance_per_learner.csv`, `best_strategy_per_learner.csv` | Gini importance of one RF classifier per base learner predicting its best resampling family (Meta-Scaler, Fig. 5), and those labels |
| `predictive_check.csv` | Leave-one-out accuracy of each importance classifier vs most-common class and 30 label shuffles (chance); only the best-learner one beats chance |
| `partial_dependence.csv` | Partial dependence of the best-resampling-family classifier on the 4 new features (FFORMS), with 95% bands from the ICE curves (`lower`, `upper`) |
| `spearman.csv`, `summary.csv`, `loss_curves.csv`, `cd.json` | LODO results per approach × feature set; CD test of the 6 approaches per feature set |
| `seed_sensitivity.csv` | Approach 1/2 mean Spearman under 5 extra forest seeds |
| `figures/` | all figures (PDF + PNG) |

## Data

Vendored from the 002 replication project of Moniz, Branco & Torgo (2017); provenance and
SHA-256 checksums in `data/SNAPSHOT.md` (`sync_data.sh` re-vendors it).
`data/atividade1_reference/` holds Atividade 1's results, used only by the tests.
