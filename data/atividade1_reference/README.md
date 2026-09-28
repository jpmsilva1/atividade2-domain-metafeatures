# Atividade 1 reference results

Used only by `tests/test_pipeline.py` to prove that this rewrite keeps the meta-target
and the protocol of Atividade 1 unchanged, as the assignment requires.

| File | What it is | How it was produced |
|---|---|---|
| `P.csv`, `R.csv` | meta-target, 24 datasets × 52 workflows | Atividade 1's `src/meta_dataset.build` run on the same `data/raw_iterations/` |
| `X.csv` | X_base, 24 × 15 | Atividade 1's `src/meta_features.extract` run on the same `data/series/` (identical to Atividade 1's committed `results/X.csv`) |
| `spearman.csv` | per-dataset Spearman of the 6 approaches | Atividade 1's committed `results/spearman.csv` (final run, 2026-09-21) |
| `loo_summary.csv` | Atividade 1's summary table | Atividade 1's committed `results/loo_summary.csv` |

`P.csv` and `R.csv` were regenerated rather than copied because Atividade 1's committed
copies predate its last data sync (2026-09-21); its `spearman.csv` was produced after that
sync, so it already reflects the current data.
