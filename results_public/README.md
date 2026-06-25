# Public Results

This directory contains curated aggregate evidence for the public GitHub
repository.

It does not contain raw ECG data, model checkpoints, embeddings, window-level
predictions, or private review examples.

## Figures

Public figures are grouped by experimental stage:

```text
figures/00_summary/
figures/01_embedding_pca/
figures/02_uncertainty_calibration/
figures/03_regularity_interpretability/
figures/04_ood_corruption/
figures/05_risk_supervisor_ablation/
figures/06_pro_geometry/
figures/07_pro_severity_robustness/
figures/08_risk_corruption_robustness/
figures/09_pro_boundary_mitigation/
figures/10_v6_pro_error_migration/
figures/11_v6_risk_distillation/
```

Each figure folder includes a `contact_sheet.png`, individual figures, and a
local README. See `docs/FIGURE_ATLAS.md` for interpretation notes.

It intentionally excludes private review examples, raw ECG waveform case
galleries, and raw image archive material.

## Tables

```text
tables/dataset_split_statistics.csv
tables/model_performance_and_geometry.csv
tables/uncertainty_error_detection.csv
tables/review_routing_boundary_lrii.csv
tables/paired_classification_comparisons.csv
tables/paired_review_routing_comparisons.csv
tables/duplicate_family_baseline_pro_summary.csv
tables/duplicate_family_selected_risk_review_aggregate.csv
tables/duplicate_family_pro_error_migration_mean_std.csv
tables/duplicate_family_risk_error_type_capture_mean_std.csv
tables/duplicate_family_risk_record_cluster_ci.csv
```

The `duplicate_family_*` tables are the current final GitHub evidence. The
older paired comparison tables are retained as historical V3/V4/V5 evidence and
should not be used to claim that PRO is a stable final improvement.

## Interpretation Notes

- The paired comparisons use three random seeds.
- The final duplicate-family summaries also use three random seeds and should
  be interpreted as internal research evidence.
- Current GitHub positioning treats PRO as boundary-structure/error-migration
  evidence and RISK as the main deployable review-routing contribution.
- The figures are intended to summarise the research direction, not to claim
  clinical validation.
- Negative and mixed results are part of the public evidence layer; the tables
  include weak uncertainty scores, unstable paired effects, and review-routing
  cases where stronger classifiers did not produce better safety ranking.
- Full run directories, model weights, embeddings, and generated reports are
  kept out of version control.
- The extended figure atlas is public-safe aggregate evidence, not a complete
  dump of all local experiment artifacts.
