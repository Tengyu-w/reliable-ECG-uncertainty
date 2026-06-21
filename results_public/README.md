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
```

## Interpretation Notes

- The paired comparisons use three random seeds.
- The figures are intended to summarise the research direction, not to claim
  clinical validation.
- Negative and mixed results are part of the public evidence layer; the tables
  include weak uncertainty scores, unstable paired effects, and review-routing
  cases where stronger classifiers did not produce better safety ranking.
- Full run directories, model weights, embeddings, and generated reports are
  kept out of version control.
- The extended figure atlas is public-safe aggregate evidence, not a complete
  dump of all local experiment artifacts.
