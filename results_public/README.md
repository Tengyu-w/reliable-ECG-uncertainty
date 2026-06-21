# Public Results

This directory contains curated aggregate evidence for the public GitHub
repository.

It does not contain raw ECG data, model checkpoints, embeddings, window-level
predictions, or private review examples.

## Figures

Main summary figures:

```text
figures/model_performance_summary.png
figures/embedding_geometry_distances.png
figures/uncertainty_error_detection.png
figures/review_routing_vtvf_capture.png
```

Extended public-safe figure atlas extracted from the integrated experiment
report:

```text
figures_compendium/
```

This atlas contains projection galleries, uncertainty/review curves, regularity
feature figures, full-spectrum shift and severity figures, PRO/RISK evidence,
risk-head results, and ablation visuals.

It intentionally excludes private review examples, raw ECG waveform case
galleries, and raw image archive material.

## Summary Tables

```text
summary_tables/dataset_split_statistics.csv
summary_tables/model_performance_and_geometry.csv
summary_tables/uncertainty_error_detection.csv
summary_tables/review_routing_boundary_lrii.csv
summary_tables/paired_classification_comparisons.csv
summary_tables/paired_review_routing_comparisons.csv
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
