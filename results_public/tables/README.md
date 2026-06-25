# Public Table Index

This folder contains aggregate, public-safe tables only. It excludes raw ECG
signals, record identifiers, embeddings, logits, probabilities, window-level
predictions, checkpoints, and private review examples.

## Current Final Evidence

Use these tables for the current GitHub interpretation:

| Table | Role |
| --- | --- |
| `duplicate_family_baseline_pro_summary.csv` | Final duplicate-family baseline and PRO seed-level summary. Shows why PRO is treated as boundary intervention and error-migration evidence, not a stable final improvement. |
| `duplicate_family_selected_risk_review_aggregate.csv` | Final validation-selected deployable RISK review-routing summary across burdens. |
| `duplicate_family_risk_error_type_capture_mean_std.csv` | Six-direction error capture for RISK under fixed review budgets. |
| `duplicate_family_risk_record_cluster_ci.csv` | Record-cluster bootstrap intervals for RISK VT/VF capture. |
| `duplicate_family_pro_record_cluster_ci.csv` | Record-cluster bootstrap intervals for PRO minus baseline. |
| `duplicate_family_pro_error_migration_mean_std.csv` | Mean/std directional error migration for PRO versus baseline. |
| `unified_review_budget_mean_std.csv` | Fixed-budget comparison of RISK, softmax uncertainty, entropy, conformal, LRII, and kNN ranking. |

## Historical Evidence

These tables come from earlier public summaries and are retained to show the
research path:

| Table | Interpretation |
| --- | --- |
| `paired_classification_comparisons.csv` | Earlier paired PRO/full-supervisor classification evidence. Historical; superseded for final PRO claims by duplicate-family analysis. |
| `paired_review_routing_comparisons.csv` | Earlier paired review-routing evidence. Historical; useful for understanding why review routing became central. |
| `model_performance_and_geometry.csv` | Public model/backbone and geometry summary. Useful as background evidence. |
| `review_routing_boundary_lrii.csv` | Earlier review-routing comparison across model families. Useful as background evidence. |
| `uncertainty_error_detection.csv` | Public uncertainty score comparison and negative results. Useful as background evidence. |
| `dataset_split_statistics.csv` | Public split statistics for the original record-level split summary. |

## Reading Rule

When writing the final project claim, use the duplicate-family tables first.
Older tables can be cited as historical or exploratory evidence, but should not
be used to claim clinical validation or stable final superiority.
