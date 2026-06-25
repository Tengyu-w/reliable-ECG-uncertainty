# Evidence Index

This page maps the main ECG reliability claims to public-safe evidence files.
It is intended for supervisors who want to verify the project quickly without
reading the full compendium first.

## Main Reports

| File | Why it matters |
|---|---|
| [RESEARCH_REPORT.md](RESEARCH_REPORT.md) | Stage-ordered narrative and final interpretation |
| [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md) | Pipeline, reliability signals, and evaluation views |
| [COMPLETE_EXPERIMENT_COMPENDIUM.md](COMPLETE_EXPERIMENT_COMPENDIUM.md) | Full experiment history and detailed evidence |
| [FIGURE_ATLAS.md](FIGURE_ATLAS.md) | Public figure inventory |
| [PROJECT_EVOLUTION.md](PROJECT_EVOLUTION.md) | How the claims changed as stronger evidence arrived |
| [DATA_STATEMENT.md](DATA_STATEMENT.md) | What data is excluded and why |

## Key Claim Map

| Claim | Evidence |
|---|---|
| The repository does not redistribute raw ECG data. | [DATA_STATEMENT.md](DATA_STATEMENT.md), [data README](../data/README.md) |
| The final interpretation uses duplicate-family evidence. | [RESEARCH_REPORT.md](RESEARCH_REPORT.md), [duplicate_family_baseline_pro_summary.csv](../results_public/tables/duplicate_family_baseline_pro_summary.csv) |
| RISK is the main review-routing contribution. | [duplicate_family_selected_risk_review_aggregate.csv](../results_public/tables/duplicate_family_selected_risk_review_aggregate.csv), [duplicate_family_risk_error_type_capture_mean_std.csv](../results_public/tables/duplicate_family_risk_error_type_capture_mean_std.csv) |
| PRO is not overclaimed as a stable accuracy improvement. | [duplicate_family_pro_error_migration_mean_std.csv](../results_public/tables/duplicate_family_pro_error_migration_mean_std.csv), [PROJECT_EVOLUTION.md](PROJECT_EVOLUTION.md) |
| Uncertainty signals are compared rather than assumed. | [uncertainty_error_detection.csv](../results_public/tables/uncertainty_error_detection.csv), [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md) |
| Aggregate figures are public-safe. | [results_public/figures](../results_public/figures/README.md), [FIGURE_ATLAS.md](FIGURE_ATLAS.md) |

## Public Tables

| Table | Purpose |
|---|---|
| [dataset_split_statistics.csv](../results_public/tables/dataset_split_statistics.csv) | Split statistics and audit context |
| [model_performance_and_geometry.csv](../results_public/tables/model_performance_and_geometry.csv) | Classification and representation summary |
| [uncertainty_error_detection.csv](../results_public/tables/uncertainty_error_detection.csv) | Error-detection uncertainty metrics |
| [review_routing_boundary_lrii.csv](../results_public/tables/review_routing_boundary_lrii.csv) | Boundary review-routing evidence |
| [duplicate_family_selected_risk_review_aggregate.csv](../results_public/tables/duplicate_family_selected_risk_review_aggregate.csv) | Final RISK review-budget summary |

## Claim Boundaries

The evidence supports an internal research prototype for uncertainty-aware ECG
classification and expert-review routing. It does not establish clinical
validation, patient-level diagnosis, regulatory readiness, or external
generalization.
