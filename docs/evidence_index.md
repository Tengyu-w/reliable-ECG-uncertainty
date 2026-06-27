# Evidence Index

This page maps the main ECG reliability claims to public-safe evidence files.
It is intended for supervisors who want to verify the project quickly without
reading every experiment script.

## Main Reports

| File | Why it matters |
| --- | --- |
| [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md) | Compact story of the problem, experiments, negative results, and final method. |
| [RESEARCH_REPORT.md](RESEARCH_REPORT.md) | Detailed stage-ordered narrative and final interpretation. |
| [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md) | Pipeline, reliability signal families, and routing policy. |
| [FIGURE_ATLAS.md](FIGURE_ATLAS.md) | Public figure inventory. |
| [DATA_STATEMENT.md](DATA_STATEMENT.md) | What data is excluded and why. |
| [EXPERIMENT_PIPELINE.md](EXPERIMENT_PIPELINE.md) | Reproducible code entry points. |

## Key Claim Map

| Claim | Evidence |
| --- | --- |
| The repository does not redistribute raw ECG data. | [DATA_STATEMENT.md](DATA_STATEMENT.md), [data README](../data/README.md), `.gitignore`. |
| The final interpretation uses stricter internal duplicate-family evidence. | [RESEARCH_REPORT.md](RESEARCH_REPORT.md), [duplicate_family_baseline_pro_summary.csv](../results_public/tables/duplicate_family_baseline_pro_summary.csv). |
| VT/VF is the central fragile boundary. | [model_performance_and_geometry.csv](../results_public/tables/model_performance_and_geometry.csv), [embedding figures](../results_public/figures/01_embedding_pca/). |
| Uncertainty signals are compared rather than assumed. | [uncertainty_error_detection.csv](../results_public/tables/uncertainty_error_detection.csv), [uncertainty figures](../results_public/figures/02_uncertainty_calibration/). |
| Structured model interventions are not overclaimed. | [duplicate_family_pro_error_migration_mean_std.csv](../results_public/tables/duplicate_family_pro_error_migration_mean_std.csv), [PRO error-migration figures](../results_public/figures/10_v6_pro_error_migration/). |
| Embedding analysis is diagnostic rather than a guaranteed model fix. | [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md), [RESEARCH_REPORT.md](RESEARCH_REPORT.md), `src/top_journal_reliability_directions.py`. |
| RISK is the reliability evidence layer. | [duplicate_family_selected_risk_review_aggregate.csv](../results_public/tables/duplicate_family_selected_risk_review_aggregate.csv), [RISK figures](../results_public/figures/11_v6_risk_distillation/). |
| v5d is the final decision policy. | [v5d figures](../results_public/figures/12_v5d_hierarchical_router/), [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md). |
| The final router is stress-tested against small-sample concerns. | [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md), `src/internal_stress_test_v5c.py`, `src/compare_routing_baselines_10seed.py`. |
| Frozen encoder comparison is included as a foundation-model-ready baseline. | [frozen encoder figures](../results_public/figures/13_frozen_ssl_encoder/), `src/frozen_ssl_encoder_comparison.py`. |
| Explanation reliability is evaluated rather than only visualized. | [explanation audit figures](../results_public/figures/14_explanation_reliability/), `src/explanation_reliability_audit.py`. |

## Public Tables

| Table | Purpose |
| --- | --- |
| [dataset_split_statistics.csv](../results_public/tables/dataset_split_statistics.csv) | Split statistics and audit context. |
| [model_performance_and_geometry.csv](../results_public/tables/model_performance_and_geometry.csv) | Classification and representation summary. |
| [uncertainty_error_detection.csv](../results_public/tables/uncertainty_error_detection.csv) | Error-detection uncertainty metrics. |
| [review_routing_boundary_lrii.csv](../results_public/tables/review_routing_boundary_lrii.csv) | Boundary review-routing evidence. |
| [duplicate_family_selected_risk_review_aggregate.csv](../results_public/tables/duplicate_family_selected_risk_review_aggregate.csv) | RISK review-budget summary. |
| [duplicate_family_risk_error_type_capture_mean_std.csv](../results_public/tables/duplicate_family_risk_error_type_capture_mean_std.csv) | Error-type capture summary. |
| [duplicate_family_risk_record_cluster_ci.csv](../results_public/tables/duplicate_family_risk_record_cluster_ci.csv) | Record-cluster bootstrap evidence. |

## Public Figure Groups

| Figure group | Purpose |
| --- | --- |
| `00_summary` | High-level performance and reliability summaries. |
| `01_embedding_pca` | Representation geometry and VT/VF boundary evidence. |
| `02_uncertainty_calibration` | Uncertainty, calibration, and review curves. |
| `03_regularity_interpretability` | ECG regularity and signal-level evidence. |
| `04_ood_corruption` | Corruption and OOD-style robustness. |
| `10_v6_pro_error_migration` | Why PRO is treated cautiously. |
| `11_v6_risk_distillation` | RISK evidence score. |
| `12_v5d_hierarchical_router` | Final mechanism-separated routing policy. |
| `13_frozen_ssl_encoder` | Frozen self-supervised encoder comparison. |
| `14_explanation_reliability` | Explanation-to-error-mechanism audit. |

## Claim Boundaries

The evidence supports an internal research prototype for uncertainty-aware ECG
classification and review routing. It does not establish clinical validation,
patient-level diagnosis, regulatory readiness, or external generalization.
