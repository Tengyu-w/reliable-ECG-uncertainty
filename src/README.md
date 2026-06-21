# Source Code Map

The source code is organised by research function rather than by numbered
filenames. The intended experiment order is documented in
`docs/EXPERIMENT_PIPELINE.md`.

## Data And Metrics

| File | Role |
|---|---|
| `data.py` | Load `RHYTHMS.mat`, create ECG windows, extract regularity features, and build record-level splits. |
| `inspect_data.py` | Inspect local ECG data structure and class composition. |
| `audit_data_protocol.py` | Check data protocol assumptions. |
| `audit_duplicate_family_splits.py` | Audit duplicate-family split logic. |
| `duplicate_leakage_sensitivity.py` | Evaluate leakage sensitivity around duplicate or near-duplicate records. |
| `metrics.py` | Classification, calibration, and summary metric utilities. |

## Models And Training

| File | Role |
|---|---|
| `models.py` | CNN, TCN, ResNet1D, InceptionTime, BiGRU, regularity-fusion, and reliability-gated models. |
| `train.py` | Main training entry point and export of predictions, logits, embeddings, and metrics. |

## Uncertainty, Calibration, And Selective Prediction

| File | Role |
|---|---|
| `uncertainty.py` | MSP, entropy, energy, temperature scaling, prototype, Mahalanobis, and kNN uncertainty scores. |
| `evaluate_uncertainty.py` | Evaluate uncertainty, calibration, error detection, and reliability diagrams. |
| `selective_analysis.py` | Coverage-risk analysis for selective prediction. |
| `per_class_selective_analysis.py` | Selective prediction broken down by class. |

## Boundary, Embedding, And OOD Analysis

| File | Role |
|---|---|
| `embedding_geometry_analysis.py` | Analyse learned representation geometry. |
| `ambiguity_analysis.py` | Quantify VT/VF ambiguity. |
| `boundary_case_analysis.py` | Inspect boundary-error cases locally. Generated case outputs are not committed. |
| `pro_geometry_comparison.py` | Compare prototype-separation effects on geometry and boundary errors. |
| `evaluate_ood.py` | Evaluate synthetic OOD/corruption behaviour. |
| `evaluate_corruption_severity.py` | Test severity-dependent corruption response. |
| `monotonicity_analysis.py` | Summarise monotonicity of uncertainty scores under corruption severity. |
| `stability_aware_analysis.py` | Analyse prediction stability and perturbation sensitivity. |

## ECG Regularity And Fusion

| File | Role |
|---|---|
| `regularity_analysis.py` | Analyse ECG rhythm/frequency regularity features. |
| `feature_only_analysis.py` | Test classical models using regularity features only. |
| `regularity_feature_ablation.py` | Ablate regularity feature groups. |
| `gate_analysis.py` | Analyse reliability-gated fusion behaviour. |

## Review Routing And Risk Modelling

| File | Role |
|---|---|
| `reliability_map.py` | Map samples into reliability regions. |
| `review_efficiency_analysis.py` | Measure review burden vs error capture. |
| `ambiguity_routing_policy.py` | Combine ambiguity and uncertainty evidence into routing policies. |
| `conformal_analysis.py` | Build conformal prediction sets. |
| `conformal_review_analysis.py` | Evaluate conformal-set review triggers. |
| `runtime_supervisor.py` | Prototype reliability-supervisor states for offline analysis. |
| `aggregate_supervisor_results.py` | Aggregate supervisor summaries across runs. |
| `error_type_routing_analysis.py` | Analyse error types under routing policies. |
| `generate_risk_targets.py` | Generate risk targets from deployable reliability evidence. |
| `train_embedding_risk_head.py` | Train an embedding risk head. |
| `fine_tune_risk_head.py` | Fine-tune the risk head. |
| `risk_head_review_analysis.py` | Evaluate risk-head review routing. |
| `evaluate_risk_corruption_robustness.py` | Test risk-head robustness under corruption. |
| `select_deployable_risk_weights.py` | Select validation-aligned risk weights. |
| `summarize_risk_ablation.py` | Summarise risk-target ablations. |

## Experiment Runners And Aggregation

| File | Role |
|---|---|
| `run_analysis_suite.py` | Run the main analysis suite for a trained model. |
| `run_multiseed_experiments.py` | Launch paired multi-seed experiments. |
| `run_core_validation_matrix.py` | Run core model validation. |
| `run_core_intervention_pipeline.py` | Run intervention comparisons. |
| `run_auxiliary_intervention_matrix.py` | Run auxiliary intervention matrix experiments. |
| `run_deployable_risk_ablation.py` | Run deployable risk ablation experiments. |
| `run_mitigation_experiments.py` | Run mitigation experiments. |
| `run_severity_validation.py` | Run corruption severity validation. |
| `aggregate_version_results.py` | Aggregate versioned model results. |
| `aggregate_multiseed_results.py` | Aggregate multi-seed results. |
| `aggregate_core_validation.py` | Aggregate core validation matrix results. |
| `aggregate_mitigation_results.py` | Aggregate mitigation results. |
| `aggregate_auxiliary_robustness.py` | Aggregate auxiliary robustness results. |
| `aggregate_risk_versions.py` | Aggregate risk-version summaries. |
| `aggregate_risk_corruption_robustness.py` | Aggregate risk corruption robustness results. |
| `aggregate_deployable_risk_ablation.py` | Aggregate deployable risk ablations. |
| `seedwise_statistical_summary.py` | Build paired seed-wise statistical summaries. |
| `record_cluster_statistics.py` | Summarise record-cluster level statistics. |
| `unified_review_budget_comparison.py` | Compare review-routing policies under unified budgets. |

