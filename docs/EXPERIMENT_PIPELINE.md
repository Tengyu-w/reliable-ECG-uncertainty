# Experiment Pipeline

This document maps the codebase to the intended experiment order. The Python
modules are kept modular so imports and reproduction commands stay stable,
while the research workflow is ordered here. The stages mirror the main report
in `docs/RESEARCH_REPORT.md`.

## 1. Data Inspection And Split Audit

Purpose: verify that the local ECG file can be read and that the split is
record-level rather than window-level.

```powershell
python -m src.inspect_data --mat RHYTHMS.mat
python -m src.audit_data_protocol --mat RHYTHMS.mat
python -m src.audit_duplicate_family_splits --mat RHYTHMS.mat
```

Relevant files:

- `src/data.py`
- `src/inspect_data.py`
- `src/audit_data_protocol.py`
- `src/audit_duplicate_family_splits.py`
- `src/duplicate_leakage_sensitivity.py`

## 2. Baseline And Backbone Training

Purpose: train ECG classifiers and export logits, probabilities, embeddings,
and prediction summaries for later reliability analysis.

```powershell
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 30
python -m src.train --mat RHYTHMS.mat --model tcn --epochs 30
python -m src.train --mat RHYTHMS.mat --model resnet1d --epochs 30
python -m src.train --mat RHYTHMS.mat --model inception_time --epochs 30
```

Relevant files:

- `src/models.py`
- `src/train.py`
- `src/metrics.py`

## 3. Calibration And Uncertainty Evaluation

Purpose: evaluate whether confidence and uncertainty scores detect model
errors.

```powershell
python -m src.evaluate_uncertainty --run-dir results\<run-name>
python -m src.selective_analysis --run-dir results\<run-name>
python -m src.per_class_selective_analysis --run-dir results\<run-name>
```

Relevant files:

- `src/uncertainty.py`
- `src/evaluate_uncertainty.py`
- `src/selective_analysis.py`
- `src/per_class_selective_analysis.py`

## 4. Embedding Geometry And VT/VF Boundary Analysis

Purpose: test whether the learned representation separates SR from ventricular
rhythms more clearly than it separates VT from VF.

```powershell
python -m src.embedding_geometry_analysis --run-dir results\<run-name>
python -m src.ambiguity_analysis --run-dir results\<run-name>
python -m src.boundary_case_analysis --mat RHYTHMS.mat --run-dir results\<run-name>
```

Relevant files:

- `src/embedding_geometry_analysis.py`
- `src/ambiguity_analysis.py`
- `src/boundary_case_analysis.py`
- `src/pro_geometry_comparison.py`

## 5. OOD And Corruption Robustness

Purpose: test whether reliability scores respond to ECG-like perturbations and
distribution shift.

```powershell
python -m src.evaluate_ood --mat RHYTHMS.mat --run-dir results\<run-name> --model cnn
python -m src.evaluate_corruption_severity --mat RHYTHMS.mat --run-dir results\<run-name> --model cnn
python -m src.monotonicity_analysis --root results
```

Relevant files:

- `src/evaluate_ood.py`
- `src/evaluate_corruption_severity.py`
- `src/monotonicity_analysis.py`
- `src/stability_aware_analysis.py`

## 6. ECG Regularity And Reliability Fusion

Purpose: test whether handcrafted ECG regularity features add reliability
information beyond the neural embedding.

```powershell
python -m src.regularity_analysis --run-dir results\<run-name>
python -m src.feature_only_analysis --mat RHYTHMS.mat
python -m src.regularity_feature_ablation --mat RHYTHMS.mat
python -m src.gate_analysis --run-dir results\<run-name>
```

Relevant files:

- `src/regularity_analysis.py`
- `src/feature_only_analysis.py`
- `src/regularity_feature_ablation.py`
- `src/gate_analysis.py`

## 7. PRO Boundary Intervention

Purpose: test whether representation-level boundary intervention can reduce
VT/VF cross-errors or make boundary risk easier to route for review. In the
final report, PRO is interpreted cautiously as boundary-structure analysis, not
as a fully solved method.

```powershell
python -m src.pro_geometry_comparison --root results
python -m src.run_core_intervention_pipeline --mat RHYTHMS.mat
python -m src.run_mitigation_experiments --mat RHYTHMS.mat
python -m src.run_auxiliary_intervention_matrix --mat RHYTHMS.mat
```

Relevant files:

- `src/pro_geometry_comparison.py`
- `src/run_core_intervention_pipeline.py`
- `src/run_mitigation_experiments.py`
- `src/run_auxiliary_intervention_matrix.py`
- `src/aggregate_mitigation_results.py`
- `src/aggregate_auxiliary_robustness.py`

## 8. Review Routing, Conformal Baselines, And RISK

Purpose: convert reliability scores into review decisions and evaluate error
capture at fixed review budgets. This stage includes conformal prediction-set
baselines and RISK review-score distillation.

```powershell
python -m src.reliability_map --run-dir results\<run-name>
python -m src.review_efficiency_analysis --run-dir results\<run-name>
python -m src.ambiguity_routing_policy --run-dir results\<run-name>
python -m src.conformal_analysis --run-dir results\<run-name>
python -m src.conformal_review_analysis --run-dir results\<run-name>
```

Risk-head experiments:

```powershell
python -m src.generate_risk_targets --run-dir results\<run-name>
python -m src.train_embedding_risk_head --run-dir results\<run-name>
python -m src.risk_head_review_analysis --run-dir results\<run-name>
```

Relevant files:

- `src/reliability_map.py`
- `src/review_efficiency_analysis.py`
- `src/ambiguity_routing_policy.py`
- `src/conformal_analysis.py`
- `src/conformal_review_analysis.py`
- `src/generate_risk_targets.py`
- `src/train_embedding_risk_head.py`
- `src/fine_tune_risk_head.py`
- `src/risk_head_review_analysis.py`
- `src/evaluate_risk_corruption_robustness.py`
- `src/select_deployable_risk_weights.py`
- `src/aggregate_risk_versions.py`
- `src/aggregate_risk_corruption_robustness.py`

## 9. Multi-Seed, Duplicate-Family, And Aggregate Summaries

Purpose: avoid relying on a single split, a single random seed, or
over-optimistic window-level evidence. This stage includes the V6 evidence
discipline: duplicate-family audit, six-direction error analysis, and
record-cluster statistics.

```powershell
python -m src.audit_duplicate_family_splits --mat RHYTHMS.mat
python -m src.run_multiseed_experiments --mat RHYTHMS.mat
python -m src.aggregate_multiseed_results --manifest results\<manifest.csv>
python -m src.seedwise_statistical_summary --root results
python -m src.error_type_routing_analysis --root results
python -m src.record_cluster_statistics --root results
```

Additional matrix runners:

```powershell
python -m src.run_core_validation_matrix --mat RHYTHMS.mat
python -m src.run_deployable_risk_ablation --mat RHYTHMS.mat
```

Relevant files:

- `src/run_analysis_suite.py`
- `src/run_multiseed_experiments.py`
- `src/audit_duplicate_family_splits.py`
- `src/duplicate_leakage_sensitivity.py`
- `src/run_core_validation_matrix.py`
- `src/run_deployable_risk_ablation.py`
- `src/aggregate_multiseed_results.py`
- `src/aggregate_core_validation.py`
- `src/aggregate_risk_corruption_robustness.py`
- `src/seedwise_statistical_summary.py`
- `src/error_type_routing_analysis.py`
- `src/record_cluster_statistics.py`

## Notes

Training and full multi-seed experiments can be expensive. For code checking,
start with imports, data inspection, or a short smoke run before launching a
full experiment.

