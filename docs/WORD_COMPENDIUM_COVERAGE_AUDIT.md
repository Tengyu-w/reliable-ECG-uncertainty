# Integrated Word Compendium Coverage Audit

This audit maps the integrated Word report
`VT_VF_reliability_c整合.docx` to the public GitHub repository. Its purpose is
to make the public release traceable without exposing raw ECG signals,
window-level outputs, private review examples, checkpoints, or sample-level
clinical-looking materials.

## Bottom Line

The public repository covers the full main experimental workflow:

1. reliability framing and leakage-safe data protocol;
2. backbone training and multi-model comparison;
3. embedding geometry, PCA/projection evidence, and VT/VF boundary analysis;
4. uncertainty, calibration, selective prediction, and conformal baselines;
5. OOD-style corruption and severity robustness;
6. ECG regularity features and interpretability;
7. review routing, ambiguity-aware policy, PRO, RISK, and supervisor logic;
8. multi-seed, paired, and limitation-aware reporting.

Some sections from the Word report are intentionally represented as summaries,
not as full raw artifacts, because they contain sample-level ECG evidence or
private review material. This is a research-safety and data-governance choice,
not an omission.

## Section-Level Mapping

| Word report section | Public GitHub coverage | Status |
| --- | --- | --- |
| 1. Mathematical definitions and reliability framing | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 1-2; `docs/METHOD_OVERVIEW.md` | Covered as readable research framing. Detailed formulas are summarized rather than copied verbatim. |
| 1A. Literature-grounded upgrade path | `docs/PROJECT_EVOLUTION.md`; `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 2 | Covered as an upgrade logic and version history, not as a citation-heavy literature review. |
| 2. Dataset, preprocessing, and split evidence | `docs/DATA_STATEMENT.md`; `docs/EXPERIMENT_PIPELINE.md` Section 1; `results_public/summary_tables/dataset_split_statistics.csv` | Covered. Raw data are excluded. |
| 3. Training, backbone comparison, and multi-seed robustness | `docs/EXPERIMENT_PIPELINE.md` Sections 2 and 8; `docs/FINAL_REPORT.md` Sections 4 and 7; `results_public/summary_tables/model_performance_and_geometry.csv` | Covered. |
| 4. Embedding geometry: 2D and 3D projections | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 5; `results_public/figures_compendium/` Figures 001-026 | Covered with public-safe projection figures. |
| 5. Best-model uncertainty, ambiguity, and review curves | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 6 and 9; `results_public/summary_tables/uncertainty_error_detection.csv`; `results_public/figures_compendium/` Figures 027-036 | Covered. |
| 6. ECG regularity features and interpretability | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 8; `src/regularity_analysis.py`; `src/regularity_feature_ablation.py`; `results_public/figures_compendium/` Figures 037-042 | Covered. |
| 7. Full-spectrum shift and OOD-style perturbation benchmark | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 7; `src/evaluate_ood.py`; `src/evaluate_corruption_severity.py`; `results_public/figures_compendium/` Figures 043-053 | Covered. |
| 8. Risk head, safety supervisor, and ablation evidence | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 9 and 11; `src/runtime_supervisor.py`; `src/risk_head_review_analysis.py`; `results_public/figures_compendium/` Figures 054-066 | Covered. |
| 8A. Stability-aware reliability upgrade | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 7 and 11; `src/stability_aware_analysis.py`; `src/evaluate_risk_corruption_robustness.py` | Covered, but folded into OOD/stability and RISK sections rather than kept as a separate public chapter. |
| 8B. Conformal VT/VF boundary prediction-set baseline | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 12; `src/conformal_analysis.py`; `src/conformal_review_analysis.py` | Covered. |
| 9. Qualitative ECG case studies for expert review | Summarized in `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 13 and 15.6 | Intentionally not published as figures because it is sample-level ECG evidence. |
| 10. VT/VF boundary waveform gallery | Summarized in `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 5, 9, and 15.6 | Intentionally not published as raw waveform gallery. |
| 11. Current conclusions and what should be strengthened next | `docs/FINAL_REPORT.md` Sections 8-11; `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 14-16 | Covered. |
| Appendix A. Full raw image archive | `results_public/figures_compendium/` contains only public-safe aggregate figures | Raw archive intentionally not published. |

## Upgrade-Version Mapping

| Word upgrade section | Public GitHub coverage | Status |
| --- | --- | --- |
| V3 boundary-aware reliability supervisor and mitigation | `docs/PROJECT_EVOLUTION.md` Iteration 5; `docs/EXPERIMENT_PIPELINE.md` Sections 7-8; `src/run_mitigation_experiments.py`; `src/runtime_supervisor.py` | Covered. |
| V3.4 added training objectives: boundary-weighted CE, stability consistency, prototype/center separation, regularity auxiliary learning | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 8, 10, and 11; `src/run_core_intervention_pipeline.py`; `src/run_auxiliary_intervention_matrix.py` | Covered as method families; objective-level details are summarized. |
| V3.5 ambiguity-aware routing policy | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 9; `src/ambiguity_routing_policy.py`; `src/review_efficiency_analysis.py` | Covered. |
| V3.6 full experiment runner and V3.8 smoke verification | `docs/EXPERIMENT_PIPELINE.md` Section 8; `src/run_analysis_suite.py`; `src/run_core_validation_matrix.py` | Covered. |
| V3.7 metrics to prioritize and V3.9 manuscript-safe claims | `docs/FINAL_REPORT.md`; `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 14-16 | Covered. |
| V4 PRO, RISK, auxiliary ablations, severity robustness | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 10-11 and 15; `results_public/figures_compendium/` Figures 067-074 | Covered. |
| V4.5 auxiliary intervention matrix | `src/run_auxiliary_intervention_matrix.py`; `src/aggregate_auxiliary_robustness.py`; `docs/EXPERIMENT_PIPELINE.md` Section 8 | Covered at code/pipeline level; concise in narrative. |
| V5 review-routing reliability, RISK corruption robustness, evidence cards, code explanation | `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Sections 9, 11, and 15; `src/README.md`; `results_public/figures_compendium/` Figures 075-083 | Covered, except evidence cards are summarized rather than published. |
| V5.8 seed-wise and paired statistical reporting | `docs/FINAL_REPORT.md` Section 7; `docs/COMPLETE_EXPERIMENT_COMPENDIUM.md` Section 15.4; public paired comparison CSVs | Covered. |
| V5.9 code-level explanation and V5.10 reproduction workflow | `src/README.md`; `docs/EXPERIMENT_PIPELINE.md`; `README.md` Quick Start | Covered. |
| V6 post-V5 upgrades, duplicate-family audit, record-cluster bootstrap, six error families, final paper framing | `docs/PROJECT_EVOLUTION.md` Iteration 6; `src/audit_duplicate_family_splits.py`; `src/record_cluster_statistics.py`; `src/error_type_routing_analysis.py` | Covered at public-code and version-history level. Some case-study material is intentionally summarized. |

## Publicly Included Evidence

The public release includes:

- aggregate model, split, uncertainty, geometry, and paired-comparison tables in
  `results_public/summary_tables/`;
- curated summary figures in `results_public/figures/`;
- 83 public-safe figures extracted from the integrated report in
  `results_public/figures_compendium/`;
- a stage-ordered code map in `src/README.md`;
- a research version history in `docs/PROJECT_EVOLUTION.md`.

## Intentionally Excluded Material

The following items are not uploaded to GitHub:

- `RHYTHMS.mat` and any raw ECG records;
- checkpoints, embeddings, logits, probabilities, or window-level predictions;
- qualitative doctor-review examples;
- raw VT/VF waveform case galleries;
- private review examples and blinded images;
- risk evidence cards that expose sample-level identifiers or sample-level ECG
  evidence;
- the full raw image archive from the Word appendix.

These exclusions preserve research professionalism. A public PhD-application
repository should show the scientific workflow and aggregate evidence without
redistributing restricted or potentially identifiable biomedical material.
