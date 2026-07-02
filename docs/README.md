# Documentation Guide

This folder contains the supervisor-facing documentation for the project. The
repository is intentionally organized as a research portfolio, not as a raw
experiment dump.

## Recommended Reading Order

For the corrected model-first mechanism story, start with
[FINAL_MODEL_SELECTION_REPORT_CN.md](FINAL_MODEL_SELECTION_REPORT_CN.md).
The mechanism-targeted full-results document explains the component evidence;
the final model-selection report turns that evidence into the current
model-choice argument.

| Time | File | Purpose |
| --- | --- | --- |
| 2 min | [Project README](../README.md) | Main contribution, result snapshot, and limitations. |
| 5 min | [Final model selection report](FINAL_MODEL_SELECTION_REPORT_CN.md) | Completed 36-run mechanism-derived model selection, main candidate, controls, figures, and limitations. |
| 5 min | [PhD application brief](phd_application_project_brief.md) | Short supervisor-facing project pitch. |
| 10 min | [Experiment evidence summary](EXPERIMENT_EVIDENCE_SUMMARY.md) | The full research logic in one compact document. |
| Taxonomy | [Mechanism routing taxonomy](MECHANISM_ROUTING_TAXONOMY_CN.md) | Evidence head vs complete router vs recovery action comparison rules. |
| V5D Upgrade | [V5D causal-Pareto weight upgrade](V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md) | Direct causal-Pareto weight tuning inside the existing stage1/stage2 V5D router. |
| Model Upgrade | [Model-side causal-Pareto optimization](MODEL_SIDE_CAUSAL_PARETO_OPTIMIZATION_CN.md) | Separates model-only, evidence-head, and fixed-router downstream comparisons. |
| Model Result | [Model-layer causal-Pareto validation](MODEL_LAYER_CAUSAL_PARETO_VALIDATION_RESULTS_CN.md) | Internal paired-seed validation of model-layer causal-Pareto interventions. |
| Model Benchmark | [All model-layer benchmark](MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md) | Comprehensive model-stage comparison across CNN, CNN-LSTM, PRO, constrained, and complex models. |
| Model Search | [Model-layer causal-Pareto search](MODEL_LAYER_CAUSAL_PARETO_SEARCH_PLAN_CN.md) | Recombination search over prototype, boundary/risk, regularity, stability, and calibration constraints. |
| Mechanism-Derived Search | [Mechanism-derived model search plan](MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md) | Builds the next model candidates from the 33-run mechanism validation instead of heuristic weights alone. |
| Final Model Selection | [Final model selection report](FINAL_MODEL_SELECTION_REPORT_CN.md) | Uses the completed 36-run search to choose `proto_center_only` as the clearest minimal sufficient candidate and define controls. |
| Thesis Method | [Causal mechanism method section](THESIS_METHOD_SECTION_CAUSAL_MECHANISM_CN.md) | Thesis-ready method framing for causal variables, mechanism variables, outcomes, and Pareto objectives. |
| Thesis Subsection | [Causal mechanism thesis subsection](THESIS_CAUSAL_MECHANISM_SUBSECTION_CN.md) | Compact thesis-ready subsection with the intervention-mechanism-outcome diagram and training-search clarification. |
| Mechanism Inventory | [Mechanism variable master inventory](MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md) | Full inventory of representation, KNN, prototype, softmax, validity, waveform, wavelet, risk, explanation, and routing variables. |
| Boundary-Prototype Chain | [Boundary075 prototype mechanism chain](BOUNDARY075_PROTOTYPE_MECHANISM_CHAIN_CN.md) | Detailed source-to-intervention-to-mechanism-to-outcome explanation for the selected model-layer candidate. |
| Targeted Ablation | [Mechanism-targeted causal ablation plan](MECHANISM_TARGETED_CAUSAL_ABLATION_PLAN_CN.md) | Fine-grained next experiment plan for prototype, KNN, softmax, gate, and regularity mechanisms. |
| Full Mechanism Results | [Mechanism-targeted causal full results](MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md) | Final 33-run mechanism-targeted causal-style quantification, including waveform regularity as non-intervenable ECG input attributes. |
| Evidence Coverage | [Mechanism evidence coverage audit](MECHANISM_EVIDENCE_COVERAGE_AUDIT_CN.md) | Separates mechanisms with paired causal-style validation from diagnostic, auxiliary, and negative evidence. |
| Mechanism Chain | [Causal mechanism quantification](CAUSAL_MECHANISM_QUANTIFICATION_RESULTS_CN.md) | Paired `do(training constraint) -> mechanism variable -> outcome` evidence chain for model-layer interventions. |
| Evidence Library | [Mechanism library evidence chain](MECHANISM_LIBRARY_EVIDENCE_CHAIN_RESULTS_CN.md) | Unified `do(evidence/policy) -> mechanism signal -> routing/review outcome` inventory for wavelet, regularity, heads, and explanations. |
| OOD | [Route OOD stress results](ROUTE_OOD_STRESS_RESULTS_CN.md) | ECG-structure-preserving shift stress test for V5D and causal-Pareto routing. |
| Reuse | [Transferable method framework](VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md) | Method template for other manuscripts or projects. |
| 15 min | [Evidence index](evidence_index.md) | Claim-by-claim pointers to public figures and tables. |
| 30 min | [Research report](RESEARCH_REPORT.md) | Detailed stage-ordered report. |

## Core Documents

- [APPLICATION_INDEX.md](APPLICATION_INDEX.md): advisor-facing entry point.
- [FINAL_MODEL_SELECTION_REPORT_CN.md](FINAL_MODEL_SELECTION_REPORT_CN.md):
  completed mechanism-derived model selection report, including traditional
  CNN/CNN-LSTM context, candidate ranking, mechanism-to-weight bridge, figures,
  and next full-validation recommendation.
- [EXPERIMENT_EVIDENCE_SUMMARY.md](EXPERIMENT_EVIDENCE_SUMMARY.md): compact
  story of what was tested, why, what worked, and what failed.
- [MECHANISM_ROUTING_TAXONOMY_CN.md](MECHANISM_ROUTING_TAXONOMY_CN.md):
  taxonomy for evidence heads, complete routing policies, recovery actions,
  and fair comparison rules against V5D.
- [V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md](V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md):
  direct causal-Pareto weight upgrade inside the existing V5D stage1/stage2
  router.
- [MODEL_SIDE_CAUSAL_PARETO_OPTIMIZATION_CN.md](MODEL_SIDE_CAUSAL_PARETO_OPTIMIZATION_CN.md):
  model-side causal variables and fair comparison layers: model-only,
  evidence-head-only, and fixed-router downstream validation.
- [MODEL_LAYER_CAUSAL_PARETO_VALIDATION_RESULTS_CN.md](MODEL_LAYER_CAUSAL_PARETO_VALIDATION_RESULTS_CN.md):
  model-only causal-Pareto validation results over paired auxiliary
  intervention runs.
- [MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md](MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md):
  full model-stage benchmark including CNN, CNN-LSTM, PRO/prototype,
  constrained models, and complex multi-objective training variants.
- [MODEL_LAYER_CAUSAL_PARETO_SEARCH_PLAN_CN.md](MODEL_LAYER_CAUSAL_PARETO_SEARCH_PLAN_CN.md):
  active model-layer causal-Pareto recombination search over the strongest
  old constraints.
- [MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md](MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md):
  second-stage model search that uses the 33-run mechanism ablation to test
  whether `boundary + prototype center` is sufficient or whether the full
  boundary-prototype margin should be retained.
- [THESIS_METHOD_SECTION_CAUSAL_MECHANISM_CN.md](THESIS_METHOD_SECTION_CAUSAL_MECHANISM_CN.md):
  thesis-ready method section defining intervenable variables,
  non-intervenable variables, mechanism variables, outcomes, causal-style
  evidence chains, and multi-objective selection rules.
- [THESIS_CAUSAL_MECHANISM_SUBSECTION_CN.md](THESIS_CAUSAL_MECHANISM_SUBSECTION_CN.md):
  compact thesis subsection with a Mermaid method diagram and a clear
  explanation of the 2026-06-30 model-layer causal-Pareto search.
- [MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md](MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md):
  full inventory of analyzed mechanism variables across representation,
  KNN, prototype, softmax, validity, waveform, wavelet, risk, explanation,
  and routing layers.
- [BOUNDARY075_PROTOTYPE_MECHANISM_CHAIN_CN.md](BOUNDARY075_PROTOTYPE_MECHANISM_CHAIN_CN.md):
  detailed mechanism chain showing how VT/VF boundary and representation
  analyses led to the `boundary075_prototype` intervention and why it worked.
- [MECHANISM_TARGETED_CAUSAL_ABLATION_PLAN_CN.md](MECHANISM_TARGETED_CAUSAL_ABLATION_PLAN_CN.md):
  fine-grained causal-style ablation plan for testing prototype ambiguity,
  KNN purity, softmax ambiguity, gate alignment, and waveform regularity.
- [MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md](MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md):
  final full-run mechanism-targeted causal-style results, with the
  intervention-mechanism-outcome chain, core/negative candidates, and a
  waveform regularity audit that treats ECG waveform attributes as
  non-intervenable input variables.
- [MECHANISM_TARGETED_CAUSAL_QUANTIFICATION_STATUS_CN.md](MECHANISM_TARGETED_CAUSAL_QUANTIFICATION_STATUS_CN.md):
  current status of the targeted causal-style quantification pipeline,
  including intervenable variables, mechanism variables, outcomes, smoke
  validation, and the running full experiment.
- [MECHANISM_EVIDENCE_COVERAGE_AUDIT_CN.md](MECHANISM_EVIDENCE_COVERAGE_AUDIT_CN.md):
  coverage audit separating paired causal-style mechanisms, mechanism-library
  evidence, auxiliary diagnostics, and negative results.
- [CAUSAL_MECHANISM_QUANTIFICATION_RESULTS_CN.md](CAUSAL_MECHANISM_QUANTIFICATION_RESULTS_CN.md):
  paired model-layer mechanism evidence chain linking training interventions
  to embedding/KNN/prototype/softmax/validity variables and model outcomes.
- [MECHANISM_LIBRARY_EVIDENCE_CHAIN_RESULTS_CN.md](MECHANISM_LIBRARY_EVIDENCE_CHAIN_RESULTS_CN.md):
  unified evidence-library chain for wavelet, regularity, mechanism heads,
  routing policies, and explanation alignment.
- [ROUTE_OOD_STRESS_RESULTS_CN.md](ROUTE_OOD_STRESS_RESULTS_CN.md):
  ECG-structure-preserving OOD-style route stress test comparing V5D and
  causal-Pareto routing profiles.
- [VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md](VT_VF_TRANSFERABLE_METHOD_FRAMEWORK_CN.md):
  reusable method framework distilled from the VT/VF project.
- [RESEARCH_REPORT.md](RESEARCH_REPORT.md): full research narrative.
- [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md): method diagram and reliability
  signal families.
- [FIGURE_ATLAS.md](FIGURE_ATLAS.md): public figure inventory.
- [DATA_STATEMENT.md](DATA_STATEMENT.md): what data is excluded from GitHub.
- [EXPERIMENT_PIPELINE.md](EXPERIMENT_PIPELINE.md): reproducible code entry
  points.

## What Was Removed From The Public Main Path

Earlier drafts, group-meeting scripts, and stage-by-stage scratch notes were
removed from the main documentation path. Their useful conclusions were
merged into the evidence summary and research report. This keeps the repository
readable for a PhD supervisor while preserving the experiment code and public
aggregate evidence.

## Scope Boundary

This repository does not distribute raw ECG data, model checkpoints,
embeddings, window-level prediction files, or private review examples. It is a
research prototype for reliability analysis, not a clinical validation package.
