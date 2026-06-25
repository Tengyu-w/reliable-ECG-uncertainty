# Reliable ECG Classification Under Uncertainty

Research code for studying whether an ECG classifier can recognise when its
own SR/VT/VF prediction is unreliable.

The project focuses on a three-class rhythm classification task:

- `SR`: sinus or non-ventricular rhythm
- `VT`: ventricular tachycardia
- `VF`: ventricular fibrillation

The central question is not only classification accuracy. The main question is
whether high-risk VT/VF boundary errors can be detected and routed for expert
review before an automated prediction is accepted.

> Research prototype only. This repository is not a medical device and must not
> be used for diagnosis or clinical decision-making.

## GitHub Reading Note

This public repository is organized around the final GitHub interpretation of
the project. The strongest current claim is review-routing reliability under a
stricter duplicate-family split, with RISK used as a deployable
review-priority score. Earlier V3/V4/V5 PRO and mitigation results are retained
as historical evidence because they explain how the project evolved, but they
should not be read as the final conclusion.

In particular, the current interpretation is:

- PRO is a boundary-structure intervention and error-migration analysis, not a
  stable standalone accuracy improvement.
- RISK is the main deployable reliability contribution because it ranks ECG
  windows for expert review under fixed review budgets.
- All claims are internal research evidence only; external clinical validation
  has not been performed.

##  Application Entry

 start here:

1. [Application index](docs/APPLICATION_INDEX.md)
2. [PhD application project brief](docs/phd_application_project_brief.md)
3. [Evidence index](docs/evidence_index.md)
4. [Research report](docs/RESEARCH_REPORT.md)
5. [Figure atlas](docs/FIGURE_ATLAS.md)

The intended application framing is trustworthy medical machine learning and
review-routing reliability. It should not be described as a clinical system.

## Research Summary

This repository contains a PyTorch pipeline for ECG window classification,
uncertainty estimation, calibration analysis, corruption/OOD testing, VT/VF
boundary analysis, and review-oriented routing.

Key ideas:

- record-level and duplicate-family train/validation/test splitting to reduce
  leakage from adjacent or exactly repeated ECG windows;
- embedding-space instability analysis, including PCA/3D projections,
  normalized class-centre distances, kNN neighbourhood atypicality, local
  VT/VF mixing, and boundary-error concentration;
- comparison of softmax uncertainty, embedding-space atypicality, local
  neighbourhood instability, ECG regularity features, conformal prediction
  sets, and perturbation sensitivity;
- explicit analysis of VT/VF cross-classification, rather than reporting only
  aggregate accuracy;
- selective prediction and review-routing evaluation, where the model accepts
  lower-risk windows and routes higher-risk windows for review;
- multi-seed paired summaries, reported cautiously because the current paired
  intervention comparison uses three seeds.
- negative and mixed results are reported explicitly, including weak
  uncertainty scores, review-routing failures, and statistically uncertain
  paired comparisons.

For the full stage-ordered research report, see
[docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md).

For a more detailed experiment-by-experiment account, including the rationale
for the embedding/PCA analysis, uncertainty comparisons, OOD tests, PRO/RISK
experiments, and review-routing conclusions, see
[docs/COMPLETE_EXPERIMENT_COMPENDIUM.md](docs/COMPLETE_EXPERIMENT_COMPENDIUM.md).

For the organized public figure atlas, see
[docs/FIGURE_ATLAS.md](docs/FIGURE_ATLAS.md).

For the research version history, see
[docs/PROJECT_EVOLUTION.md](docs/PROJECT_EVOLUTION.md).

For a section-by-section audit against the integrated Word report, see
[docs/WORD_COMPENDIUM_COVERAGE_AUDIT.md](docs/WORD_COMPENDIUM_COVERAGE_AUDIT.md).

## Method And Evidence Chain

The project follows a deliberately staged reliability workflow. Each stage asks
one research question and produces public aggregate evidence.

| Stage | Why this was done | Public evidence | Main finding |
| --- | --- | --- | --- |
| 1. Leakage-aware data protocol | ECG windows from the same or duplicated records can make results look artificially strong. | `src/audit_data_protocol.py`, `src/audit_duplicate_family_splits.py`, `results_public/tables/dataset_split_statistics.csv` | The final GitHub interpretation uses stricter duplicate-family evidence before making reliability claims. |
| 2. Backbone classification | Reliability claims should not depend on a single neural architecture. | `src/train.py`, `src/models.py`, `results_public/tables/model_performance_and_geometry.csv` | Several backbones reach strong internal accuracy, but classification performance alone does not identify unsafe VT/VF boundary errors. |
| 3. Embedding geometry and PCA | The core hypothesis is that unstable ECG windows occupy ambiguous regions of learned representation space. | `src/embedding_geometry_analysis.py`, `results_public/figures/01_embedding_pca/contact_sheet.png`, `results_public/figures/00_summary/embedding_geometry_distances.png` | Across the public backbone summary, normalized SR-VT and SR-VF centre distances are much larger than VT-VF distance: SR-VT is about 2.05-2.97, SR-VF about 2.73-4.61, and VT-VF only about 0.61-0.96. This supports the interpretation that VT/VF is the key boundary. |
| 4. Local neighbourhood instability | A sample can be uncertain because its nearest embedding neighbours are atypical or locally mixed, not only because softmax confidence is low. | `src/ambiguity_analysis.py`, `src/generate_risk_targets.py`, `results_public/tables/uncertainty_error_detection.csv` | kNN-based error detection is informative in several models, with public AUROC values such as 0.897 for TCN-20 and 0.855 for ResNet1D-12. This is why RISK includes neighbourhood evidence rather than only softmax confidence. |
| 5. Calibration, uncertainty, and conformal sets | A safety-critical classifier should be able to abstain or express ambiguity instead of forcing every ECG window into one label. | `src/evaluate_uncertainty.py`, `src/selective_analysis.py`, `src/conformal_analysis.py`, `results_public/figures/02_uncertainty_calibration/contact_sheet.png` | MSP and entropy are strong ordinary error detectors, while energy score is weak or inverted in this setting. Conformal `{VT, VF}` sets are useful as a boundary-aware baseline but do not replace fixed-budget review evaluation. |
| 6. OOD and corruption robustness | Clean-test accuracy does not show whether the model reacts to degraded ECG signals. | `src/evaluate_ood.py`, `src/evaluate_corruption_severity.py`, `results_public/figures/04_ood_corruption/contact_sheet.png` | Reliability must be tested under noise, drift, masking, clipping, amplitude change, and mixed degradation. Severe corruptions remain a limitation rather than a solved problem. |
| 7. Regularity features | VT and VF differ in rhythm/frequency structure, so signal-level features can help explain instability. | `src/regularity_analysis.py`, `src/regularity_feature_ablation.py`, `results_public/figures/03_regularity_interpretability/contact_sheet.png` | Regularity features do not replace neural embeddings, but they make the failure analysis less black-box. |
| 8. PRO boundary intervention | If VT/VF errors are structured in embedding space, boundary intervention should be tested rather than only reported. | `src/pro_geometry_comparison.py`, `results_public/figures/06_pro_geometry/contact_sheet.png`, `results_public/figures/10_v6_pro_error_migration/contact_sheet.png` | PRO is retained as boundary-structure and error-migration evidence. Under stricter duplicate-family analysis it should not be overclaimed as a stable standalone improvement. |
| 9. RISK review routing | The final decision problem is which windows should be automatically accepted and which should be routed to expert review. | `src/train_embedding_risk_head.py`, `src/risk_head_review_analysis.py`, `results_public/tables/duplicate_family_selected_risk_review_aggregate.csv` | In the final duplicate-family aggregate, validation-selected RISK captures all VT/VF boundary errors at a 20% review burden across three seeds, while the automatic-route VT/VF error rate is 0. This is internal evidence only, not clinical validation. |

The central evidence chain is therefore:

```text
ECG windows -> model embeddings -> PCA/centre-distance geometry
            -> kNN/local-mixing instability -> uncertainty and OOD tests
            -> RISK review score -> fixed-budget expert-review routing
```

This is why the project is not presented as a simple ECG classifier. The main
research contribution is the analysis and routing of unreliable predictions,
especially around the VT/VF boundary.

## Public Evidence

The public evidence layer contains only aggregate tables and figures. It does
not include raw ECG signals, model weights, embeddings, window-level prediction
files, or private review examples.

Current final GitHub evidence is in the duplicate-family and V6 RISK tables:

```text
results_public/tables/duplicate_family_baseline_pro_summary.csv
results_public/tables/duplicate_family_selected_risk_review_aggregate.csv
results_public/tables/duplicate_family_risk_error_type_capture_mean_std.csv
results_public/tables/duplicate_family_risk_record_cluster_ci.csv
```

Older paired PRO and review-routing tables remain in `results_public/tables/`
as historical evidence from earlier experiment versions. They are useful for
understanding the research path, but final claims should use the
duplicate-family tables above.

![Model performance summary](results_public/figures/00_summary/model_performance_summary.png)

![Review routing summary](results_public/figures/00_summary/review_routing_vtvf_capture.png)

Additional figures and tables are available in
[results_public/](results_public/README.md). The public figures are grouped by
experiment stage in [results_public/figures/](results_public/figures/README.md).

## Repository Structure

```text
src/                      Core training, uncertainty, calibration, OOD,
                          boundary, and review-routing code
docs/                     Research report, method overview, data statement,
                          and experiment pipeline
results_public/tables/    Curated aggregate summary tables only
results_public/figures/   Public-safe figure atlas grouped by experiment stage
data/README.md            Dataset access note; raw ECG data are not distributed
requirements.txt          Minimal Python dependencies
```

## Experiment Order

The code is modular, but the intended experiment order is documented in
[docs/EXPERIMENT_PIPELINE.md](docs/EXPERIMENT_PIPELINE.md).

Main entry points:

- `src/inspect_data.py`: dataset inspection
- `src/train.py`: model training
- `src/evaluate_uncertainty.py`: uncertainty and calibration evaluation
- `src/evaluate_ood.py`: corruption and OOD evaluation
- `src/embedding_geometry_analysis.py`: representation-space analysis
- `src/ambiguity_analysis.py`: VT/VF boundary ambiguity analysis
- `src/review_efficiency_analysis.py`: review burden and error-capture curves
- `src/run_multiseed_experiments.py`: paired multi-seed experiments
- `src/seedwise_statistical_summary.py`: paired seed-level summaries

See [src/README.md](src/README.md) for a code map organised by experiment
stage.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.inspect_data --mat RHYTHMS.mat
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 30
python -m src.evaluate_uncertainty --run-dir results\<run-name>
python -m src.evaluate_ood --mat RHYTHMS.mat --run-dir results\<run-name> --model cnn
python -m src.ambiguity_analysis --run-dir results\<run-name>
python -m src.review_efficiency_analysis --run-dir results\<run-name>
```

The raw ECG file is expected locally as `RHYTHMS.mat`, but it is not included in
this repository.

## Data And Scope

The dataset is institutionally restricted and is not redistributed here. The
repository provides code and aggregate non-identifiable evidence only.

Before any public or archival release, the dataset source, licence, consent or
ethics status, and access procedure should be documented in
[data/README.md](data/README.md).

## Limitations

- Current evidence comes from an internal ECG dataset and synthetic corruption
  tests.
- The final duplicate-family evidence uses three paired random seeds and
  aggregate internal validation.
- Some methods were weak or unstable: energy-based error detection was poor in
  the selected summaries, stronger classifiers did not always rank VT/VF
  boundary errors well for review, and PRO showed error migration under the
  stricter duplicate-family split.
- Window-level classification should not be interpreted as patient-level
  diagnosis.
- External-dataset and clinical validation have not been performed.
- The repository is intended for research review, not deployment.

