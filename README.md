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

## Research Summary

This repository contains a PyTorch pipeline for ECG window classification,
uncertainty estimation, calibration analysis, corruption/OOD testing, VT/VF
boundary analysis, and review-oriented routing.

Key ideas:

- record-level train/validation/test splitting to reduce leakage from adjacent
  windows of the same ECG recording;
- comparison of softmax uncertainty, embedding-space atypicality, local
  neighbourhood instability, ECG regularity features, and perturbation
  sensitivity;
- explicit analysis of VT/VF cross-classification, rather than reporting only
  aggregate accuracy;
- selective prediction and review-routing evaluation, where the model accepts
  lower-risk windows and routes higher-risk windows for review;
- multi-seed paired summaries, reported cautiously because the current paired
  intervention comparison uses three seeds.

For the full research narrative, see
[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).

For a more detailed experiment-by-experiment account, including the rationale
for the embedding/PCA analysis, uncertainty comparisons, OOD tests, PRO/RISK
experiments, and review-routing conclusions, see
[docs/COMPLETE_EXPERIMENT_COMPENDIUM.md](docs/COMPLETE_EXPERIMENT_COMPENDIUM.md).

## Public Evidence

The public evidence layer contains only aggregate tables and figures. It does
not include raw ECG signals, model weights, embeddings, window-level prediction
files, or private review examples.

![Model performance summary](results_public/figures/model_performance_summary.png)

![Review routing summary](results_public/figures/review_routing_vtvf_capture.png)

Additional figures and summary tables are available in
[results_public/](results_public/README.md). The extended public figure atlas is
available in
[results_public/figures_compendium/](results_public/figures_compendium/README.md).

## Repository Structure

```text
src/                      Core training, uncertainty, calibration, OOD,
                          boundary, and review-routing code
docs/                     Research report, method overview, data statement,
                          and experiment pipeline
results_public/           Curated aggregate figures and summary tables only
results_public/figures_compendium/
                          Extended public-safe figure atlas from the integrated
                          experiment report
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
- The paired intervention comparison uses three paired random seeds.
- Window-level classification should not be interpreted as patient-level
  diagnosis.
- External-dataset and clinical validation have not been performed.
- The repository is intended for research review, not deployment.

