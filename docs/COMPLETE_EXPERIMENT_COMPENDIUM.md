# Complete Experiment Compendium

This document is the public research narrative for the project. It is based on
the integrated experiment report, but it is rewritten for GitHub so that the
reader sees the actual research logic rather than a loose collection of scripts.

Private review examples, raw ECG waveform galleries, model checkpoints,
embeddings, and window-level prediction files are not included. The public
figures and tables are aggregate evidence only.

## 1. Research Framing

The project starts from a practical weakness of accuracy-only ECG
classification. A model can achieve high overall accuracy while still making
clinically important VT/VF boundary errors. In this task, SR is often more
separable from ventricular rhythms than VT and VF are from each other. That is
why this repository treats the problem as reliability-aware classification:

1. predict SR, VT, or VF;
2. estimate when that prediction is unreliable;
3. route high-risk windows to expert review rather than forcing automatic
   acceptance.

The research target is not a clinical product. The target is a prototype
evidence pipeline for asking whether an ECG classifier knows when it should not
be trusted.

## 2. Why This Methodology

The methodology has four layers.

First, the data split is record-level. ECG windows from the same original record
can be highly correlated. A window-level split would overstate performance by
letting near-duplicate temporal segments appear in train and test. The
record-level split is therefore a methodological requirement, not an optional
engineering detail.

Second, the project compares multiple time-series backbones. A single CNN
baseline would not show whether reliability findings are model-specific. The
backbone comparison includes CNN, TCN, ResNet1D, InceptionTime, BiGRU, and
regularity-fusion models.

Third, the project evaluates multiple uncertainty families. Softmax confidence
captures decision uncertainty, while embedding distances and local neighbourhood
signals capture atypicality and boundary structure. These are not interchangeable
signals. A useful reliability study should compare them rather than assume one
score is sufficient.

Fourth, the project evaluates review routing. Uncertainty is only useful if it
changes a downstream decision. For this project, the downstream decision is:
which ECG windows should be automatically accepted, and which should be sent for
expert review?

## 3. Dataset And Leakage Control

The code expects a local `RHYTHMS.mat` file with SR, VT, and VF records. The raw
dataset is not redistributed.

The public split summary is:

```text
results_public/summary_tables/dataset_split_statistics.csv
```

The split is evaluated at the level of source records, not only windows. Later
audits also check duplicate-family and exact-window overlap risks. This matters
because leakage would make the downstream uncertainty and review-routing results
less meaningful.

## 4. Model Training And Backbone Comparison

The main training code is:

```text
src/train.py
src/models.py
src/metrics.py
```

The training pipeline exports logits, probabilities, embeddings, predictions,
and aggregate metrics. These outputs are then consumed by uncertainty,
calibration, OOD, embedding, and review-routing modules.

The public model summary is:

```text
results_public/summary_tables/model_performance_and_geometry.csv
```

![Model performance](../results_public/figures/model_performance_summary.png)

Interpretation:

- Overall accuracy is useful but incomplete.
- Macro-F1 is important because the task is class-imbalanced.
- A high-performing backbone does not automatically produce the best review
  ranking.
- The strongest research signal is the difference between ordinary
  classification performance and reliability under constrained review.

## 5. Embedding Geometry, PCA, And Projection Evidence

The embedding analysis asks whether the model's learned representation makes the
three rhythm classes geometrically separable. This is important because
uncertainty is not only a probability problem. A confident prediction can still
sit in a mixed or atypical region of the representation space.

The project uses PCA and related projections for diagnostic visualisation. PCA
does not prove class separability by itself. Its purpose is to provide a
low-dimensional view of the embedding so that the following questions can be
inspected:

- Are SR windows separated from ventricular rhythms?
- Do VT and VF remain close or mixed?
- Are error cases concentrated around ambiguous regions?
- Do prototype or regularity interventions change the representation geometry?

The public geometry summary is:

```text
results_public/summary_tables/model_performance_and_geometry.csv
```

![Embedding distances](../results_public/figures/embedding_geometry_distances.png)

The full public projection gallery is available here:

```text
results_public/figures_compendium/
```

See the rendered image index:

```text
results_public/figures_compendium/README.md
```

Main conclusion:

VT/VF is consistently the most important boundary. SR-to-ventricular separation
is easier than VT-to-VF separation, so VT/VF cross-errors deserve separate
analysis rather than being hidden inside overall accuracy.

## 6. Uncertainty And Calibration

The uncertainty code is organised around:

```text
src/uncertainty.py
src/evaluate_uncertainty.py
src/selective_analysis.py
src/per_class_selective_analysis.py
```

Scores include:

- maximum softmax probability;
- entropy;
- temperature-scaled confidence;
- energy score;
- prototype distance;
- Mahalanobis distance;
- kNN distance;
- hybrid decision/embedding scores.

The public uncertainty table is:

```text
results_public/summary_tables/uncertainty_error_detection.csv
```

![Uncertainty error detection](../results_public/figures/uncertainty_error_detection.png)

Interpretation:

Softmax-based uncertainty is often strong for ordinary error detection, but it
does not fully describe representation-space atypicality or boundary mixing.
Embedding-based and neighbourhood-based signals provide complementary evidence.
The project therefore avoids the overclaim that a single uncertainty score is
universally best.

## 7. Full-Spectrum Shift And OOD-Style Perturbations

Clean test accuracy is not enough for a reliability study. The project evaluates
ECG-like corruptions and shift conditions, including noise, baseline changes,
masking, spikes, and amplitude changes.

Relevant modules:

```text
src/evaluate_ood.py
src/evaluate_corruption_severity.py
src/monotonicity_analysis.py
src/stability_aware_analysis.py
```

The reason for this experiment is simple: a reliability score should react when
the input becomes less trustworthy. If the classifier remains confident under
signal degradation, that is a reliability failure even when the original clean
test accuracy is strong.

The complete public shift and severity figures are included in:

```text
results_public/figures_compendium/
```

Main conclusion:

Representation-space scores such as kNN and Mahalanobis distance are useful for
detecting some corrupted or shifted ECG windows, while confidence scores are
more directly tied to decision uncertainty. Both perspectives are needed.

## 8. Regularity Features And Interpretability

The regularity branch tests whether ECG signal structure can improve reliability
analysis. Instead of relying only on neural embeddings, the project extracts
features related to rhythm and frequency regularity.

Relevant modules:

```text
src/regularity_analysis.py
src/feature_only_analysis.py
src/regularity_feature_ablation.py
src/gate_analysis.py
```

Why this matters:

- VT and VF are not just arbitrary class labels. They differ in rhythm structure.
- A reliability model should be sensitive to signal regularity, not only to
  softmax confidence.
- Feature ablation helps test whether handcrafted signal features contribute
  independent information.

Main conclusion:

Regularity features do not replace learned embeddings, but they help connect
model reliability to ECG signal structure. That makes the project stronger than
a black-box classifier-only study.

## 9. Selective Prediction And Review Routing

This is the main decision-level evaluation. The model ranks windows by a
boundary-aware reliability score. The high-risk subset is routed for review; the
remaining subset is accepted automatically.

Relevant modules:

```text
src/review_efficiency_analysis.py
src/ambiguity_routing_policy.py
src/reliability_map.py
src/runtime_supervisor.py
```

The public review-routing table is:

```text
results_public/summary_tables/review_routing_boundary_lrii.csv
```

![Review routing](../results_public/figures/review_routing_vtvf_capture.png)

Representative evidence:

- CNN-10 captures about 91.7% of VT/VF boundary errors at a 20% review budget.
- TCN-20 captures about 93.7% at the same budget.
- RegularityFusion-12 captures about 96.5% at the same budget.
- ResNet1D-12 is not automatically superior for low-budget review routing,
  even when classification performance is strong.

Main conclusion:

Reliability should be evaluated by asking whether important errors are captured
under a realistic review budget. This is more informative than reporting only
accuracy, ECE, or an uncertainty AUROC.

## 10. PRO: Boundary-Structure Mitigation

Prototype separation is evaluated as a boundary-structure intervention rather
than as a generic performance trick. The goal is to test whether representation
geometry can be modified so that boundary errors become less likely or easier to
detect.

Relevant modules:

```text
src/pro_geometry_comparison.py
src/run_core_intervention_pipeline.py
src/seedwise_statistical_summary.py
```

The public paired classification table is:

```text
results_public/summary_tables/paired_classification_comparisons.csv
```

Main conclusion:

Prototype separation shows promising improvements in the paired seed summaries,
but the seed count is small. The correct interpretation is preliminary evidence,
not final proof.

## 11. RISK: Multi-Source Reliability Evidence

The risk-head experiments distil multiple reliability signals into a lightweight
embedding-based risk predictor.

Relevant modules:

```text
src/generate_risk_targets.py
src/select_deployable_risk_weights.py
src/train_embedding_risk_head.py
src/fine_tune_risk_head.py
src/risk_head_review_analysis.py
src/evaluate_risk_corruption_robustness.py
```

The important research lesson is not that adding more signals always helps.
Additional evidence can dilute the most important VT/VF boundary signal if the
weights are not aligned with the review objective. The project therefore treats
review routing as a validation-aligned reliability problem rather than a simple
score averaging problem.

The public paired review-routing table is:

```text
results_public/summary_tables/paired_review_routing_comparisons.csv
```

## 12. Conformal And Prediction-Set Baselines

The conformal baseline tests whether uncertainty can be represented as a set
prediction rather than a forced single label. For VT/VF boundary cases, a set
such as `{VT, VF}` may be more informative than a single overconfident decision.

Relevant modules:

```text
src/conformal_analysis.py
src/conformal_review_analysis.py
```

Main conclusion:

Conformal prediction sets are useful as a review baseline, but they must be
interpreted alongside fixed-budget review capture. A method that is statistically
well-calibrated is not automatically the best operational review policy.

## 13. Public Evidence Atlas

The repository now includes a larger public evidence layer:

```text
results_public/figures/
results_public/figures_compendium/
results_public/summary_tables/
```

The compendium figure set contains the public-safe experimental images extracted
from the integrated Word report:

```text
results_public/figures_compendium/README.md
```

This includes projection galleries, uncertainty and calibration figures,
regularity feature figures, corruption severity plots, risk-head results,
prototype-separation evidence, and RISK aggregate evidence figures.

The excluded visual material is deliberate:

- doctor-review example figures;
- raw ECG waveform case galleries;
- private review examples;
- raw image archive sections;
- generated clinical-looking case materials that could be mistaken for public
  clinical evidence.

## 14. What This Project Shows

This project shows more than a classifier implementation. It shows a research
workflow:

1. define a safety-relevant failure mode;
2. control leakage at the data split level;
3. compare multiple model families;
4. inspect embedding geometry and PCA projections;
5. evaluate uncertainty and calibration;
6. stress the model under shift and perturbation;
7. evaluate review routing under fixed review budgets;
8. test mitigation and risk-distillation ideas;
9. report limits instead of overstating the evidence.

## 15. Limitations

The current evidence remains limited:

- the dataset is restricted and not externally validated here;
- paired intervention summaries use three seeds;
- synthetic corruptions do not replace external OOD datasets;
- window-level classification is not patient-level diagnosis;
- the repository is not a clinical system and does not make medical-device
  claims.

These limitations are part of the research contribution. They show where the
next stage of doctoral-level work should go: external validation, stronger
record-level statistics, broader seed counts, and clinically grounded review
protocols.

