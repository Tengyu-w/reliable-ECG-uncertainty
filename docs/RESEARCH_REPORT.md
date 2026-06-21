# Research Report

Reliable ECG Classification Under Uncertainty

This report summarizes the public, research-safe version of the project. It is
written for a reader who has not seen the local experiment archive and needs to
understand the full experimental logic: why the project was designed, what was
tested, what improved, what failed, and what remains unproven.

The project is a research prototype only. It is not a diagnostic system, not a
medical device, and not evidence of clinical validation.

## 1. Research Question

The task is short-window ECG classification into three rhythm classes:

- `SR`: sinus or non-ventricular rhythm;
- `VT`: ventricular tachycardia;
- `VF`: ventricular fibrillation.

The central research question is not only whether a neural network can classify
these windows accurately. The more important question is:

> Can the model recognize when its own SR/VT/VF prediction is unreliable, and
> can high-risk VT/VF boundary cases be routed to expert review under a fixed
> review budget?

This framing changes the project from ordinary classification into
review-oriented reliability analysis. A model can have high overall accuracy
while still making dangerous VT/VF boundary errors. The project therefore
evaluates accuracy, calibration, uncertainty, embedding geometry, signal
regularity, corruption robustness, and review-routing behavior together.

## 2. Experimental Roadmap

The experiments are organized into nine stages.

| Stage | Purpose | Main evidence |
| --- | --- | --- |
| 1. Data protocol and leakage audit | Make the split defensible before trusting metrics. | Record-level split, duplicate-family audit, public split table. |
| 2. Backbone training | Establish classification baselines across model families. | CNN, TCN, ResNet1D, InceptionTime, BiGRU, fusion variants. |
| 3. Embedding and PCA analysis | Inspect whether VT/VF errors have representation structure. | PCA, LDA, 3D projections, normalized class distances. |
| 4. Uncertainty and selective prediction | Test whether confidence scores detect errors and support abstention. | MSP, entropy, temperature-scaled confidence, energy, coverage-risk curves. |
| 5. OOD and corruption robustness | Test whether reliability scores respond to ECG-like signal degradation. | Noise, baseline drift, masking, spikes, amplitude and mixed corruptions. |
| 6. Regularity interpretability | Connect reliability to ECG signal structure. | Rhythm/frequency features and regularity ablations. |
| 7. PRO boundary intervention | Test whether representation structure can be changed. | Prototype/center separation and automatic-route VT/VF residual errors. |
| 8. RISK review scoring | Distil multi-source reliability evidence into a deployable review score. | Risk targets, risk head, review capture, corruption behavior. |
| 9. V6 evidence discipline | Make the final claims more conservative and defensible. | Duplicate-family split, six-direction error analysis, record-cluster bootstrap. |

## 3. Stage 1: Data Protocol And Leakage Control

The code expects a local `RHYTHMS.mat` file containing SR, VT, and VF ECG
records. The raw data are not redistributed in this repository.

The split is record-level rather than window-level. This matters because
adjacent windows from the same ECG record can be highly correlated. A
window-level split would risk train-test leakage and would make reliability
metrics look stronger than they really are.

The later V6 upgrade adds a stricter duplicate-family perspective. The purpose
is to handle the possibility that identical or near-identical ECG windows can
connect different records. Instead of treating those records as independent,
duplicate-family grouping keeps connected records inside the same split group.
This makes the final interpretation more conservative.

Public evidence:

- `results_public/tables/dataset_split_statistics.csv`
- `src/audit_data_protocol.py`
- `src/audit_duplicate_family_splits.py`
- `src/duplicate_leakage_sensitivity.py`

## 4. Stage 2: Backbone Training And Classification Baselines

The first modeling stage compares several time-series architectures rather than
depending on one backbone. This is important because a reliability conclusion is
weak if it only appears for one model family.

Public summary:

![Model performance summary](../results_public/figures/00_summary/model_performance_summary.png)

Selected aggregate results:

| Model | Accuracy | Macro-F1 | ECE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| CNN-10 | 92.4% | 71.6% | 1.6% | Strong baseline and useful review-routing behavior. |
| TCN-20 | 88.6% | 66.0% | 2.3% | Lower accuracy, but strong VT/VF review capture. |
| ResNet1D-12 | 92.4% | 67.3% | 6.0% | Competitive classifier, weaker under review-budget routing. |
| InceptionTime-12 | 93.9% | 74.1% | 1.4% | Strong classification baseline. |
| BiGRU-12 | 81.9% | 53.4% | 9.5% | Weaker baseline in this setup. |
| RegularityFusion-12 | 91.6% | 69.1% | 3.0% | Useful bridge between signal features and reliability. |
| GatedFusion-12 | 94.9% | 77.5% | 2.9% | Best public aggregate classifier, but accuracy alone is not the final objective. |

Main conclusion: classification quality is necessary, but it is not enough. The
project continues by asking whether the models know when not to trust
themselves.

## 5. Stage 3: Embedding Geometry, PCA, And VT/VF Boundary Analysis

The embedding analysis asks whether the learned feature space explains why some
errors occur. PCA and related projections are used as diagnostic tools, not as
proof of separability.

The key question is:

> Are VT and VF locally mixed in the model's representation space, and do
> boundary errors concentrate in those mixed regions?

The answer is important because it motivates later stages. If the error is
structural, then the project should not only tune hyperparameters. It should
study boundary-aware reliability signals, representation interventions, and
review routing.

Public summary:

![Embedding geometry distances](../results_public/figures/00_summary/embedding_geometry_distances.png)

Full public PCA/projection evidence:

- `results_public/figures/01_embedding_pca/contact_sheet.png`
- `results_public/figures/01_embedding_pca/`

Interpretation:

- SR is generally easier to separate from ventricular rhythms than VT and VF
  are from each other.
- VT/VF ambiguity is not just a metric artifact. It appears in local
  representation structure.
- PCA and 3D projections support the research narrative, but they should be
  interpreted as explanatory evidence rather than standalone statistical proof.

## 6. Stage 4: Uncertainty, Calibration, Selective Prediction, And Conformal Baselines

This stage evaluates whether confidence and uncertainty scores identify model
errors. It compares softmax confidence, entropy, temperature-scaled confidence,
energy score, embedding distance, local neighborhood evidence, and conformal
prediction sets.

Public summary:

![Uncertainty error detection](../results_public/figures/00_summary/uncertainty_error_detection.png)

Representative error-detection AUROC values:

| Model | MSP AUROC | Entropy AUROC | Energy AUROC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| CNN-10 | 0.902 | 0.899 | 0.100 | Confidence scores work; energy is weak/inverted here. |
| TCN-20 | 0.947 | 0.949 | 0.052 | Strong softmax uncertainty; energy fails. |
| ResNet1D-12 | 0.892 | 0.893 | 0.112 | Useful confidence signal with weaker calibration profile. |
| RegularityFusion-12 | 0.932 | 0.930 | 0.070 | Strong uncertainty ranking. |
| GatedFusion-12 | 0.846 | 0.847 | 0.150 | Strong classifier, but not the strongest uncertainty ranker. |

Main conclusion: uncertainty is not one thing. Softmax-based scores can be
strong for ordinary error detection, while embedding and neighborhood scores
help interpret atypicality and boundary mixing. Energy score is a clear
negative result in these summaries.

Conformal prediction is included as a baseline for set-valued prediction. For
VT/VF ambiguity, a set such as `{VT, VF}` can be more honest than forcing one
label. However, conformal validity and fixed-budget review capture answer
different questions, so both are reported.

## 7. Stage 5: OOD And Corruption Robustness

Clean test metrics do not show what happens when ECG quality degrades. This
stage applies ECG-like perturbations such as noise, baseline drift, masking,
spikes, amplitude changes, clipping, and mixed degradation.

Public evidence:

- `results_public/figures/04_ood_corruption/contact_sheet.png`
- `results_public/figures/08_risk_corruption_robustness/contact_sheet.png`
- `src/evaluate_ood.py`
- `src/evaluate_corruption_severity.py`
- `src/evaluate_risk_corruption_robustness.py`

The V5 upgrade adds a dedicated RISK corruption experiment. The model obtains
embeddings from progressively corrupted ECG windows, the RISK head produces a
review score, and the analysis asks three questions:

1. does the RISK score rise as degradation severity increases?
2. does RISK still rank corrupted prediction errors?
3. how many errors are captured at 10%, 20%, and 30% review budgets?

Main conclusion: RISK is degradation-sensitive under many corruptions, but
severe clipping, strong noise, and mixed degradation remain difficult. The
project therefore does not claim solved OOD robustness.

## 8. Stage 6: ECG Regularity And Interpretability

The regularity branch asks whether signal-level rhythm structure helps explain
model reliability. VT and VF are not arbitrary labels; they differ in rhythm
regularity and signal morphology. The project therefore evaluates features such
as spectral entropy, dominant frequency, autocorrelation structure, and related
regularity descriptors.

Public evidence:

- `results_public/figures/03_regularity_interpretability/contact_sheet.png`
- `src/regularity_analysis.py`
- `src/regularity_feature_ablation.py`
- `src/feature_only_analysis.py`

Main conclusion: regularity features do not replace learned embeddings, but
they make the reliability story more interpretable. They help explain why some
short windows are unstable: the signal itself may be morphologically ambiguous,
irregular, degraded, or locally atypical.

## 9. Stage 7: PRO As Boundary-Structure Intervention

PRO was introduced to test whether the VT/VF embedding boundary can be improved
through prototype or center-separation style intervention.

The mature interpretation is careful:

- In earlier evidence, PRO reduced automatic-route VT/VF errors and improved
  some representation-geometry summaries.
- V5 reframed PRO as boundary-structure mitigation, not merely a prototype
  loss.
- V6 made the conclusion more conservative: under stricter duplicate-family
  evidence, PRO can expose error migration. It may reduce one error direction
  while increasing another.

Public evidence:

- `results_public/figures/06_pro_geometry/contact_sheet.png`
- `results_public/figures/09_pro_boundary_mitigation/contact_sheet.png`
- `results_public/figures/10_v6_pro_error_migration/contact_sheet.png`
- `results_public/tables/paired_classification_comparisons.csv`
- `results_public/tables/paired_review_routing_comparisons.csv`

Paired three-seed summary for prototype separation:

| Metric | Baseline mean | PRO mean | Mean difference | 95% CI | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| Accuracy | 0.892 | 0.929 | +0.037 | -0.016 to 0.091 | Promising, but CI crosses zero. |
| Macro-F1 | 0.624 | 0.669 | +0.044 | 0.024 to 0.064 | Consistent positive signal across three seeds. |
| VT/VF cross-errors | 194.7 | 179.3 | -15.3 | -40.8 to 10.2 | Directionally useful, not definitive. |
| Automatic-route VT/VF cross-errors | 41.3 | 13.3 | -28.0 | -71.0 to 15.0 | Safety-relevant improvement, but still small-n evidence. |

Main conclusion: PRO is best presented as an analysis contribution and a
boundary intervention study. It should not be overclaimed as the final robust
solution.

## 10. Stage 8: RISK As Reliability-Privileged Knowledge Distillation

RISK is the strongest final contribution because it directly matches the
review-routing objective.

The idea is to use rich reliability evidence during training or analysis:

- entropy;
- local instability;
- VT/VF neighborhood mixing;
- KNN atypicality;
- softmax VT/VF ambiguity;
- embedding-neighborhood evidence.

These signals are distilled into a lightweight risk head. At deployment time,
the classifier embedding is sufficient to produce a review-priority score.

This is why the project frames RISK as reliability-privileged knowledge
distillation rather than as another ordinary classifier.

Public review-routing evidence:

![Review routing summary](../results_public/figures/00_summary/review_routing_vtvf_capture.png)

VT/VF boundary error capture at 20% review burden:

| Model | VT/VF error captured | All error captured | Automatic-route error rate |
| --- | ---: | ---: | ---: |
| CNN-10 | 91.7% | 73.4% | 2.54% |
| TCN-20 | 93.7% | 79.9% | 2.87% |
| ResNet1D-12 | 71.4% | 73.2% | 2.54% |
| RegularityFusion-12 | 96.5% | 68.4% | 3.31% |

Public RISK evidence:

- `results_public/figures/05_risk_supervisor_ablation/contact_sheet.png`
- `results_public/figures/11_v6_risk_distillation/contact_sheet.png`
- `src/generate_risk_targets.py`
- `src/select_deployable_risk_weights.py`
- `src/train_embedding_risk_head.py`
- `src/risk_head_review_analysis.py`

Main conclusion: RISK is not a diagnosis. It is a review-priority signal. The
research value is that it connects multiple reliability mechanisms to a
decision policy: automatic accept, boundary review, signal-quality review, or
forced expert review.

## 11. Stage 9: V5 To V6 Upgrade Before Public Release

The V5 to V6 upgrade changed the project in several important ways.

First, PCA and embedding analysis became more explicitly interpretive. The
projection figures are not decorative. They explain why VT/VF ambiguity is a
structured boundary problem and why boundary-aware methods were tested.

Second, duplicate-family split logic made the evidence stricter. This reduced
the risk that repeated or near-repeated ECG windows inflated the apparent
generalization quality.

Third, PRO was reframed. Instead of presenting it as a clean success, V6 treats
PRO as a useful boundary intervention that can also reveal error migration.

Fourth, RISK became the main defensible contribution. It is better aligned with
the project goal because it ranks cases for review rather than trying to make
every short ECG window safe for automatic acceptance.

Fifth, V6 adds stronger evidence discipline: six-direction error analysis,
record-cluster bootstrap, fixed review budgets, and clearer statements about
what the project does not prove.

## 12. Negative Results And Limitations

The negative and mixed results are part of the research contribution.

Important negative findings:

- A stronger classifier does not automatically give better review routing.
  ResNet1D-12 is competitive as a classifier but weaker for VT/VF review
  capture under constrained budgets.
- Energy score is weak or inverted as an error detector in the public
  summaries.
- PRO improves some paired summaries, but the evidence is small-n and becomes
  more complex under stricter duplicate-family analysis.
- RISK is degradation-sensitive, but severe-corruption error ranking is not
  uniformly robust.
- Three paired seeds are useful but not enough for strong statistical claims.

Limitations:

- The dataset is restricted and not externally validated here.
- The public repository excludes raw ECG signals and sample-level evidence.
- Synthetic corruption does not replace external OOD validation.
- Window-level classification is not patient-level diagnosis.
- The work is a research prototype, not a clinical system.

## 13. Public Repository Evidence

The GitHub repository is organized to expose the research logic without
redistributing restricted biomedical material.

| Evidence type | Location |
| --- | --- |
| Full stage-ordered report | `docs/RESEARCH_REPORT.md` |
| Experiment pipeline | `docs/EXPERIMENT_PIPELINE.md` |
| Figure atlas | `docs/FIGURE_ATLAS.md` and `results_public/figures/` |
| Public tables | `results_public/tables/` |
| Code map | `src/README.md` |
| Version history | `docs/PROJECT_EVOLUTION.md` |
| Word-compendium coverage audit | `docs/WORD_COMPENDIUM_COVERAGE_AUDIT.md` |

The repository intentionally excludes raw ECG records, private review examples,
doctor-review figures, model checkpoints, embeddings, logits, probabilities,
window-level prediction files, and raw waveform galleries.

## 14. Final Research Positioning

The safest final positioning is:

> This project studies review-oriented reliability supervision for short-window
> SR/VT/VF ECG classification. It shows that VT/VF boundary errors have
> representation and uncertainty structure, that ordinary accuracy is
> insufficient for safety-relevant evaluation, and that multi-source reliability
> evidence can be distilled into a lightweight review-priority score. The work
> remains a research prototype and requires external validation before any
> clinical interpretation.
