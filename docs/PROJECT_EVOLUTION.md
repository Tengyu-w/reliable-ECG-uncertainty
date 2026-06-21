# Project Evolution

This project went through six major research iterations before the current
public GitHub release. The version labels in the local reports are not perfectly
linear because early research versions and later compendium/manuscript versions
used overlapping names. The timeline below merges them into a single research
evolution.

## Iteration 1: Leakage-Safe Classification Baseline

Core question:

> Can SR, VT, and VF be classified under a record-level split without window
> leakage?

Main work:

- Built the ECG loading and 5-second windowing pipeline.
- Used record-level train/validation/test splitting.
- Trained early CNN and TCN baselines.
- Reported accuracy, macro-F1, sensitivity, specificity, and confusion matrix.

Why it mattered:

This stage established the basic classification task and showed that the
project should not be judged by overall accuracy alone. VT/VF confusion remained
important even when SR was relatively separable.

## Iteration 2: Confidence And Embedding Uncertainty

Core question:

> Do decision uncertainty and representation-space atypicality detect the same
> unreliable ECG windows?

Main work:

- Added MSP, entropy, temperature scaling, and energy scores.
- Added prototype, Mahalanobis, and kNN embedding-distance scores.
- Evaluated error-detection AUROC/AUPR.
- Added calibration and reliability-diagram analysis.

Why it mattered:

This stage moved the project from classification to uncertainty estimation. It
also produced an early negative finding: not every uncertainty score was useful.
For example, energy-based error detection was weak in the selected public
summary.

## Iteration 3: Atypicality, Boundary Ambiguity, And Review Pathways

Core question:

> Are unreliable predictions caused by generic atypicality, VT/VF boundary
> ambiguity, or signal-quality/OOD-like behaviour?

Main work:

- Added VT/VF ambiguity analysis.
- Added embedding geometry and PCA/2D/3D projection evidence.
- Added conformal prediction-set analysis.
- Added selective prediction and review-routing pathways.

Why it mattered:

This stage separated different failure modes. It became clear that VT/VF
boundary ambiguity should be evaluated directly rather than hidden inside
overall error rates.

## Iteration 4: ECG Regularity And Interpretability

Core question:

> Can ECG rhythm/frequency regularity features help explain where the model is
> unreliable?

Main work:

- Extracted regularity features such as spectral entropy, dominant frequency,
  bandwidth, autocorrelation, line length, and sample entropy.
- Added feature-only analysis and feature-group ablations.
- Added regularity-fusion and reliability-gated fusion models.
- Analysed whether gates and handcrafted features aligned with boundary or
  atypicality evidence.

Why it mattered:

This stage connected black-box representation behaviour to ECG signal structure.
It strengthened the research story because the project was no longer only a
neural classifier plus uncertainty scores.

## Iteration 5: Boundary-Aware Mitigation, PRO, RISK, And Supervisor Logic

Core question:

> Can identified reliability failures be mitigated or converted into a useful
> review-routing policy?

Main work:

- Added prototype-separation / PRO-style boundary-structure intervention.
- Added boundary-aware review routing and supervisor-style states.
- Added risk targets and embedding risk-head training.
- Added risk-head review analysis and risk corruption robustness.
- Added paired multi-seed summaries for classification and automatic-route
  behaviour.

Why it mattered:

This stage turned reliability analysis into intervention and routing. It also
produced important mixed results: multi-source risk evidence was not
automatically better, and some paired confidence intervals still crossed zero.

## Iteration 6: Deployable Corrections, Duplicate-Family Audit, And Stronger Evidence Discipline

Core question:

> Which reliability evidence is deployable, and which evidence accidentally uses
> information that would not be available at inference time?

Main work:

- Corrected deployable risk definitions so query true labels were not used to
  construct risk scores.
- Added validation-selected risk weighting.
- Added record-balanced and duplicate-family split audits.
- Added duplicate-family stage reporting.
- Separated oracle-style diagnostic evidence from deployable evidence.

Why it mattered:

This was the most research-mature iteration. It corrected earlier assumptions,
made leakage and deployment boundaries explicit, and prevented overclaiming.

## Current Public Release: GitHub Research Portfolio

Core question:

> How can the project be shown publicly for PhD applications without exposing
> restricted ECG data or private review material?

Main work:

- Created a clean public Git history.
- Kept core source code and method documentation.
- Added public-safe aggregate figures and tables.
- Added a complete experiment compendium.
- Explicitly documented negative results and limitations.

Why it matters:

This release is not another experimental method iteration. It is a communication
and research-portfolio iteration: it makes the work readable to potential
supervisors while preserving data boundaries.

## Summary Count

The project has six substantive research iterations:

1. leakage-safe classification baseline;
2. confidence and embedding uncertainty;
3. atypicality, VT/VF boundary ambiguity, and review pathways;
4. ECG regularity and interpretability;
5. boundary-aware mitigation, PRO/RISK, and supervisor logic;
6. deployable corrections, duplicate-family audit, and stronger evidence
   discipline.

The current GitHub version is best described as a seventh communication stage,
not a seventh scientific method iteration.

