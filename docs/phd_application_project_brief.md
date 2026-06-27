# PhD Application Project Brief

## Title

Reliable ECG Classification Under Uncertainty

## One-Paragraph Summary

This project studies reliability-aware short-window SR/VT/VF ECG
classification. It begins with standard neural classifiers, shows that the
VT/VF boundary is the fragile region, tests multiple architectures and
structured interventions, and then converts uncertainty, representation,
regularity, validity, and wavelet evidence into a mechanism-separated
hierarchical review-routing policy. In the final framing, RISK is the evidence
score and `v5d` is the decision policy.

## Research Motivation

High aggregate accuracy can hide safety-relevant boundary errors. For a
medical-AI reliability prototype, the important question is not only whether
the classifier can output a label, but whether it can identify predictions
that should not be automatically accepted.

## Method Overview

The project has eight technical layers:

1. Record-level and duplicate-family leakage auditing.
2. Multi-backbone ECG window classification.
3. Embedding geometry and VT/VF boundary diagnostics.
4. Uncertainty, calibration, conformal, and selective-prediction analysis.
5. ECG regularity, corruption, OOD-style, validity, and wavelet evidence.
6. Structured model interventions: CNN-LSTM, PRO, ProRisk/Risk-Pro-readable,
   CNN-TCN-Validity, and CNN-Wavelet-TCN variants.
7. RISK evidence scoring for review prioritization.
8. v5d mechanism-separated hierarchical routing.

## Key Evidence

| Component | Evidence |
| --- | --- |
| Leakage control | Record-level and duplicate-family audit summaries. |
| Backbone baselines | CNN, TCN, ResNet1D, InceptionTime, BiGRU, RegularityFusion, GatedFusion. |
| Boundary diagnosis | PCA, class-center distance, kNN mixing, prototype conflict, layerwise analysis. |
| Negative result | Representation separation does not guarantee safer VT/VF decisions. |
| Structured interventions | PRO, ProRisk/Risk-Pro-readable, CNN-LSTM, CNN-TCN-Validity, wavelet variants. |
| Final policy | v5d boundary-first routing plus residual mechanism budget. |
| Stress test | Validation downsampling and cluster-concentration audits reduce small-sample-artifact concerns. |
| Public evidence | Aggregate tables and figures only; raw ECG is excluded. |

## Strongest Current Result

Across ten paired duplicate-family splits, v5d with a 20% residual-budget
reserve achieved 86.0% all-error capture and 99.0% VT/VF cross-error capture
at a 20% action budget, with an automatic unresolved VT/VF rate of 0.07%.

This supports an internal review-routing reliability claim. It is not clinical
validation.

## Suitable Application Framing

> I built a reliability-aware ECG classification pipeline that diagnoses
> VT/VF boundary failures and turns multi-source reliability evidence into a
> mechanism-separated review-routing policy, rather than relying only on
> aggregate accuracy or a single uncertainty score.

## Limitations

- Internal dataset only; no external clinical validation.
- Raw ECG data, model weights, embeddings, and private sample-level material
  are excluded from GitHub.
- Window-level classification is not patient-level diagnosis.
- Some structured models improved representation metrics without improving
  safety-relevant VT/VF behavior.
- The repository is a research prototype, not a medical device.
