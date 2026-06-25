# PhD Application Project Brief

## Title

Reliable ECG Classification Under Uncertainty

## One-Paragraph Summary

This project studies uncertainty-aware SR/VT/VF ECG classification with a
focus on whether high-risk VT/VF boundary errors can be detected and routed for
expert review before automated acceptance. The repository includes training,
uncertainty estimation, calibration, embedding geometry, corruption/OOD tests,
boundary analysis, duplicate-family leakage audits, and fixed-budget review
routing. The final GitHub interpretation treats RISK as the main deployable
review-priority score and treats PRO as boundary/error-migration analysis, not
as a stable standalone accuracy improvement.

## Research Motivation

High aggregate accuracy can hide clinically important boundary errors. For a
safety-adjacent classifier, the central question is whether the model knows
when it is unreliable and whether uncertain or high-risk windows can be routed
to review under realistic budget constraints.

## Method Overview

The project has six technical layers:

1. Record-level and duplicate-family split auditing.
2. Multi-backbone ECG window classification.
3. Uncertainty, calibration, and selective prediction analysis.
4. Embedding geometry and VT/VF boundary ambiguity diagnostics.
5. ECG-like corruption and OOD-style robustness tests.
6. Review-routing evaluation under fixed expert-review budgets.

## Key Evidence

| Component | Evidence |
|---|---|
| Leakage control | Duplicate-family split audit over seeds 42, 43, 44 |
| Classification baselines | CNN, TCN, ResNet1D, InceptionTime, BiGRU, fusion variants |
| Review routing | Duplicate-family RISK review-budget tables |
| Boundary analysis | VT/VF cross-error and embedding-geometry summaries |
| Negative results | Weak/inverted energy scores and unstable PRO gains are reported |
| Public evidence | Aggregate tables and figures only; raw ECG is excluded |

## Strongest Current Result

Under the final duplicate-family interpretation, the deployable RISK score is
the strongest contribution: it ranks ECG windows for expert review and captures
high-risk VT/VF errors under fixed review budgets in the internal aggregate
evidence. This supports a review-routing reliability claim, not a clinical
deployment claim.

## Limitations

- Internal dataset only; no external clinical validation.
- Raw ECG provenance and redistribution rights still need formal documentation.
- Final paired duplicate-family evidence uses three seeds.
- Window-level results should not be interpreted as patient-level diagnosis.
- The repository is a research prototype, not a medical device.

## Suitable Application Framing

> I built a reliability-aware ECG classification pipeline that goes beyond
> accuracy by auditing leakage, comparing uncertainty signals, studying VT/VF
> boundary ambiguity, testing corruption robustness, and evaluating how well a
> RISK score routes high-risk windows to expert review under fixed budgets.

## What I Would Improve Next

1. Add external validation or cross-source testing.
2. Document dataset provenance, licensing, and access conditions.
3. Increase seed count and report bootstrap confidence intervals.
4. Compare RISK against stronger conformal or calibrated review-routing
   baselines.
5. Separate patient-level and window-level claims more explicitly.
