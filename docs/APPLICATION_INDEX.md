# Application Index

This page is the supervisor-facing entry point for the repository. It frames
the project for PhD applications in trustworthy medical machine learning,
uncertainty estimation, calibration, selective prediction, and safety-adjacent
AI review routing.

## Thirty-Second Summary

This project studies whether an ECG classifier can recognize when its own
SR/VT/VF prediction is unreliable. The strongest current claim is not ordinary
classification accuracy; it is review-routing reliability under a stricter
duplicate-family split, where a RISK score ranks high-risk windows for expert
review under fixed review budgets.

## Best Application Framing

Use this project when writing to supervisors in trustworthy ML, medical AI
reliability, uncertainty estimation, calibration, OOD/shift robustness, or
human-in-the-loop review systems.

Suggested wording:

> I built a reliability-aware ECG classification pipeline that evaluates
> uncertainty, embedding geometry, calibration, corruption robustness, and
> review routing for SR/VT/VF windows. The project emphasizes leakage audits,
> negative results, duplicate-family splitting, and fixed-budget capture of
> high-risk VT/VF boundary errors.

## What To Read

| Time budget | File | Purpose |
|---|---|---|
| 2 minutes | [README](../README.md) | Main question, public evidence, limitations |
| 5 minutes | [PhD application brief](phd_application_project_brief.md) | Concise supervisor-facing project narrative |
| 10 minutes | [Evidence index](evidence_index.md) | Claim-by-claim evidence map |
| 20 minutes | [Research report](RESEARCH_REPORT.md) | Stage-ordered experiment narrative |
| Deep dive | [Complete compendium](COMPLETE_EXPERIMENT_COMPENDIUM.md) | Full experiment history and interpretation |

## Evidence Snapshot

| Claim | Current evidence | Strength |
|---|---|---|
| Data leakage risk is explicitly audited. | Record-level and duplicate-family split audits report zero overlap for seeds 42, 43, 44. | Strong for internal protocol |
| Accuracy alone is insufficient. | Backbone comparison reports accuracy, macro-F1, ECE, uncertainty, and review-routing views. | Strong framing evidence |
| RISK is useful for review routing. | Duplicate-family selected RISK tables show fixed-budget VT/VF error capture up to 1.0 at >=20% review burden. | Promising internal evidence |
| Uncertainty methods vary in usefulness. | Softmax scores are useful in summaries; energy score is weak/inverted in selected results. | Good negative-result reporting |
| PRO is not overclaimed. | Duplicate-family evidence treats PRO as boundary/error-migration analysis, not stable standalone gain. | Strong claim discipline |

## What Is Shown

- A complete PyTorch research pipeline for ECG classification, uncertainty,
  calibration, OOD/corruption testing, embedding analysis, and review routing.
- A stricter duplicate-family split interpretation to reduce leakage risk.
- Public-safe aggregate figures and tables without raw ECG data.
- Explicit negative and mixed results.
- A review-routing perspective that asks whether uncertainty captures important
  errors, not merely whether accuracy improves.

## What Remains Unproven

- External clinical validation.
- Patient-level diagnosis or deployment readiness.
- Public redistribution rights for the raw ECG dataset.
- More than three paired seeds for the final duplicate-family comparison.
- Generalization to other ECG datasets, devices, hospitals, or populations.

## Best-Fit Supervisor Directions

| Direction | How to position the project |
|---|---|
| Trustworthy ML | Reliability scores, leakage audits, negative results, selective prediction. |
| Medical AI | Expert-review routing for high-risk VT/VF boundary errors. |
| Calibration / uncertainty | Confidence, entropy, temperature scaling, ECE, coverage-risk curves. |
| OOD robustness | ECG-like corruptions and severity analysis. |
| Human-AI teaming | Fixed review-budget capture and review burden tradeoffs. |

## Next Upgrade Before Submission

1. Confirm that the duplicate-family/V6 evidence tables are visible on GitHub
   before sharing the repository link.
2. Add a clear dataset provenance statement before any public archival release.
3. Add an external dataset or cross-source validation if available.
4. Report confidence intervals or bootstrap summaries alongside the three-seed
   paired results.
5. Keep all wording explicit that this is research evidence, not clinical
   validation.
