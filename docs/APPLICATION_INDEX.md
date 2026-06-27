# Application Index

This page is the supervisor-facing entry point for the repository. It frames
the project for PhD applications in trustworthy medical machine learning,
uncertainty estimation, calibration, selective prediction, and human-in-the-loop
review routing.

## Thirty-Second Summary

This project studies whether an ECG classifier can recognize when its own
short-window SR/VT/VF prediction is unreliable. The main contribution is not
ordinary classification accuracy. The final method uses a multi-source
reliability evidence layer, including RISK, and converts it into `v5d`, a
mechanism-separated hierarchical routing policy for high-risk review under
fixed action budgets.

## Best Application Framing

Use this project when writing to supervisors in:

- trustworthy ML and reliable decision systems;
- medical AI reliability and uncertainty estimation;
- calibration, selective prediction, and OOD robustness;
- human-AI teaming and review-routing workflows;
- representation analysis for safety-adjacent time-series models.

Suggested wording:

> I built a reliability-aware ECG classification pipeline that goes beyond
> accuracy by auditing leakage, comparing uncertainty signals, diagnosing VT/VF
> representation-boundary failures, testing structured model interventions,
> and converting reliability evidence into a mechanism-separated review-routing
> policy.

## What To Read

| Time budget | File | Purpose |
| --- | --- | --- |
| 2 minutes | [README](../README.md) | Main contribution, result snapshot, and limitations. |
| 5 minutes | [PhD application brief](phd_application_project_brief.md) | Concise supervisor-facing project narrative. |
| 10 minutes | [Experiment evidence summary](EXPERIMENT_EVIDENCE_SUMMARY.md) | Why each experiment was done and what it showed. |
| 15 minutes | [Evidence index](evidence_index.md) | Claim-by-claim public evidence map. |
| 30 minutes | [Research report](RESEARCH_REPORT.md) | Detailed stage-ordered research report. |
| Figures | [Figure atlas](FIGURE_ATLAS.md) | Public visual evidence index. |

## Evidence Snapshot

| Claim | Current evidence | Strength |
| --- | --- | --- |
| Data leakage risk is explicitly audited. | Record-level and duplicate-family split audits report no overlap in the final audited split summaries. | Strong for internal protocol. |
| Accuracy alone is insufficient. | Backbone comparison reports accuracy, macro-F1, calibration, uncertainty, representation geometry, and review-routing behavior. | Strong framing evidence. |
| VT/VF is the central fragile boundary. | Embedding geometry, local neighborhood mixing, and boundary-error analyses show VT/VF ambiguity. | Strong internal diagnostic evidence. |
| Model-side structure is useful but not sufficient. | CNN-LSTM, PRO, ProRisk/Risk-Pro-readable, CNN-TCN-Validity, and wavelet variants reveal improvements, failures, and error migration. | Strong mechanistic evidence, not a final classifier claim. |
| Embedding evidence is used carefully. | Representation analysis explains failure mechanisms, but intervention results show that embedding improvement alone does not guarantee safer VT/VF decisions. | Important negative-result discipline. |
| RISK is a useful evidence score. | Duplicate-family RISK tables show strong fixed-budget review capture, especially for VT/VF boundary errors. | Promising internal evidence. |
| v5d is the final decision policy. | Ten paired duplicate-family splits show improved VT/VF cross-error capture and lower unresolved VT/VF rate versus v4 routing. | Strongest current internal method result. |
| Dataset-size concerns are stress-tested. | Validation downsampling and cluster-concentration audits check whether routing gains are only one-split or one-cluster artifacts. | Useful internal robustness evidence, not external validation. |

## What Is Shown

- A PyTorch pipeline for ECG classification, uncertainty, calibration,
  OOD/corruption testing, representation analysis, and review routing.
- A progression from baseline models to mechanism-guided routing.
- Public-safe aggregate figures and tables without raw ECG data.
- Explicit negative and mixed results, including error migration.
- A final reliability policy that separates VT/VF boundary review from
  residual error mechanisms.

## What Remains Unproven

- External clinical validation.
- Patient-level diagnosis or deployment readiness.
- Public redistribution rights for the raw ECG dataset.
- Generalization to other ECG datasets, devices, hospitals, or populations.
- Replacement of clinician judgment.

## Best-Fit Supervisor Directions

| Direction | How to position the project |
| --- | --- |
| Trustworthy ML | Reliability evidence, negative results, selective prediction, and fixed-budget routing. |
| Medical AI | Review routing for high-risk VT/VF boundary cases. |
| Uncertainty and calibration | Confidence, entropy, ECE, coverage-risk curves, and conformal baselines. |
| Robustness | ECG-like corruption, degradation sensitivity, and OOD-style tests. |
| Representation learning | Embedding geometry, kNN mixing, prototype conflict, and layerwise diagnostics. |
| Human-AI teaming | Mechanism-separated routing rather than unconditional automation. |
