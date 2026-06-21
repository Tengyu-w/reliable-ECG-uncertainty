# Figure Atlas

Public-safe figures are grouped by experimental stage. Each stage folder contains a contact sheet plus the individual images.

| Stage | Folder | Count | Purpose |
| --- | --- | ---: | --- |
| Summary | `00_summary/` | 4 | Main report summary figures. |
| Embedding Geometry And PCA Projections | `01_embedding_pca/` | 26 | Projection and embedding evidence used to inspect SR/VT/VF separation, especially VT/VF boundary overlap. |
| Uncertainty, Calibration, And Review Curves | `02_uncertainty_calibration/` | 10 | Uncertainty, ambiguity, selective prediction, and review-routing curves from best-model analysis. |
| ECG Regularity And Feature Interpretability | `03_regularity_interpretability/` | 6 | Regularity and feature-level evidence linking model reliability to ECG signal structure. |
| OOD And Corruption Robustness | `04_ood_corruption/` | 11 | Full-spectrum shift, severity, and ECG-like perturbation evidence. |
| Risk Head, Supervisor, And Ablation Evidence | `05_risk_supervisor_ablation/` | 13 | Risk-head, supervisor, and ablation curves for review-oriented reliability. |
| PRO Geometry And Safety Coupling | `06_pro_geometry/` | 2 | PRO geometry evidence linking embedding structure to automatic-route VT/VF residual errors. |
| PRO Severity Robustness | `07_pro_severity_robustness/` | 6 | Severity robustness validation for boundary-structure intervention experiments. |
| RISK Corruption Robustness | `08_risk_corruption_robustness/` | 3 | RISK score behavior and error capture under progressive signal degradation. |
| PRO Boundary-Structure Mitigation | `09_pro_boundary_mitigation/` | 2 | V5 reinterpretation of PRO as boundary-structure mitigation rather than a generic prototype loss. |
| V6 PRO Error Migration | `10_v6_pro_error_migration/` | 2 | V6 duplicate-family evidence showing why PRO should be interpreted cautiously as a boundary intervention. |
| V6 RISK Distillation | `11_v6_risk_distillation/` | 2 | V6 evidence for deployable RISK distillation and review-routing validation. |

## Excluded Material

Raw ECG waveforms, private review examples, doctor-review examples, risk evidence cards with sample-level information, and the raw image archive are not included in this public atlas.
