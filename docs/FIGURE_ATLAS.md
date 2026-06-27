# Figure Atlas

This document explains how the public figures are organized. The goal is to
make the visual evidence readable as a research story rather than as a loose
folder of PNG files.

All figures in this atlas are public-safe aggregate or method-evidence figures.
Raw ECG waveform galleries, private review examples, doctor-review examples,
and sample-level risk evidence cards are intentionally excluded.

Some figure groups preserve historical V3/V4/V5 evidence. For final GitHub
claims, read those historical groups together with `10_v6_pro_error_migration`
and `11_v6_risk_distillation`, which contain the stricter duplicate-family
interpretation.

## Figure Groups

| Stage | Folder | What The Reader Should Look For |
| --- | --- | --- |
| Summary | `results_public/figures/00_summary/` | Four high-level figures used by the main report: model performance, embedding distances, uncertainty error detection, and review routing. |
| Embedding geometry and PCA | `results_public/figures/01_embedding_pca/` | Whether SR separates more cleanly than VT/VF, whether errors cluster near mixed regions, and why PCA is used as interpretation rather than proof. |
| Uncertainty and calibration | `results_public/figures/02_uncertainty_calibration/` | Whether confidence, entropy, and review curves identify unreliable predictions. |
| Regularity interpretability | `results_public/figures/03_regularity_interpretability/` | Whether ECG rhythm/frequency regularity helps explain difficult VT/VF cases. |
| OOD and corruption | `results_public/figures/04_ood_corruption/` | How performance and uncertainty behave under ECG-like perturbations. |
| Risk head and supervisor | `results_public/figures/05_risk_supervisor_ablation/` | How review-routing policies and risk-head variants behave under ablations. |
| PRO geometry | `results_public/figures/06_pro_geometry/` | How PRO changes representation geometry and why that matters for VT/VF boundary safety. |
| PRO severity robustness | `results_public/figures/07_pro_severity_robustness/` | How boundary intervention behaves under progressive severity tests. |
| RISK corruption robustness | `results_public/figures/08_risk_corruption_robustness/` | Whether RISK scores increase under signal degradation and whether review capture persists. |
| PRO boundary mitigation | `results_public/figures/09_pro_boundary_mitigation/` | The V5 interpretation of PRO as boundary-structure mitigation rather than a generic loss term. |
| V6 PRO error migration | `results_public/figures/10_v6_pro_error_migration/` | Why the final report treats PRO cautiously under stricter duplicate-family evidence. |
| V6 RISK distillation | `results_public/figures/11_v6_risk_distillation/` | RISK as the reliability evidence layer and review-budget evidence. |
| V5d hierarchical router | `results_public/figures/12_v5d_hierarchical_router/` | Final decision-policy evidence: v5d budget curves, residual VT/VF rate, stage allocation, and method diagram. |
| Frozen self-supervised encoder | `results_public/figures/13_frozen_ssl_encoder/` | Lightweight frozen encoder baseline showing classification limits but strong risk-ranking behavior. |
| Explanation reliability | `results_public/figures/14_explanation_reliability/` | Quantitative alignment between explanation evidence families and intended error mechanisms. |

Each folder contains:

- `contact_sheet.png`: a compact overview of all figures in that group;
- `individual_figures/`: the underlying figures;
- `README.md`: a short local index.

## Recommended Reading Order

1. Start with `00_summary/contact_sheet.png`.
2. Open `01_embedding_pca/contact_sheet.png` to understand why VT/VF boundary
   structure became the central problem.
3. Compare `02_uncertainty_calibration/` and `04_ood_corruption/` to see why
   uncertainty, calibration, and signal shift are separate reliability views.
4. Read `06_pro_geometry/`, `09_pro_boundary_mitigation/`, and
   `10_v6_pro_error_migration/` together. These show why PRO is useful but not
   overclaimed.
5. Finish with `05_risk_supervisor_ablation/`, `08_risk_corruption_robustness/`,
   and `11_v6_risk_distillation/`, which support the RISK evidence layer.
6. Finish with `12_v5d_hierarchical_router/`, then check
   `13_frozen_ssl_encoder/` and `14_explanation_reliability/` for the new
   final-method, frozen-encoder, and explanation-reliability evidence.

## Interpretation Notes

PCA and projection figures should not be read as statistical proof that classes
are separable. They are diagnostic views of learned embeddings. Their value is
to show that the main failure mode has structure: VT and VF can be locally
mixed, errors often occur near those regions, and boundary-aware review routing
is therefore a justified research direction.

The public atlas is intentionally more compact than the local experiment
archive. It prioritizes figures that communicate the research logic without
exposing raw or sample-level ECG material.
