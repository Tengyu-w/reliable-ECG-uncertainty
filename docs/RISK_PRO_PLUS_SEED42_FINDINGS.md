# RISK-PRO++ Seed 42 Pilot Findings

Date: 2026-06-25

This note records the first complete seed-42 pilot for the risk-aware intervention pipeline. It is research evidence only; it is not clinical validation and should not be described as a medical-device result.

## Question

Can a stronger training objective that combines embedding geometry, KNN neighbourhood risk, prototype distance, softmax uncertainty, local instability, VT/VF mixing, perturbation consistency, and regularity supervision improve reliable SR/VT/VF classification beyond the current teacher and the earlier PRO centre-distance objective?

## Experimental Setup

- Split grouping: `duplicate_family`
- Seed: `42`
- Main comparison:
  - `teacher`: reliability-gated fusion baseline
  - `PRO_center_margin`: prototype centre/margin objective only
  - `boundary_contrastive`: boundary-aware contrastive objective only
  - `RISK_PRO_plus`: combined multi-signal risk-aware objective
- Core analyses run: uncertainty metrics, embedding geometry, selective prediction, VT/VF ambiguity, conformal sets, reliability map, review efficiency, regularity features, and per-class selective analysis.
- Corruption, full boundary, and stability analyses were skipped in the resume command for speed, except where earlier teacher outputs already existed.

## Main Evidence

| Model | Accuracy | Macro-F1 | ECE | VT/VF Cross Errors | Total Errors | VT/VF Centre Distance | VT/VF Mixing | Entropy Error AUROC | Temp-MSP Error AUROC | KNN Error AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher | 0.924 | 0.762 | 0.044 | 193 | 319 | 0.680 | 0.210 | 0.859 | 0.861 | 0.706 |
| PRO_center_margin | 0.907 | 0.714 | 0.074 | 281 | 388 | 0.761 | 0.224 | 0.870 | 0.875 | 0.779 |
| boundary_contrastive | 0.897 | 0.681 | 0.095 | 342 | 431 | 0.981 | 0.248 | 0.919 | 0.920 | 0.867 |
| RISK_PRO_plus | 0.902 | 0.706 | 0.078 | 289 | 410 | 0.866 | 0.228 | 0.751 | 0.745 | 0.798 |

## Interpretation

The intervention objectives successfully changed the embedding geometry: PRO increased VT/VF centre distance from 0.680 to 0.761, contrastive increased it to 0.981, and RISK-PRO++ increased it to 0.866. However, the improved geometric separation did not translate into better classification reliability. All three intervention models reduced macro-F1 and increased VT/VF cross-errors relative to the teacher.

This is an important negative result. It suggests that centre-distance separation is not sufficient as a training target for this dataset. The model may learn embeddings that are easier to rank or review, while simultaneously shifting the decision boundary in a way that increases VT/VF confusion.

The boundary-contrastive model is especially informative: it achieved the largest VT/VF centre distance and the strongest error-detection AUROC for entropy, temperature-MSP, and KNN, but it also produced the worst macro-F1 and the most VT/VF cross-errors. In other words, the model became better at making its mistakes detectable, but worse as a classifier.

RISK-PRO++ was less damaging than the boundary-contrastive-only objective, but it still underperformed the teacher. The likely cause is over-regularisation: too many risk and geometry objectives compete with the supervised classification objective, especially under class imbalance and a hard VT/VF boundary.

## Practical Conclusion

Do not treat the current RISK-PRO++ weighting as a final stronger model. The current result should be reported as a pilot showing that multi-signal reliability supervision is feasible, but the objective needs reweighting and staged training.

## Recommended Next Step

Run a lighter RISK-PRO++ variant before any three-seed formal claim:

- Reduce `boundary_ce_weight` from `0.75` to `0.25`.
- Reduce `contrastive_weight` from `0.03` to `0.005` or `0.01`.
- Reduce `risk_entropy_weight` from `0.10` to `0.02`.
- Keep prototype centre/margin weak, or train it only after a supervised warm-up.
- Consider two-stage training: first train the classifier normally, then fine-tune with a small reliability regulariser for a few epochs.
- Use paired seeds only after a one-seed pilot shows no major degradation in macro-F1 or VT/VF cross-errors.

## Files

The key local result directories are:

- `results/core_interventions_risk_pro_plus/20260625_202904_reliability_gated_fusion_core_regularity_injection_seed42`
- `results/core_interventions_risk_pro_plus/20260625_204803_reliability_gated_fusion_core_prototype_separation_seed42`
- `results/core_interventions_risk_pro_plus/20260625_205817_reliability_gated_fusion_core_boundary_contrastive_seed42`
- `results/core_interventions_risk_pro_plus/20260625_210910_reliability_gated_fusion_core_risk_pro_plus_seed42`

The partial seed-43 directory was created when the pipeline auto-advanced after seed 42. It should be ignored for reporting unless that seed is intentionally rerun to completion.
