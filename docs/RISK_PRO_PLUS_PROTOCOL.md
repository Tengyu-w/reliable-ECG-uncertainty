# RISK-PRO++ Protocol

This note defines the next reliability-aware intervention experiment. It is a
research prototype, not a clinical or medical-device claim.

## Motivation

PRO showed that the VT/VF representation boundary can be manipulated through
prototype compactness and VT/VF margin regularisation. The limitation is that
PRO uses only representation geometry during training, while the reliability
analysis found several complementary signals:

- output uncertainty: entropy and maximum-softmax uncertainty;
- representation atypicality: prototype distance and kNN distance;
- local boundary instability: neighbour-label support and VT/VF mixing;
- boundary ambiguity: softmax VT/VF ambiguity;
- perturbation robustness: prediction and embedding consistency.

RISK-PRO++ tests whether these signals can be folded back into training without
directly optimising non-differentiable neighbourhood metrics.

## Method

The experiment uses a two-stage teacher-student design.

1. Train a regularity-capable teacher model.
2. Generate a teacher risk target from post-hoc evidence:
   entropy, MSP uncertainty, kNN atypicality, prototype distance, local
   instability, VT/VF mixing, and softmax VT/VF ambiguity.
3. Train a student with a multi-objective reliability loss.

The student objective is:

```text
classification CE
+ risk-weighted CE
+ risk entropy alignment
+ risk-head / gate distillation, when using reliability_gated_fusion
+ prototype compactness
+ VT/VF prototype margin
+ boundary-aware contrastive loss
+ perturbation consistency
+ regularity auxiliary loss
```

The kNN, prototype-distance, and neighbourhood-mixing evidence is used as a
teacher signal instead of being inserted directly as a live batch loss. This
keeps the training objective differentiable and avoids making batch composition
dominate the result.

## Implementation

- `src/generate_risk_targets.py` now supports optional `msp` and `prototype`
  components.
- `src/train.py` now supports `--risk-entropy-weight`, which aligns predictive
  entropy with the teacher risk target.
- `src/run_core_intervention_pipeline.py` now includes a `risk_pro_plus` stage.

The new target weighting used in the core pipeline is:

| Component | Weight |
| --- | ---: |
| Entropy | 0.25 |
| MSP uncertainty | 0.10 |
| Local instability | 0.20 |
| VT/VF mixing | 0.20 |
| kNN atypicality | 0.15 |
| Prototype distance | 0.05 |
| Softmax VT/VF ambiguity | 0.05 |

## Evaluation

RISK-PRO++ should not be judged only by accuracy. It should be compared against
baseline, PRO, risk distillation, and boundary contrastive variants using:

- accuracy and macro-F1;
- ECE;
- VT/VF cross-errors;
- automatic-route VT/VF residual errors;
- review-capture rate under fixed review budgets;
- duplicate-family split stability;
- directional error migration.

## Reporting Rule

The strongest acceptable claim is conditional:

> RISK-PRO++ tests whether post-hoc reliability evidence can be distilled back
> into training to improve VT/VF boundary reliability.

Do not claim that it is a stable improvement until it beats baseline and PRO
across the duplicate-family split and multiple seeds without unacceptable error
migration.
