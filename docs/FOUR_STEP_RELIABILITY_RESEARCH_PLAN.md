# Four-Step Reliability Research Plan

This plan reframes the ECG uncertainty project as a model-structure and decision-reliability study. The aim is not only to output a risk level, but to explain why SR/VT/VF decisions become unreliable and how representation evidence can be converted into better downstream decisions.

This is research evidence only. It should not be described as clinical validation or as a medical-device claim.

Seed-42 pilot findings are summarized in `docs/FOUR_STEP_SEED42_DIAGNOSTIC_FINDINGS_CN.md`.

The current end-to-end decision-calibration endpoint is summarized in `docs/VT_VF_DECISION_CALIBRATION_ENDPOINT_CN.md`.

## Step 1. Representation-Layer Diagnosis

### Question

Does the model progressively separate SR, VT, and VF across its internal representation path, and where does VT/VF ambiguity remain?

### Evidence

- waveform embedding geometry;
- regularity-feature embedding geometry;
- reliability gate behaviour;
- fused embedding geometry;
- final classifier-logit geometry;
- centre distances between SR-VT, SR-VF, and VT-VF;
- local neighbourhood purity and VT/VF mixing;
- whether VT/VF errors sit near the VT/VF centre boundary.

### Implementation

Use:

```bash
python -m src.layerwise_representation_diagnosis --mat RHYTHMS.mat --run-dir <run-dir>
```

Main outputs:

- `layerwise_representation_summary.csv`
- `layerwise_representation_diagnosis.json`
- `layerwise_centroid_distances.png`

### Interpretation Target

If SR separates early but VT/VF remains close or locally mixed, the model has a structured ventricular-boundary problem rather than a generic low-accuracy problem.

## Step 2. Decision-Boundary Diagnosis

### Question

Are mistakes caused by overlapping embeddings, or by a classifier head whose decision boundary disagrees with the representation geometry?

### Evidence

- VT as VF vs VF as VT asymmetry;
- nearest-prototype label vs classifier prediction;
- prototype margin vs logit margin;
- representation-overlap errors;
- classifier-boundary-mismatch errors;
- class imbalance and frozen-head retraining probes;
- temperature calibration before/after ECE.

### Implementation

Use:

```bash
python -m src.decision_boundary_diagnosis --run-dir <run-dir>
```

Main outputs:

- `decision_boundary_diagnosis.csv`
- `decision_boundary_summary.csv`
- `decision_boundary_mechanism_counts.csv`
- `decision_boundary_vtvf_margin_map.png`

### Interpretation Target

If many errors are nearest-prototype-correct but classifier-wrong, the decision head is the main suspect. If many errors are nearest-prototype-wrong, the representation itself is ambiguous. This distinction decides whether the next step should be classifier recalibration, class-balanced head retraining, conformal review, or representation learning.

## Step 3. Uncertainty Mechanism Decomposition

### Question

What type of uncertainty does each score actually capture?

### Mechanism Map

| Signal | What It Captures | Expected Failure Mode |
|---|---|---|
| Entropy / MSP | softmax hesitation | can miss confident wrong predictions |
| Temperature scaling | probability calibration | improves confidence realism, not representation |
| KNN distance | embedding atypicality | may detect outliers but not all boundary errors |
| KNN VT/VF mixing | local ventricular neighbourhood confusion | directly relevant to VT/VF ambiguity |
| Prototype distance | unlike typical class centres | can fail if class centres are not sufficient |
| Regularity features | rhythm/frequency structure | interpretable but lower-dimensional |
| Conformal sets | prediction-set uncertainty | can express `{VT, VF}` instead of forced single label |

### Implementation

Use the existing suite:

```bash
python -m src.evaluate_uncertainty --run-dir <run-dir>
python -m src.ambiguity_analysis --run-dir <run-dir>
python -m src.review_efficiency_analysis --run-dir <run-dir>
python -m src.regularity_analysis --mat RHYTHMS.mat --run-dir <run-dir>
python -m src.conformal_analysis --run-dir <run-dir>
```

Main outputs:

- `uncertainty_metrics.csv`
- `ambiguity_summary.csv`
- `review_efficiency_curves.csv`
- `regularity_summary.csv`
- `conformal_summary.csv`

### Interpretation Target

The goal is not to find one universal risk score. The goal is to identify which uncertainty mechanism captures which failure type, especially ordinary errors, VT/VF cross-errors, high-confidence errors, and boundary-set ambiguity.

## Step 4. From Explanation To Decision Improvement

### Question

How can representation-derived risk evidence improve decisions without damaging the backbone representation?

### Conservative Improvement Path

1. Freeze the teacher backbone.
2. Use embedding, logits, prototype distances, KNN mixing, regularity features, and calibration signals as decision evidence.
3. Train or fit only lightweight downstream components:
   - class-balanced classifier head;
   - temperature/vector/neural calibration head;
   - VT/VF-specific boundary calibrator;
   - review-routing risk head;
   - conformal `{VT, VF}` prediction-set policy.
4. Compare these methods under fixed review burden and class-specific error capture.

### Avoided Path

Do not immediately force all signals into a strong backbone loss. The seed-42 RISK-PRO++ pilot showed that centre distance and geometric separation can improve while macro-F1, ECE, and VT/VF cross-errors worsen.

### Interpretation Target

The final contribution should be framed as:

> Uncertainty analysis reveals a mismatch between representation geometry and decision reliability. The reliable solution is not necessarily stronger representation forcing, but decision calibration and review routing based on interpretable representation-derived evidence.

## Recommended First Run

Use the completed seed-42 teacher run:

```bash
python -m src.layerwise_representation_diagnosis --mat RHYTHMS.mat --run-dir results/core_interventions_risk_pro_plus/20260625_202904_reliability_gated_fusion_core_regularity_injection_seed42
python -m src.decision_boundary_diagnosis --run-dir results/core_interventions_risk_pro_plus/20260625_202904_reliability_gated_fusion_core_regularity_injection_seed42
```

Then compare with the intervention runs:

```bash
python -m src.layerwise_representation_diagnosis --mat RHYTHMS.mat --run-dir results/core_interventions_risk_pro_plus/20260625_204803_reliability_gated_fusion_core_prototype_separation_seed42
python -m src.decision_boundary_diagnosis --run-dir results/core_interventions_risk_pro_plus/20260625_204803_reliability_gated_fusion_core_prototype_separation_seed42
python -m src.layerwise_representation_diagnosis --mat RHYTHMS.mat --run-dir results/core_interventions_risk_pro_plus/20260625_210910_reliability_gated_fusion_core_risk_pro_plus_seed42
python -m src.decision_boundary_diagnosis --run-dir results/core_interventions_risk_pro_plus/20260625_210910_reliability_gated_fusion_core_risk_pro_plus_seed42
```
