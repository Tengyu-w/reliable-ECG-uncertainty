# Reliable ECG Classification Under Uncertainty

## 1. Research Question

This project studies reliable ECG rhythm classification under uncertainty. The
classification task has three labels: SR, VT, and VF. The medically important
failure mode is not just a wrong prediction in the aggregate test set. A more
specific concern is VT/VF cross-classification, where two ventricular rhythms
can occupy a close boundary in the learned representation space.

For the full experiment-by-experiment evidence narrative, see
[COMPLETE_EXPERIMENT_COMPENDIUM.md](COMPLETE_EXPERIMENT_COMPENDIUM.md). For the
extended public-safe figure atlas, see
[../results_public/figures/README.md](../results_public/figures/README.md).

The main research question is:

> Can an ECG classifier identify windows where its SR/VT/VF prediction is
> unreliable, so that high-risk VT/VF boundary cases can be routed for expert
> review?

The work is therefore framed as a reliability and review-routing prototype,
not as a clinical diagnostic system.

## Current GitHub Position

This report should be read as the public GitHub summary, not as a dump of every
local experiment. The final interpretation uses the V6 duplicate-family
evidence. Earlier paired PRO and supervisor tables are retained in the
repository as historical evidence, but final claims should use the
`duplicate_family_` public tables.

The current position is:

- PRO is a useful boundary-structure intervention, but under the stricter
  duplicate-family split it can shift errors and is not the final deployable
  solution.
- Validation-selected deployable RISK is the main contribution because it
  directly ranks cases for expert review under fixed review budgets.
- The repository shows internal reliability evidence only, not clinical
  validation.

## 2. Data Protocol

The code expects a local MATLAB file named `RHYTHMS.mat` containing ECG records
for SR, VT, and VF. The raw ECG file is not redistributed in this repository.

Each ECG record is converted into fixed-length windows. The split is performed
at the record level, so windows from the same original ECG recording do not
appear across train, validation, and test sets.

Public summary statistics retained in this repository:

```text
results_public/tables/dataset_split_statistics.csv
```

These statistics describe the aggregate split composition only. They do not
include raw ECG waveforms, record identifiers, model inputs, or window-level
predictions.

## 3. Method Overview

The pipeline has five stages:

1. ECG preprocessing and record-level splitting.
2. Baseline and extended time-series model training.
3. Uncertainty, calibration, and embedding-space reliability analysis.
4. VT/VF boundary-specific error analysis.
5. Selective prediction and review-routing evaluation.

The models include CNN, TCN, ResNet1D, InceptionTime, BiGRU, regularity-fusion
models, and reliability-gated variants. The analysis compares softmax-based
uncertainty with representation-space and ECG-specific signals such as kNN
distance, Mahalanobis distance, local neighbourhood instability, VT/VF
ambiguity, and rhythm regularity features.

For the code order, see [EXPERIMENT_PIPELINE.md](EXPERIMENT_PIPELINE.md).

## 4. Classification And Representation Evidence

The model-level summary shows that high overall accuracy does not remove the
need for reliability analysis. Macro-F1 and class-specific behaviour remain
important because SR is more frequent and easier to separate than the VT/VF
boundary.

![Model performance](../results_public/figures/00_summary/model_performance_summary.png)

The embedding geometry analysis supports the same conclusion. Across the
selected models, the normalised VT/VF centroid distance is much smaller than
the SR-VT or SR-VF distances. This suggests that VT and VF form the most
important boundary for uncertainty-aware analysis.

![Embedding geometry](../results_public/figures/00_summary/embedding_geometry_distances.png)

The corresponding aggregate table is:

```text
results_public/tables/model_performance_and_geometry.csv
```

## 5. Uncertainty And Calibration

The project compares several uncertainty scores:

- maximum softmax probability and entropy;
- temperature-scaled confidence;
- energy score;
- prototype distance;
- Mahalanobis distance;
- kNN embedding distance;
- hybrid decision/embedding scores.

Softmax-based scores are strong for detecting ordinary misclassifications in
some models, while embedding-based scores add complementary information about
atypical samples and distribution shift. The evidence does not justify a simple
claim that one uncertainty family is always superior. The safer conclusion is
that different scores detect different failure mechanisms.

![Uncertainty error detection](../results_public/figures/00_summary/uncertainty_error_detection.png)

Public table:

```text
results_public/tables/uncertainty_error_detection.csv
```

## 6. Review-Oriented Reliability

The project evaluates uncertainty as a decision policy, not just as a plot. A
review-routing policy ranks ECG windows by boundary-aware risk. The lower-risk
subset is accepted automatically, while the higher-risk subset is routed for
expert review.

This directly tests the practical reliability question:

> If only 10-30% of ECG windows can be reviewed, how many VT/VF boundary errors
> can be captured?

![Review routing](../results_public/figures/00_summary/review_routing_vtvf_capture.png)

Representative aggregate result:

- CNN-10 captured about 91.7% of VT/VF boundary errors at a 20% review budget.
- TCN-20 captured about 93.7% at the same budget.
- RegularityFusion-12 captured about 96.5% at the same budget.
- ResNet1D-12 had higher classification strength in some settings, but weaker
  low-budget VT/VF error capture, illustrating that a stronger backbone is not
  automatically a more reliable review-ranking model.

Public table:

```text
results_public/tables/review_routing_boundary_lrii.csv
```

## 7. Multi-Seed And Paired Evidence

The current paired comparison uses three random seeds. This is enough to show
the direction of several effects, but it is not enough for strong clinical or
deployment claims.

The public paired summaries are:

```text
results_public/tables/paired_classification_comparisons.csv
results_public/tables/paired_review_routing_comparisons.csv
```

These tables are historical paired evidence from earlier experiment versions.
They explain why PRO and supervisor-style routing were investigated, but they
are not the final GitHub conclusion.

The final duplicate-family summaries are:

```text
results_public/tables/duplicate_family_baseline_pro_summary.csv
results_public/tables/duplicate_family_selected_risk_review_aggregate.csv
results_public/tables/duplicate_family_pro_error_migration_mean_std.csv
results_public/tables/duplicate_family_risk_error_type_capture_mean_std.csv
```

In the final duplicate-family rerun, PRO is interpreted as error-migration
evidence rather than stable improvement. Validation-selected RISK is the main
review-routing result: it captured 82.8% of VT/VF cross-errors at 10% review
burden and 100.0% at 20% review burden across the three seeds.

## 8. What This Repository Demonstrates

This repository demonstrates:

- a complete PyTorch ECG classification pipeline;
- record-level splitting and leakage-aware data handling;
- uncertainty and calibration evaluation;
- embedding-space reliability analysis;
- VT/VF boundary-specific ambiguity analysis;
- review-routing metrics that connect uncertainty to expert workload;
- cautious aggregate reporting with multi-seed summaries.

The main research contribution is not a single model. It is the reliability
framing: the project asks whether the model can identify the ECG windows where
automation is least trustworthy.

## 9. Negative Results

The project includes negative and mixed results because they are part of the
research evidence.

- Stronger classification did not always imply stronger review routing.
  ResNet1D-12 was competitive as a classifier, but at a 10% review budget it
  captured only 38.3% of VT/VF boundary errors under the Boundary-LRII policy.
- Some uncertainty scores were weak for ordinary error detection. The energy
  score was especially poor in the selected public summary, and prototype or
  Mahalanobis distances were not consistently strong for CNN error detection.
- Multi-source risk evidence was not automatically better. Hand-weighted
  combinations can dilute the VT/VF boundary signal if the weights are not
  aligned with the review objective.
- Earlier paired comparisons were statistically uncertain because they used
  three seeds and some confidence intervals crossed zero.
- Under the stricter duplicate-family rerun, PRO can shift errors rather than
  consistently improve the classifier, so it is presented as boundary
  intervention evidence rather than the final method.
- Calibration is useful but not sufficient. Temperature scaling and ECE should
  be interpreted alongside review capture, OOD response, and boundary errors.

These results make the project more useful as a research prototype: they show
where the reliability hypothesis held, where it weakened, and which assumptions
need stronger validation.

## 10. Limitations

The current evidence is limited in several important ways:

- The dataset is internal/restricted and is not externally validated here.
- The public repository does not redistribute the raw ECG data.
- Current corruption tests are synthetic and do not replace real external OOD
  validation.
- The paired intervention evidence uses three seeds.
- Window-level evaluation is not patient-level diagnosis.
- The project is not clinically validated and is not a medical device.

These limitations are part of the research record rather than something to hide.
They define the next validation steps.

## 11. Next Steps

The most important next steps are:

1. Validate on an external ECG dataset with a clearly documented licence.
2. Extend the paired seed count beyond the current three-seed duplicate-family
   evidence.
3. Run external validation on a dataset with documented access and licence
   conditions.
4. Compare fixed review budgets against clinically meaningful review workflows.
5. Stress-test the selected duplicate-family RISK heads under corruption and
   shift, while keeping tuning restricted to validation data.

