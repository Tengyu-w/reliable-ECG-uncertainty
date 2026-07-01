# Mechanism-Aware ECG Reliability for VT/VF Boundary Classification

This repository studies reliable short-window ECG classification for three
rhythm labels:

- `SR`: sinus or non-ventricular rhythm
- `VT`: ventricular tachycardia
- `VF`: ventricular fibrillation

The project is not framed as a general accuracy-improvement exercise. Its main
question is:

> Why do VT and VF remain confusable even when overall ECG classification
> accuracy is high, and which mechanism-aware constraints can reduce that
> boundary risk without causing calibration failure or error migration?

This is a research prototype only. It is not a medical device, is not clinical
validation, and must not be used for diagnosis or clinical decision-making.

## Core Narrative

The current paper logic is:

```text
CNN / CNN-LSTM baselines
  -> VT/VF boundary problem definition
  -> representation-layer and signal-level mechanism analysis
  -> GatedFusion and constrained-model attempts
  -> mechanism-targeted causal-style ablation
  -> mechanism-derived model constraint search
  -> Stage 1 / Stage 2 recover routing as a fallback
```

The central contribution is the model-layer mechanism analysis and constraint
selection. The recover/router layer is retained as a downstream safety fallback
for residual high-risk errors.

## Claim-To-Evidence Matrix

| Main claim | Evidence used | Where to look |
| --- | --- | --- |
| High accuracy hides a VT/VF-specific reliability problem. | CNN/CNN-LSTM comparisons show VT/VF cross-errors behave differently from overall accuracy. | [paired_classification_comparisons.csv](results_public/tables/paired_classification_comparisons.csv), [model_performance_summary.png](results_public/figures/00_summary/model_performance_summary.png) |
| VT/VF confusion has representation-level structure. | Embedding geometry, local KNN purity, prototype ambiguity, and softmax ambiguity show boundary mixing. | [embedding PCA figures](results_public/figures/01_embedding_pca/), [MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md](docs/MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md) |
| Representation improvement alone is not enough. | PRO/prototype and regularity-style experiments can improve structure while leaving outcome trade-offs or error migration. | [PRO geometry figures](results_public/figures/06_pro_geometry/), [PRO error migration figures](results_public/figures/10_v6_pro_error_migration/) |
| Mechanism constraints must be evaluated by outcomes. | The 33-run mechanism-targeted ablation links each intervention to accuracy, macro-F1, ECE, VT/VF cross-errors, total errors, and migration. | [MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md](docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md) |
| The next model should be mechanism-derived rather than heuristic. | Current search tests whether `boundary + center` is sufficient or whether the full boundary-prototype margin is necessary. | [MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md](docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md) |
| Recover is a fallback layer, not the main model contribution. | V5D stage1/stage2 routing catches residual high-risk errors under fixed review budgets. | [V5D results](docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md), [V5D figures](results_public/figures/12_v5d_hierarchical_router/) |

## 1. Problem Definition: Accuracy Hides VT/VF Confusion

The first stage compares conventional classifiers such as CNN and CNN-LSTM.
Overall accuracy can look acceptable because the dataset contains many easier
SR or non-boundary windows. The reliability issue is concentrated in the VT/VF
boundary: VT and VF are both ventricular rhythms, and the model can confuse
them even when aggregate metrics look strong.

CNN-LSTM provides an important partial positive result. It reduces VT/VF
cross-errors relative to CNN, but does not solve the full reliability problem:
accuracy and calibration are worse, and total errors increase.

| Metric | CNN mean | CNN-LSTM mean | Interpretation |
| --- | ---: | ---: | --- |
| Accuracy | 0.8649 | 0.8518 | CNN-LSTM is lower. |
| Macro-F1 | 0.5928 | 0.6145 | CNN-LSTM improves class-balanced F1. |
| ECE | 0.0670 | 0.0747 | CNN-LSTM is less calibrated. |
| Total errors | 589.5 | 640.9 | CNN-LSTM makes more errors overall. |
| VT/VF cross-errors | 232.4 | 183.1 | CNN-LSTM reduces the key boundary error. |

Evidence:

- Model and geometry summary:
  [results_public/tables/model_performance_and_geometry.csv](results_public/tables/model_performance_and_geometry.csv)
- Paired CNN/CNN-LSTM comparisons:
  [results_public/tables/paired_classification_comparisons.csv](results_public/tables/paired_classification_comparisons.csv)
- Public performance figure:

![Model performance summary](results_public/figures/00_summary/model_performance_summary.png)

## Model Evolution

The model sequence is not presented as a leaderboard. Each stage tests a
specific hypothesis and motivates the next one.

| Model stage | Why introduced | What improved | What remained unresolved | What it motivated |
| --- | --- | --- | --- | --- |
| CNN | Establish a conventional short-window ECG baseline. | Basic SR/VT/VF classification. | VT/VF boundary errors remain hidden by aggregate accuracy. | Define VT/VF confusion as the core reliability problem. |
| CNN-LSTM | Add temporal context after CNN features. | VT/VF cross-errors decrease relative to CNN. | Accuracy, ECE, and total errors worsen on average. | Separate "boundary improvement" from general reliability. |
| GatedFusion | Fuse learned representation with regularity/reliability-style evidence. | Stronger aggregate model behavior and a better backbone. | It does not explain which mechanism produces the improvement. | Move from architecture comparison to mechanism analysis. |
| PRO / prototype / RiskPro-style constraints | Reshape representation geometry and risk-aware structure. | Some embedding and prototype measures improve. | Better-looking representations can still cause outcome trade-offs or error migration. | Introduce outcome guards and causal-style ablation. |
| Mechanism-targeted ablation | Test each mechanism as a controlled intervention. | Boundary and prototype-center mechanisms show strong evidence. | Margin, contrastive, gate, and regularity are not all safe to add directly. | Construct mechanism-derived model candidates. |
| Mechanism-derived search | Recompose the final model from validated mechanisms. | Active validation: test `boundary + center` versus the older four-term candidate. | Pending final 3-seed result. | Select the final model-layer constraint set. |
| V5D / recover | Catch residual high-risk cases after model prediction. | VT/VF boundary capture improves under fixed review budgets. | Not a replacement for model improvement. | Provides the final safety fallback layer. |

## 2. Mechanism Analysis: Why VT/VF Are Confused

After defining the VT/VF boundary problem, the project analyzes why it happens.
The failure is not treated as a random classification error. It is studied
through multiple measurable mechanisms:

| Mechanism family | What it measures | Why it matters |
| --- | --- | --- |
| Embedding geometry | PCA/LDA projections, silhouette, class-center distances | Tests whether VT and VF are separated in learned representation space. |
| KNN neighborhood structure | Local purity, label entropy, VT/VF local mixing | Tests whether a sample is surrounded by conflicting rhythm labels. |
| Prototype ambiguity | Distance to VT and VF class prototypes | Tests whether a sample is geometrically ambiguous between VT and VF. |
| Softmax ambiguity | Entropy, probability margin, VT/VF ambiguity score | Tests whether the classifier expresses uncertainty at the boundary. |
| Waveform regularity | Spectral entropy, dominant frequency, autocorrelation, line length | Tests whether ECG signal structure explains boundary or atypical cases. |
| Gate/validity evidence | Validity gate, boundary score, gate-boundary interaction | Tests whether the model has learned regions where automatic prediction is unsafe. |

The key representation finding is that VT/VF errors are associated with local
mixing and ambiguity in embedding, prototype, KNN, and softmax space. This
motivates model constraints that target specific mechanisms rather than simply
adding a larger neural network.

Visual evidence:

![Embedding geometry](results_public/figures/01_embedding_pca/contact_sheet.png)

Additional evidence:

- Embedding and geometry figures:
  [results_public/figures/01_embedding_pca/](results_public/figures/01_embedding_pca/)
- Regularity and waveform features:
  [results_public/figures/03_regularity_interpretability/](results_public/figures/03_regularity_interpretability/)
- Mechanism variable inventory:
  [docs/MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md](docs/MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md)

## 3. First Model Upgrade: GatedFusion Is Useful But Not Sufficient

The project then tests whether mechanism evidence can be integrated into the
classifier. GatedFusion and related models combine learned ECG representations
with regularity or reliability-style evidence. This improves aggregate
classification behavior and provides a stronger backbone than plain CNN-style
baselines.

However, GatedFusion alone does not answer the central causal question:

> Which mechanism changed, and did that mechanism actually improve the final
> outcome rather than only making the representation look cleaner?

This distinction matters because several experiments showed that representation
improvement can be misleading. A model can separate embeddings more cleanly
while still moving errors into another clinically important direction. This is
why the project moves from ordinary model comparison to mechanism-targeted
intervention and multi-objective outcome guards.

Evidence:

- PRO / prototype geometry:
  [results_public/figures/06_pro_geometry/](results_public/figures/06_pro_geometry/)
- PRO error migration:
  [results_public/figures/10_v6_pro_error_migration/](results_public/figures/10_v6_pro_error_migration/)
- Full model benchmark:
  [docs/MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md](docs/MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md)

![Prototype geometry and safety coupling](results_public/figures/06_pro_geometry/contact_sheet.png)

## 4. Mechanism-Targeted Causal-Style Ablation

The next stage treats model constraints as interventions:

```text
do(training constraint)
  -> measured mechanism change
  -> model outcome change
```

The outcomes are not limited to accuracy. A candidate must be checked against:

```text
accuracy
macro-F1
ECE
VT/VF cross-errors
total errors
error migration penalty
```

The 33-run mechanism-targeted ablation tests 11 candidates across 3 paired
seeds. Representative paired mean effects relative to baseline are:

| Candidate | Mechanism tested | Accuracy | Macro-F1 | ECE | VT/VF cross-errors | Total errors | Migration penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `proto_center_only` | Prototype compactness | +0.0415 | +0.0471 | -0.0244 | -20.67 | -178.0 | -109.3 |
| `proto_margin_only` | VT/VF prototype margin | +0.0099 | +0.0019 | -0.0071 | 0.00 | -43.67 | -26.83 |
| `boundary075` | Softmax boundary weighting | +0.0318 | +0.0230 | -0.0153 | -2.67 | -137.3 | -77.5 |
| `boundary075_prototype` | Boundary + prototype geometry | +0.0317 | +0.0429 | -0.0183 | -20.33 | -135.0 | -85.0 |
| `prototype_plus_contrastive` | Prototype + KNN/contrastive | +0.0288 | +0.0186 | -0.0150 | +2.00 | -123.7 | -68.0 |
| `regularity_aux_medium` | Waveform regularity auxiliary | -0.0016 | +0.0039 | +0.0081 | -5.67 | +9.0 | -1.67 |

Interpretation:

- Prototype center compactness is a strong mechanism.
- Prototype margin alone is weak.
- Boundary weighting improves global errors but does not fully solve VT/VF
  cross-errors by itself.
- Prototype plus contrastive can improve some representation signals while
  worsening VT/VF cross-errors.
- Regularity evidence is useful diagnostically, but direct auxiliary training
  is not stable enough to enter the main model objective.

Full evidence:
[docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md](docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md)

## 5. From Mechanism Analysis To Model Weights

The mechanism analysis is translated into explicit constraint weights. This is
the bridge between representation analysis and final model construction.

| Analysis source | Constraint weight | Intended model effect | Current evidence |
| --- | --- | --- | --- |
| VT/VF softmax ambiguity and boundary risk | `boundary_ce_weight` | Upweight high-risk boundary samples in CE loss | Useful, but needs representation constraint support. |
| Loose class clusters and low local purity | `prototype_center_weight` | Make within-class embeddings more compact | Strong single-mechanism result. |
| VT/VF prototype ambiguity | `prototype_margin_weight` | Penalize insufficient VT/VF prototype separation | Weak alone; must be tested with center/boundary. |
| Desired VT/VF separation target | `prototype_vtvf_margin` | Define margin threshold for VT/VF centers | Only meaningful when margin loss is active. |
| KNN local mixing | `contrastive_weight` | Improve local neighborhood purity | Strong alone, but can conflict with prototype constraints. |
| Calibration/overconfidence | `risk_entropy_weight`, `anti_confident_risk_weight` | Align uncertainty with risk and reduce confident high-risk errors | Kept as a guarded add-on, not a main mechanism yet. |
| Regularity features | `regularity_aux_weight` | Encourage embedding to encode waveform attributes | Useful diagnostically; unstable as direct loss. |
| Gate/validity evidence | `risk_gate_weight`, `risk_boundary_weight` | Align gate/boundary heads with risk targets | Better suited for routing/recover evidence. |

The old four-term boundary-prototype candidate was:

```text
boundary_ce_weight = 0.75
prototype_center_weight = 0.02
prototype_margin_weight = 0.05
prototype_vtvf_margin = 1.0
```

The current mechanism-derived model search asks whether all four terms are
necessary, or whether the model can be simplified to:

```text
boundary075_center:
  boundary_ce_weight = 0.75
  prototype_center_weight = 0.02
```

This is not a preference for a smaller neural network. It is a search for the
smallest sufficient mechanism-supported constraint set.

Active model-search plan:
[docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md](docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md)

### Active Validation Candidates

The current validation run is not an open-ended search over arbitrary losses.
It is a targeted decomposition of the old boundary-prototype model.

| Candidate | Constraint structure | Purpose |
| --- | --- | --- |
| `boundary075` | `boundary_ce_weight=0.75` | Test the boundary-risk term alone. |
| `proto_center_only` | `prototype_center_weight=0.02` | Test whether class compactness is the main prototype contribution. |
| `proto_margin_only` | `prototype_margin_weight=0.05`, `prototype_vtvf_margin=1.0` | Test whether VT/VF margin works without center compactness. |
| `proto_center_margin` | `center=0.02`, `margin=0.05`, `vtvf_margin=1.0` | Test the prototype-only combination. |
| `boundary075_center` | `boundary=0.75`, `center=0.02` | Main new candidate: boundary risk plus prototype compactness. |
| `boundary075_margin` | `boundary=0.75`, `margin=0.05`, `vtvf_margin=1.0` | Test whether margin adds value without center. |
| `boundary075_prototype` | `boundary=0.75`, `center=0.02`, `margin=0.05`, `vtvf_margin=1.0` | Older four-term reference candidate. |
| `boundary050_center` / `boundary100_center` | boundary dose `0.50` or `1.00` with center fixed | Check whether the boundary dose is sensitive. |
| `boundary075_contrastive` | `boundary=0.75`, `contrastive=0.02` | Test whether KNN/local-purity control can replace the prototype path. |
| `boundary075_center_calibrated` | `boundary=0.75`, `center=0.02`, entropy/confidence terms | Test whether calibration can be added without sacrificing VT/VF safety. |

## 6. Multi-Objective Selection Logic

The model is not selected by a single score. The selection logic is Pareto-style
and safety-guarded:

1. A candidate should improve or preserve classification performance.
2. It should reduce VT/VF cross-errors or avoid making them worse.
3. It should not sacrifice calibration.
4. It should reduce total errors and error migration.
5. Its mechanism should be interpretable from the earlier analysis.

This is why not all mechanisms are added to the final training objective.
Mechanisms such as regularity, validity/gate, stability, and explanation heads
remain important evidence sources, but they are not automatically valid
classifier losses.

Mechanism-outcome association evidence:

| Mechanism variable | Outcome | Association |
| --- | --- | --- |
| `local_purity_k_mean` | Total errors | Spearman -0.874 |
| `local_purity_k_mean` | Accuracy | Spearman +0.853 |
| `local_purity_k_mean` | Error migration penalty | Spearman -0.850 |
| `knn_label_entropy_mean` | Accuracy | Spearman -0.735 |
| `entropy_mean` | Accuracy | Spearman -0.663 |

These are internal causal-style proxies, not biological causal claims.

## 7. Recover / V5D As A Fallback Layer

The recover layer is placed after the model layer. Its purpose is not to replace
the classifier, but to handle residual high-risk errors under limited review
resources.

| Stage | Target | Evidence used |
| --- | --- | --- |
| Stage 1: VT/VF boundary-first protection | Samples near the VT/VF boundary | Softmax ambiguity, validity-boundary evidence, wavelet risk, prototype ambiguity, KNN mixing |
| Stage 2: residual mechanism recovery | Remaining high-risk non-boundary cases | SR-ventricular confusion, representation conflict, atypical signal, hidden confident evidence |

At a 20% action budget:

| Method | All-error capture | VT/VF cross-error capture | Automatic unresolved VT/VF rate |
| --- | ---: | ---: | ---: |
| v4 optimized mechanism router | 82.6% | 87.9% | 0.82% |
| v5d, 20% residual reserve | 86.0% | 99.0% | 0.07% |

Visual evidence:

![V5D hierarchical router](results_public/figures/12_v5d_hierarchical_router/contact_sheet.png)

Routing evidence:
[docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md](docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md)

## Current Status

| Layer | Status | Evidence |
| --- | --- | --- |
| CNN / CNN-LSTM baseline comparison | Complete | [results_public/tables/paired_classification_comparisons.csv](results_public/tables/paired_classification_comparisons.csv) |
| Representation and mechanism analysis | Complete as internal evidence | [docs/MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md](docs/MECHANISM_VARIABLE_MASTER_INVENTORY_CN.md) |
| 33-run mechanism-targeted ablation | Complete | [docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md](docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md) |
| Mechanism-derived model search | Active validation | [docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md](docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md) |
| V5D recover routing | Complete as fallback evidence | [docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md](docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md) |

## Repository Map

```text
src/
  train.py                                      Model training
  run_mechanism_targeted_causal_ablation.py    Component-level mechanism interventions
  summarize_mechanism_targeted_causal_quantification.py
                                                Intervention -> mechanism -> outcome summaries
  run_model_layer_causal_pareto_search.py      Mechanism-derived model search
  run_model_layer_all_model_benchmark.py       Model-stage benchmark inventory
  run_v5d_causal_pareto_weight_upgrade.py      V5D stage1/stage2 routing intervention
  hierarchical_router_v5d_reserved_budget.py   Reserved-budget V5D router

docs/
  Mechanism, model-layer, routing, and thesis-facing reports

results_public/
  Public-safe aggregate tables and figures only
```

## Main Documents

Start with these documents:

1. [Mechanism-targeted causal full results](docs/MECHANISM_TARGETED_CAUSAL_FULL_RESULTS_CN.md)
2. [Mechanism-derived model search plan](docs/MECHANISM_DERIVED_MODEL_SEARCH_PLAN_CN.md)
3. [Model-layer all-model benchmark](docs/MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md)
4. [V5D causal-Pareto weight upgrade](docs/V5D_CAUSAL_PARETO_WEIGHT_UPGRADE_RESULTS_CN.md)
5. [Thesis method section draft](docs/THESIS_METHOD_SECTION_CAUSAL_MECHANISM_CN.md)

Full documentation guide:
[docs/README.md](docs/README.md)

## Reproduce Key Experiments

The raw ECG file is expected locally as `RHYTHMS.mat`. It is not included in
this repository.

```powershell
# Inspect local data availability.
python -m src.inspect_data --mat RHYTHMS.mat

# Train a simple baseline.
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 30

# Mechanism-targeted ablation.
python -m src.run_mechanism_targeted_causal_ablation --seeds 42 43 44 --epochs 30

# Mechanism-derived model search.
python -m src.run_model_layer_causal_pareto_search --candidate-set mechanism-derived --seeds 42 43 44 --epochs 30

# V5D routing weight intervention.
python -m src.run_v5d_causal_pareto_weight_upgrade --budgets 0.20
```

## Public Evidence Boundary

The repository provides code, documentation, and aggregate non-identifiable
evidence. It does not distribute raw ECG waveforms, private review examples,
model checkpoints, full window-level prediction files, or internal metadata.

Public-safe summaries and figures are under:

```text
results_public/tables/
results_public/figures/
```

## Limitations

- Evidence is internal and paired-seed based; it is not external clinical
  validation.
- The mechanism-derived model search is still an active validation step.
- Mechanism variables are post-training diagnostics and causal-style proxies,
  not formal biological mediation proof.
- Several comparisons have only 3 paired seeds.
- Window-level classification should not be interpreted as patient-level
  diagnosis.
- This repository is intended for research review, not deployment.
