# 模型层因果机制定量证据链结果

日期：2026-06-30

本文件回应导师提出的核心问题：不能只分析一堆 embedding、KNN、prototype、validity、regularity、wavelet 机制，然后直接把它们放进优化或路由中；必须定量说明这些机制为什么与模型结果有关。

## 1. 这次回答的问题

本次新增分析把模型层实验写成如下证据链：

```text
do(training constraint)
  -> mechanism variable change
  -> model outcome change
```

这里的 `do(training constraint)` 是可干预变量，例如：

- `prototype_guard`;
- `boundary_risk`;
- `boundary075_prototype`;
- `boundary075_prototype_calibrated`;
- `boundary075_prototype_reg`;
- `boundary075_prototype_stability`;
- `boundary100_prototype`.

机制变量是训练后观察到的内部证据变量，不直接人为干预，包括：

- embedding geometry：`silhouette_full`, `davies_bouldin_full`, `sr_vt_norm_dist`, `sr_vf_norm_dist`, `vt_vf_norm_dist`;
- KNN neighborhood：`local_purity_k_mean`, `knn_distance_mean`, `knn_label_entropy_mean`, `knn_vtvf_mix_ventricular_mean`;
- prototype ambiguity：`prototype_vtvf_ambiguity_ventricular_mean`, `prototype_vtvf_ambiguity_auroc`;
- softmax boundary/confidence：`softmax_vtvf_ambiguity_ventricular_mean`, `entropy_mean`, `confidence_mean`, `prob_margin_mean`;
- validity boundary：`validity_gate_mean`, `boundary_score_mean`, `gate_x_boundary_any_error_auroc`, `boundary_score_vtvf_cross_auroc`.

Outcome 是模型层结果：

- `accuracy`;
- `macro_f1`;
- `ece`;
- `vtvf_cross_errors`;
- `total_errors`;
- `error_migration_penalty`.

## 2. 生成的结果文件

脚本：

- `src/run_causal_mechanism_quantification.py`

输入实验：

- `results/model_layer_causal_pareto_search_full_20260630/`

输出目录：

- `results/causal_mechanism_quantification_20260630/`

核心文件：

| 文件 | 含义 |
| --- | --- |
| `causal_mechanism_variable_dictionary.csv` | 区分可干预变量、机制变量、outcome |
| `run_level_mechanism_outcome_table.csv` | 24 个模型 run 的机制变量 + outcome |
| `paired_candidate_seed_mechanism_outcome_deltas.csv` | 同 seed 下 candidate - baseline 的配对差值 |
| `intervention_to_mechanism_effects.csv` | 干预对机制变量的平均影响 |
| `intervention_to_outcome_effects.csv` | 干预对 outcome 的平均影响 |
| `mechanism_to_outcome_association.csv` | 机制变化与 outcome 变化的 Spearman/Pearson 关联 |
| `mediation_or_path_effect_summary.csv` | `干预 -> 机制 -> outcome` 的路径证据摘要 |

本次共纳入：

- 24 个 completed model runs；
- 21 个 paired candidate-seed deltas；
- 32 个机制变量；
- 192 条 mechanism-outcome association。

## 3. 主要结果

### 3.1 `boundary075_prototype` 是当前最清楚的新机制链

相对同 seed baseline，`boundary075_prototype` 在 3 个 seed 上所有 6 个 outcome 都朝好方向变化：

| outcome | 平均变化 | 3 seed 同向 |
| --- | ---: | ---: |
| accuracy | +0.0317 | 3/3 |
| macro-F1 | +0.0429 | 3/3 |
| ECE | -0.0183 | 3/3 |
| VT/VF cross-errors | -20.33 | 3/3 |
| total errors | -135.0 | 3/3 |
| error migration penalty | -85.0 | 3/3 |

它对应的机制变化也不是空的：

| mechanism variable | 平均变化 | 解释 |
| --- | ---: | --- |
| `silhouette_full` | +0.2067 | overall embedding separation 增强 |
| `local_purity_k_mean` | +0.0204 | KNN 邻域同类纯度提升 |
| `prototype_vtvf_ambiguity_ventricular_mean` | -0.0488 | VT/VF prototype ambiguity 降低 |
| `softmax_vtvf_ambiguity_ventricular_mean` | -0.0417 | VT/VF softmax ambiguity 降低 |
| `gate_x_boundary_any_error_auroc` | +0.6785 | validity × boundary 对 any-error 的识别能力增强 |

这说明 `boundary075_prototype` 不只是 outcome 变好，而是同时改变了表征邻域、prototype ambiguity、softmax boundary ambiguity 和 validity-boundary evidence。

### 3.2 机制变量与 outcome 的关联方向合理

在 21 个 paired candidate-seed deltas 上，若干机制变量和 outcome 变化有较强关联：

| mechanism variable | outcome | Spearman r | p |
| --- | --- | ---: | ---: |
| `gate_x_boundary_any_error_auroc` | error migration penalty | -0.709 | 0.00032 |
| `local_purity_k_mean` | error migration penalty | -0.701 | 0.00040 |
| `local_purity_k_mean` | ECE | -0.691 | 0.00052 |
| `local_purity_k_mean` | total errors | -0.681 | 0.00068 |
| `knn_label_entropy_mean` | accuracy | -0.670 | 0.00089 |
| `gate_x_boundary_any_error_auroc` | total errors | -0.666 | 0.00098 |
| `local_purity_k_mean` | accuracy | +0.665 | 0.00101 |
| `entropy_mean` | accuracy | -0.661 | 0.00110 |
| `gate_x_boundary_any_error_auroc` | accuracy | +0.660 | 0.00114 |

这给出了导师想要的定量逻辑：

- KNN local purity 越高，total errors、ECE、error migration 越低；
- gate × boundary 对错误识别越强，total errors 和 migration penalty 越低；
- 平均 entropy 或 KNN label entropy 越高，accuracy 越差；
- embedding separation 增强与 migration penalty 降低相关。

### 3.3 旧强候选 `prototype_guard` 仍然必须保留作对照

`prototype_guard` 相对 baseline 的 outcome 变化：

| outcome | 平均变化 | 3 seed 同向 |
| --- | ---: | ---: |
| accuracy | +0.0371 | 3/3 |
| macro-F1 | +0.0442 | 3/3 |
| ECE | -0.0277 | 3/3 |
| VT/VF cross-errors | -15.33 | 3/3 |
| total errors | -160.33 | 3/3 |
| error migration penalty | -94.0 | 3/3 |

因此论文里不能把 `boundary075_prototype` 写成绝对击败所有旧机制。更合理的表述是：

> Prototype constraint 是旧强基线；`boundary075_prototype` 是把 boundary weighting 和 prototype geometry 做成因果-Pareto重组后的新候选。它在 VT/VF cross-error 上更强，而 `prototype_guard` 在 total errors、ECE 和整体指标上仍然很强。

## 4. 目前可以写成论文贡献的部分

可以写：

> We quantify the mechanism chain behind model-layer interventions rather than treating representation diagnostics as post-hoc observations only. In paired same-seed comparisons, the boundary-prototype intervention improves all six model outcomes and simultaneously changes representation purity, prototype VT/VF ambiguity, softmax VT/VF ambiguity, and validity-boundary error separability. Mechanism-outcome association further shows that higher local neighborhood purity and stronger gate-boundary error separability are associated with lower total errors, lower calibration error, and lower error migration penalty.

中文表述：

> 本文不是简单地把 embedding、KNN、prototype、validity 等诊断结果作为经验性解释，而是将训练约束视为可干预变量，定量估计其对机制变量和模型 outcome 的影响。配对 seed 结果显示，boundary-prototype 干预同时改善 6 个 outcome，并改变 KNN 邻域纯度、prototype VT/VF ambiguity、softmax VT/VF ambiguity 和 validity-boundary 错误识别能力。机制-outcome 关联进一步表明，局部邻域纯度和 gate-boundary 错误可分性与 total errors、ECE 和 error migration penalty 的降低一致相关。

## 5. 重要限制

这不是外部临床验证，也不是正式的随机化中介因果证明。更准确地说，它是内部 paired do-intervention proxy：

- 每个候选只有 3 个 seeds；
- 没有外部 ECG 数据集；
- 机制变量是 post-training mediator/diagnostic，不能说成已经证明真实生理因果；
- 当前脚本覆盖的是本次模型层 Pareto search 的 embedding/KNN/prototype/softmax/validity 机制；
- wavelet、regularity、explanation alignment、mechanism routing heads 有历史 10-seed 或多 seed 证据，但还需要被整理进同一个 `do(evidence/policy intervention) -> mechanism -> routing outcome` 表中。

## 6. 下一步

下一步应该做第二张证据链表，不是重新训练模型，而是把历史机制结果接入同一套格式：

| 机制族 | 已有证据位置 | 下一步证据链形式 |
| --- | --- | --- |
| wavelet time-frequency | `wavelet_boundary_routing_audit/*mean_std.csv` | `do(add/weight wavelet evidence) -> wavelet risk AUROC -> VT/VF capture/residual risk` |
| regularity waveform | `regularity_summary.csv`, `regularity_feature_ablation_mean_std.csv` | `do(add regularity features/auxiliary) -> atypicality/regularity separation -> error or routing outcome` |
| mechanism risk heads | `mechanism_head_mean_std.csv` | `do(mechanism head) -> mechanism AUROC -> mechanism-specific capture` |
| explanation alignment | `explanation_alignment_mean_std.csv` | `do(explanation family) -> alignment AUROC -> trusted routing explanation` |

这一步完成后，项目的因果推断部分会从“模型层机制链”扩展成“完整机制库证据链”。
