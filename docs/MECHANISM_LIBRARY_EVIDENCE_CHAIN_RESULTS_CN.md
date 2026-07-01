# 完整机制库证据链结果

日期：2026-06-30

本文件是 `CAUSAL_MECHANISM_QUANTIFICATION_RESULTS_CN.md` 的第二步补充。前一份文档回答的是模型层：

```text
do(training constraint) -> embedding/KNN/prototype/validity mechanism -> model outcome
```

本文件回答的是完整机制库：

```text
do(add/use evidence family or policy profile)
  -> mechanism signal quality
  -> routing/review/explanation outcome
```

也就是说，它把历史上已经分析过的 wavelet、regularity、mechanism heads、explanation alignment 等结果，统一整理成导师要求的“为什么这些机制有理由进入模型/路由”的定量证据链。

## 1. 输出文件

脚本：

- `src/build_mechanism_library_evidence_chain.py`

输入是已有历史结果表，没有重新训练模型。

输出目录：

- `results/mechanism_library_evidence_chain_20260630/`

核心表：

| 文件 | 含义 |
| --- | --- |
| `mechanism_signal_strength_inventory.csv` | 每个机制信号对目标错误/机制的 AUROC、AUPR、capture、lift 等 |
| `mechanism_policy_outcome_inventory.csv` | 使用某类机制证据或 policy 后的 routing/review outcome |
| `mechanism_evidence_source_inventory.csv` | 每类证据来自哪个历史结果文件 |
| `strongest_mechanism_signals.csv` | 最强机制信号摘要 |
| `mechanism_policy_highlights.csv` | routing/review outcome 高亮 |
| `mechanism_library_variable_dictionary.csv` | 可干预变量、机制变量、outcome 字典 |
| `mechanism_family_signal_summary.csv` | 机制族级别摘要 |

本次纳入：

- 400 条 mechanism signal evidence；
- 288 条 mechanism policy/review outcome evidence；
- 9 个历史结果文件；
- 5 类机制族：wavelet、regularity、mechanism-specific heads、mechanism router、explanation reliability。

## 2. 变量层级

### 可干预变量

这里的可干预变量不是 ECG 波形本身，而是实验中可以改变的 evidence 或 policy：

- `do(policy=v5_wavelet_boundary_router)`;
- `do(policy=optimized_mechanism_router_v4)`;
- `do(review_score=entropy/local_instability/vtvf_mixing)`;
- `do(feature_set=frequency_only/periodicity_only/all)`;
- `do(add wavelet evidence head)`;
- `do(add mechanism-specific risk head)`.

### 机制变量

机制变量是这些证据族对应的信号：

- `wavelet_vtvf_boundary_risk`;
- `wavelet_any_error_risk`;
- `representation_conflict`;
- `vtvf_boundary`;
- `atypical_signal`;
- `sr_ventricular`;
- `regularity waveform features`;
- `boundary_explanation`;
- `representation_explanation`;
- `regularity_atypicality_explanation`.

### Outcome

这里的 outcome 不是单纯模型分类 accuracy，而是机制层和路由层 outcome：

- signal quality：AUROC、AUPR；
- review/routing capture：`vtvf_cross_error_addressed`, `all_error_addressed`;
- residual risk：`automatic_unresolved_vtvf_cross_error_rate`;
- explanation alignment：解释分数与目标错误机制的一致性。

## 3. Wavelet 机制链

Wavelet/time-frequency evidence head 的核心结果：

| mechanism signal | target | AUROC mean | std | seeds |
| --- | --- | ---: | ---: | ---: |
| `wavelet_any_error_risk` | any error | 0.8886 | 0.0611 | 10 |
| `wavelet_vtvf_boundary_risk` | VT/VF cross-error | 0.9619 | 0.0187 | 10 |

这说明 wavelet 不是“装饰性波形特征”，而是对 VT/VF boundary error 有高可分性的时频证据。

使用 `v5_wavelet_boundary_router` 后：

| budget | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
| ---: | ---: | ---: | ---: |
| 20% | 0.8469 | 0.9345 | 0.00428 |
| 30% | 0.9513 | 0.9974 | 0.00021 |

因此 wavelet 机制链可以写成：

```text
do(use wavelet boundary evidence)
  -> wavelet_vtvf_boundary_risk AUROC = 0.9619
  -> high VT/VF cross-error capture under fixed review budget
```

## 4. Mechanism-Specific Risk Heads

机制风险头的 10-seed AUROC：

| mechanism head | target | AUROC mean | std | seeds |
| --- | --- | ---: | ---: | ---: |
| `representation_conflict` | representation conflict | 0.9899 | 0.0076 | 10 |
| `vtvf_boundary` | VT/VF boundary | 0.9539 | 0.0297 | 10 |
| `atypical_signal` | atypical signal | 0.9492 | 0.0379 | 10 |
| `sr_ventricular` | SR/ventricular confusion | 0.9125 | 0.0948 | 10 |
| `hidden_confident` | hidden confident | 0.5000 | 0.0000 | 10 |

结论：

- `representation_conflict`, `vtvf_boundary`, `atypical_signal`, `sr_ventricular` 有明确可学习信号；
- `hidden_confident` 当前不能作为可靠机制头，因为 AUROC 接近 0.5，正例也少，应保留为失败/负结果。

使用 optimized mechanism router：

| budget | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
| ---: | ---: | ---: | ---: |
| 10% | 0.5773 | 0.5968 | 0.0204 |
| 20% | 0.8263 | 0.8786 | 0.00822 |
| 30% | 0.9381 | 0.9725 | 0.00222 |

机制链可以写成：

```text
do(use mechanism-specific heads)
  -> representation_conflict / vtvf_boundary / atypical_signal heads achieve high AUROC
  -> optimized mechanism router captures most high-risk errors under fixed budget
```

## 5. Explanation Alignment 机制链

解释可靠性结果说明：某些 explanation family 确实对准了它声称解释的错误机制。

| explanation family | target | AUROC mean | std | seeds |
| --- | --- | ---: | ---: | ---: |
| `representation_explanation` | representation conflict error | 0.9701 | 0.0153 | 10 |
| `boundary_explanation` | VT/VF cross-error | 0.9646 | 0.0203 | 10 |
| `boundary_explanation` | VF -> VT | 0.9669 | 0.0146 | 10 |
| `boundary_explanation` | VT -> VF | 0.9465 | 0.0332 | 10 |
| `hidden_confidence_explanation` | hidden confident error | 0.8821 | 0.0781 | 10 |
| `regularity_atypicality_explanation` | atypical signal error | 0.6364 | 0.1191 | 10 |

结论：

- boundary explanation 和 representation explanation 最强；
- hidden-confidence explanation 对自身目标有一定对齐，但对应机制头本身不可靠，所以不能作为主路由机制；
- regularity explanation 对 atypical signal 的对齐较弱，只适合作辅助解释，不能过度写成核心机制。

机制链可以写成：

```text
do(use boundary explanation)
  -> explanation aligns with VT/VF cross-error
  -> route/review decision has interpretable mechanism reason
```

## 6. Regularity / Waveform 机制链

Regularity waveform feature ablation 显示，某些 ECG 波形结构特征对 VT/VF boundary 有强信号：

| feature set / model | VT/VF boundary AUROC | std | seeds |
| --- | ---: | ---: | ---: |
| `periodicity_only:gradient_boosting` | 0.9834 | 0.00025 | 3 |
| `without_frequency:gradient_boosting` | 0.9794 | 0.00576 | 3 |
| `frequency_only:gradient_boosting` | 0.9759 | 0.00685 | 3 |
| `all:gradient_boosting` | 0.9733 | 0.00368 | 3 |
| `frequency_only:random_forest` | 0.9616 | 0.00944 | 3 |

Review-policy 结果中，regularity/instability 相关 score 在 20% review burden 下也能捕获大量 VT/VF 错误：

| score | VT/VF error captured at 20% | std | seeds |
| --- | ---: | ---: | ---: |
| `waveform_only_baseline:resnet1d:vtvf_mixing` | 0.9884 | 0.0083 | 3 |
| `waveform_only_baseline:resnet1d:local_instability` | 0.9841 | 0.0055 | 3 |
| `regularity_feature_injection:reliability_gated_fusion:local_instability` | 0.8578 | 0.0754 | 3 |
| `regularity_feature_injection:reliability_gated_fusion:entropy` | 0.8450 | 0.0800 | 3 |

这里要谨慎解释：

- regularity/waveform features 对 VT/VF boundary 检出很强；
- 但 regularity explanation alignment 不如 boundary/representation explanation；
- 因此 regularity 更适合写成辅助波形结构证据，而不是唯一主机制。

## 7. 与模型层机制链的关系

现在项目里有两张互补证据链：

| 层级 | 可干预变量 | 机制变量 | outcome |
| --- | --- | --- | --- |
| 模型层 | training constraint weights | embedding/KNN/prototype/softmax/validity | accuracy, macro-F1, ECE, VT/VF errors |
| 机制库/路由层 | evidence family / policy profile / review score | wavelet, regularity, mechanism heads, explanation alignment | capture, residual risk, explanation reliability |

这能回应导师的问题：

> 你不是只分析了一堆原因结构，而是把每类原因结构变成了可量化机制变量，并检验它们如何影响模型 outcome 或 routing/review outcome。

## 8. 仍然不能过度声称的地方

不能写成：

> 我们证明了 ECG 真实生理因果机制。

更准确的写法是：

> We provide internal, paired, mechanism-aware evidence that model interventions and routing policies affect measurable representation, waveform, uncertainty, and explanation variables, which are associated with improved classification reliability or fixed-budget error capture.

限制：

- 没有外部数据集；
- 机制库聚合是历史结果表整理，不是新的随机化实验；
- regularity 主要是 3-seed 证据，routing/explanation/mechanism heads 多为 10-seed；
- model-layer outcome 和 routing outcome 已分层，不能直接互相比较；
- hidden-confident 是当前负结果，需要保留为机制边界。

## 9. 论文中的建议表述

中文：

> 为避免机制分析停留在事后解释层面，本文进一步构建机制库级证据链，将 wavelet 时频证据、regularity 波形结构、机制风险头和解释对齐结果统一为 `do(evidence/policy choice) -> mechanism signal -> routing/review outcome`。结果显示，wavelet VT/VF boundary risk 对 VT/VF cross-error 具有较高区分能力，机制风险头中的 representation conflict、VT/VF boundary 和 atypical signal 均具有稳定 AUROC，而 boundary/representation explanation 与其对应错误机制高度对齐。这些结果为后续多目标因果优化中的机制选择和权重分配提供了定量依据。

英文：

> To avoid treating mechanistic diagnostics as post-hoc observations only, we organize the evidence library into a causal-style chain: `do(evidence/policy choice) -> mechanism signal -> routing/review outcome`. Wavelet boundary evidence, mechanism-specific risk heads, regularity waveform features, and explanation alignment are evaluated as separate evidence families. The results show strong VT/VF boundary separability from wavelet evidence, high AUROC for representation-conflict and VT/VF-boundary mechanism heads, and strong alignment between boundary/representation explanations and their intended error mechanisms. These findings provide quantitative support for mechanism selection in the subsequent multi-objective causal optimization framework.
