# 模型层因果-Pareto建模与内部验证结果

对应脚本：

```text
python -m src.run_model_layer_causal_pareto_validation
```

输出目录：

```text
results/model_layer_causal_pareto_validation_20260629/
```

## 1. 问题定义

这一步只做**模型层**验证，不比较 V5D routing/recover。

重要更正：

> 本文档验证的是 `auxiliary_intervention_matrix_20260606` 这一组辅助干预内部的 Pareto candidate。它不能被解释为“所有模型段里的全局最优模型”。全模型段比较见 `MODEL_LAYER_ALL_MODEL_BENCHMARK_CN.md`。

研究问题是：

> 在同一个 ECG 分类任务、同一个 `reliability_gated_fusion` backbone、同一个 paired seed 设计下，哪些训练层干预 `do(model_layer_intervention)` 能同时改善分类性能、校准、VT/VF 边界错误、总错误和错误迁移？

这相当于把模型改善写成因果式多目标优化：

```text
do(training_objective) -> model-only outcomes
```

## 2. 可干预变量

这次验证使用已有真实训练结果 `results/auxiliary_intervention_matrix_20260606`，包含 3 个 paired seeds：42、43、44。

| Variant | Causal intervention |
| --- | --- |
| `baseline` | `do(no_extra_model_layer_intervention)` |
| `boundary_weighted` | `do(boundary_ce_weight=1.0)` |
| `stability_consistency` | `do(stability_consistency_weight=0.2, embedding_consistency_weight=0.05)` |
| `full_supervisor` | `do(boundary_ce_weight=1.0, stability_consistency_weight=0.2, embedding_consistency_weight=0.05, prototype_center_weight=0.02, prototype_margin_weight=0.05, regularity_aux_weight=0.05)` |

不可干预变量包括原始 ECG 波形结构、真实 SR/VT/VF 标签、record-level split、seed/block 设计、已有数据规模。这些只作为设计约束，不作为 treatment。

## 3. Outcome

模型层 outcome 只包含分类模型本身的指标：

| Outcome | Direction |
| --- | --- |
| accuracy | higher better |
| macro-F1 | higher better |
| ECE | lower better |
| VT/VF cross-errors | lower better |
| total errors | lower better |
| error migration penalty | lower better |

没有使用 V5D capture、review budget、recover action 等 routing/recover 指标。

## 4. 验证结果

相对 baseline 的 paired mean effects 如下：

| Intervention | Accuracy delta | Macro-F1 delta | ECE delta | VT/VF cross-error delta | Total-error delta | Migration penalty delta | Pareto | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `full_supervisor` | +0.0333 | +0.0358 | -0.0174 | -8.67 | -141.33 | -83.17 | yes | yes |
| `boundary_weighted` | +0.0358 | +0.0265 | -0.0175 | -1.00 | -152.33 | -90.33 | yes | no |
| `stability_consistency` | +0.0150 | +0.0076 | -0.0039 | +15.00 | -59.67 | -24.00 | no | no |

两个候选在 Pareto 前沿：

- `boundary_weighted` 更偏 accuracy、total errors 和 migration penalty。
- `full_supervisor` 更偏 macro-F1、VT/VF cross-errors 和整体稳定性。

最终选中 `full_supervisor`，原因不是它每个数字都最大，而是它通过了模型层稳定性 guard：

| Intervention | accuracy good seeds | macro-F1 good seeds | ECE good seeds | VT/VF good seeds | total-error good seeds | migration good seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_supervisor` | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| `boundary_weighted` | 3/3 | 3/3 | 2/3 | 1/3 | 3/3 | 3/3 |
| `stability_consistency` | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 |

## 5. 建立的模型层候选

第一版模型层因果-Pareto候选为：

```text
causal_pareto_full_supervisor
```

它不是一个新的网络名字，而是一个经过因果-Pareto筛选的训练目标组合，基于 `reliability_gated_fusion` backbone：

```text
boundary_ce_weight = 1.0
stability_consistency_weight = 0.2
embedding_consistency_weight = 0.05
prototype_center_weight = 0.02
prototype_margin_weight = 0.05
prototype_vtvf_margin = 1.0
regularity_aux_weight = 0.05
```

这个模型候选把你前面分析的大量机制保留下来了：

- ECG 边界错误：通过 `boundary_ce_weight` 处理；
- 表征层稳定性：通过 `embedding_consistency_weight` 处理；
- 预测稳定性：通过 `stability_consistency_weight` 处理；
- prototype/embedding 距离：通过 prototype center 和 VT/VF margin 处理；
- rhythm/morphology regularity：通过 `regularity_aux_weight` 处理；
- error migration：作为 Pareto guard，而不是事后解释。

## 6. 结论边界

确认事实：

> 在 3 个 paired seeds 的内部模型层验证中，`causal_pareto_full_supervisor` 相对 baseline 同时改善 accuracy、macro-F1、ECE、VT/VF cross-errors、total errors 和 error migration penalty 的均值，并通过最小稳定性 guard。

合理解释：

> 单独的 stability consistency 不够；单独强化 boundary loss 有 trade-off；把 boundary、stability、embedding、prototype、regularity 组合起来，才更符合多目标因果优化的逻辑。

限制：

- 当前只有 3 个 paired seeds；
- 这是内部验证，不是外部验证；
- 没有临床验证或医疗器械意义；
- 还需要 OOD/ECG-structure-preserving shift stress test；
- 还需要在 model-only 过关后，再做 `V5D(causal_pareto_full_supervisor)` vs `V5D(baseline)` 的固定路由下游验证。

## 7. 下一步

建议下一步不是立刻宣称最终提升，而是做两件事：

1. 扩展 `causal_pareto_full_supervisor` 到更多 seeds 或重复 split，验证它是否仍然稳定。
2. 做 ECG 结构保持的 OOD stress test，确认它没有把 VT/VF 波形边界改坏。

只有这两步之后，才进入固定 V5D downstream：比较 `V5D(causal_pareto_full_supervisor)` 与 `V5D(baseline)`。
