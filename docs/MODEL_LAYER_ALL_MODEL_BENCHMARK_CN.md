# 模型段全量比较：CNN、CNN-LSTM、PRO、约束模型与复杂模型

对应脚本：

```text
python -m src.run_model_layer_all_model_benchmark
```

输出目录：

```text
results/model_layer_all_model_benchmark_20260629/
```

## 1. 为什么要做这一步

前一版只验证了 `reliability_gated_fusion` 内部的辅助干预矩阵，因此只能说明：

> `full_supervisor` 是该子实验内部的 model-layer Pareto candidate。

但这还不等于它优于所有模型段方法。模型段还包括 CNN、CNN-LSTM、PRO/prototype separation、boundary CE、risk/prototype/readable、validity、wavelet、regularity、以及更复杂的多约束组合。因此必须把所有模型段结果放到同一个总表里。

## 2. 本次扫描范围

脚本自动扫描：

```text
results/**/metrics.json
```

并从 `best_model.pt` 的训练参数中读取：

- `model`
- `seed`
- `epochs`
- `risk_targets`
- `boundary_ce_weight`
- `stability_consistency_weight`
- `embedding_consistency_weight`
- `prototype_center_weight`
- `prototype_margin_weight`
- `regularity_aux_weight`
- `risk_boundary_weight`
- `risk_entropy_weight`
- `anti_confident_risk_weight`
- `vtvf_specialist_weight`
- validity / gate / wavelet 等相关约束

本次共发现：

| Item | Count |
| --- | ---: |
| run-level `metrics.json` | 137 |
| model signatures | 29 |
| non-smoke/non-dry model signatures | 15 |

输出文件：

```text
all_discovered_model_runs.csv
all_discovered_model_signature_summary.csv
all_model_layer_benchmark.csv
all_model_layer_ranked_summary.csv
all_model_layer_benchmark_report.json
```

## 3. 重要方法边界

这些对象都属于模型段，但证据强度不同：

- 10 seed paired：CNN-LSTM、RiskProReadable；
- 3 seed paired：PRO、full supervisor、boundary/stability/prototype 类约束；
- single seed：部分 validity-v2、wavelet、top-level early runs；
- public aggregate snapshot：CNN、TCN、GatedFusion 等公开表。

因此不能只按一个统一排名宣布“绝对第一”。更合理的结论是：看完整指标、non-smoke、至少 3 seeds 的候选，再按目标解释 trade-off。

## 4. 核心候选比较

下面只列出 **完整指标 + non-smoke + 至少 3 seeds** 的模型签名：

| Model signature | Seeds | Accuracy | Macro-F1 | ECE | VT/VF cross-errors | Total errors | Migration penalty | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prototype_center=0.02 + prototype_margin=0.05` | 3 | 0.9199 | 0.6908 | 0.0587 | 174.17 | 342.78 | 229.00 | 7.33 |
| `boundary_ce=1.0 + risk_targets` | 3 | 0.9276 | 0.6509 | 0.0579 | 193.67 | 321.67 | 214.00 | 7.67 |
| `boundary + stability + embedding + prototype + regularity` | 3 | 0.9251 | 0.6602 | 0.0581 | 186.00 | 332.67 | 221.17 | 7.83 |
| `ResNet1D` | 3 | 0.9092 | 0.6365 | 0.0639 | 193.00 | 395.00 | 255.67 | 11.17 |
| `RiskProReadable complex objective` | 10 | 0.8951 | 0.6496 | 0.0535 | 198.20 | 459.80 | 296.20 | 12.67 |
| `stability + embedding consistency` | 3 | 0.9067 | 0.6320 | 0.0715 | 209.67 | 414.33 | 280.33 | 14.17 |
| `CNN-LSTM` | 10 | 0.8518 | 0.6145 | 0.0747 | 183.10 | 640.90 | 375.05 | 16.83 |

## 5. 各目标上的最优模型

| Objective | Best current model-stage candidate | Value |
| --- | --- | ---: |
| Accuracy | `boundary_ce=1.0 + risk_targets` | 0.9276 |
| Macro-F1 | `prototype_center=0.02 + prototype_margin=0.05` | 0.6908 |
| ECE | `RiskProReadable complex objective` | 0.0535 |
| VT/VF cross-errors | `prototype_center=0.02 + prototype_margin=0.05` | 174.17 |
| Total errors | `boundary_ce=1.0 + risk_targets` | 321.67 |
| Error migration penalty | `boundary_ce=1.0 + risk_targets` | 214.00 |

## 6. 现在的真实结论

更正后的结论是：

> `causal_pareto_full_supervisor` 不是所有模型段里的绝对最优模型。它是辅助干预矩阵内部的 Pareto candidate，在多目标上比较均衡，但放到全部模型段一起比较后，目前最值得保留的两个主候选是：prototype 约束模型和 boundary CE + risk-targets 模型。

具体解释：

- **Prototype 约束模型**：macro-F1 和 VT/VF cross-errors 最好，说明它对 VT/VF 表征边界有价值；但 total errors 和 migration penalty 不是最好，需要继续 guard。
- **Boundary CE + risk-targets 模型**：accuracy、total errors、migration penalty 最好，更像稳健的总体分类候选；但 macro-F1 和 VT/VF cross-errors 不如 prototype。
- **Full supervisor / causal-Pareto full supervisor**：均衡，但不是全局第一。它适合保留为综合约束候选，而不是当前唯一主模型。
- **CNN-LSTM**：VT/VF cross-errors 很低，但 accuracy、ECE、total errors 和 migration penalty 明显差，因此不能作为总体最优模型。
- **RiskProReadable complex objective**：ECE 最好，但 total errors、migration penalty 和 VT/VF cross-errors 不理想，因此不能作为主模型，只能作为复杂约束的负结果/校准参考。

## 7. 论文里应该怎么写

建议写成：

> We aggregated all model-stage experiments, including CNN, CNN-LSTM, prototype-constrained models, boundary-weighted models, risk/prototype objectives, validity/wavelet variants, and complex multi-constraint objectives. The results showed that no single model dominated all objectives. Prototype-constrained training achieved the best macro-F1 and VT/VF boundary error reduction, while boundary-weighted training with risk targets achieved the best accuracy, total-error reduction, and error-migration control. The previously selected full-supervisor model remained a balanced Pareto candidate within its intervention family, but was not the global best model-stage candidate.

中文：

> 全模型段比较显示，没有一个模型在所有目标上同时最优。prototype 约束模型最擅长 macro-F1 和 VT/VF 边界错误控制；boundary CE + risk-targets 最擅长 accuracy、total errors 和 error migration control；full-supervisor 是较均衡的多约束候选，但不是全局最优。因此下一步模型层因果-Pareto优化不应只押注 full-supervisor，而应围绕 prototype 约束和 boundary/risk-target 约束做 Pareto 组合与更多 seed 验证。

## 8. 下一步

下一轮真正应该验证的模型层 Pareto candidates 是：

1. `prototype_center=0.02 + prototype_margin=0.05`
2. `boundary_ce=1.0 + risk_targets`
3. `boundary + prototype + regularity` 的轻量组合，不直接采用过重的 RiskProReadable complex objective

验证顺序：

1. 先做 model-only 多 seed paired comparison；
2. 再做 ECG-structure-preserving OOD stress test；
3. 最后才进入固定 V5D downstream。

