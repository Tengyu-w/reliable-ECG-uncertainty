# 基于机制验证后的模型约束重组计划

对应脚本：

```text
python -m src.run_model_layer_causal_pareto_search --candidate-set mechanism-derived
```

## 1. 为什么需要这一步

早期的 `boundary075_prototype` 是一个合理的机制组合候选，但它的四个权重并不是由后续 33-run 机制验证实验严格推导出来的。33-run 实验已经回答了一个更基础的问题：

```text
do(single mechanism)
-> mechanism variables change
-> model-only outcomes change
```

因此，下一步应从这些单机制干预结果出发，重新构造机制验证后的候选模型，而不是简单把旧的四权重模型当作最终模型。

## 2. 33-run 机制验证给出的组合规则

| 机制验证结果 | 对新候选的启发 |
| --- | --- |
| `proto_center_only` 稳定改善多个 outcome | prototype compactness 可能是 prototype 机制的主贡献 |
| `proto_margin_only` 单独较弱 | VT/VF margin 不应单独作为主机制，需要和 center 或 boundary 配合验证 |
| `proto_center_margin` 表现较强 | center + margin 仍是必要对照 |
| `boundary075` 改善总错误和 accuracy，但 VT/VF cross-error 改善有限 | boundary risk weighting 有用，但需要与表征机制组合 |
| `boundary075_prototype` 六个 outcome 均改善 | 保留为旧主候选和联合机制对照 |
| `contrastive_vtvf_light` 单独较强 | 需要测试 boundary + contrastive，而不是盲目 prototype + contrastive |
| `prototype_plus_contrastive` VT/VF cross-error 不稳 | prototype 与 contrastive 可能存在冲突，不能简单堆叠 |
| `regularity_aux_medium` outcome 不稳 | regularity 更适合作为机制证据或后续路由信号，不进入当前主训练候选 |
| `gate_boundary_joint` VT/VF outcome 不稳 | gate/validity 可保留为解释或 recover 信号，不作为当前主模型约束 |

## 3. 新候选集合

| Candidate | Constraint weights | 目的 |
| --- | --- | --- |
| `baseline` | none | paired baseline |
| `boundary075` | `boundary_ce_weight=0.75` | boundary 机制单独对照 |
| `proto_center_only` | `prototype_center_weight=0.02` | center compactness 单独对照 |
| `proto_margin_only` | `prototype_margin_weight=0.05`, `prototype_vtvf_margin=1.0` | margin 单独对照 |
| `proto_center_margin` | `center=0.02`, `margin=0.05`, `vtvf_margin=1.0` | prototype 内部组合对照 |
| `boundary075_prototype` | `boundary=0.75`, `center=0.02`, `margin=0.05`, `vtvf_margin=1.0` | 旧主候选/联合机制对照 |
| `boundary075_center` | `boundary=0.75`, `center=0.02` | 机制验证后最关键的新候选 |
| `boundary050_center` | `boundary=0.50`, `center=0.02` | boundary dose 下调敏感性 |
| `boundary100_center` | `boundary=1.0`, `center=0.02` | boundary dose 上调敏感性 |
| `boundary075_margin` | `boundary=0.75`, `margin=0.05`, `vtvf_margin=1.0` | 检查 margin 在 boundary 条件下是否有独立价值 |
| `boundary075_contrastive` | `boundary=0.75`, `contrastive=0.02`, boundary/VT-VF weights `2.0/2.0` | 检查 contrastive 是否与 boundary 比与 prototype 更兼容 |
| `boundary075_center_calibrated` | `boundary=0.75`, `center=0.02`, `risk_entropy=0.05`, `anti_confident=0.02` | 检查轻量校准是否能补 ECE 且不牺牲 VT/VF |

## 4. Outcome guard

这一步仍然只评估模型层 outcome，不混入 V5D/recover：

| Outcome | Direction |
| --- | --- |
| accuracy | higher better |
| macro-F1 | higher better |
| ECE | lower better |
| VT/VF cross-errors | lower better |
| total errors | lower better |
| error migration penalty | lower better |

最终选择规则不是单指标最高，而是：

```text
paired same-seed improvement
+ no clear degradation on safety-relevant outcomes
+ Pareto non-dominated or close-to-Pareto
+ mechanism interpretation remains simple enough to defend
```

## 5. 建议运行顺序

先做 smoke：

```text
python -m src.run_model_layer_causal_pareto_search \
  --candidate-set mechanism-derived \
  --seeds 42 \
  --epochs 1 \
  --max-windows-per-record 2 \
  --out results/mechanism_derived_model_search_smoke_20260701
```

若 smoke 通过，再做 3-seed 正式验证：

```text
python -m src.run_model_layer_causal_pareto_search \
  --candidate-set mechanism-derived \
  --seeds 42 43 44 \
  --epochs 30 \
  --out results/mechanism_derived_model_search_3seed_20260701
```

只有 3-seed 出现稳定候选后，再考虑扩到 10 seeds。

## 6. 论文口径

建议写法：

> Based on the 33-run mechanism-targeted causal ablation, we constructed a second-stage mechanism-derived model search. Instead of treating the previous boundary-prototype weights as final, we decomposed the verified mechanisms into boundary weighting, prototype compactness, VT/VF prototype margin, contrastive local-purity control, and lightweight calibration. Candidate models were then recomposed and evaluated using paired Pareto outcome guards. This design links representation-level evidence to model-level outcomes through explicit, testable constraint weights.

