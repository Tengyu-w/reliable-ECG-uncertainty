# `boundary075_prototype` 机制来源与因果证据链

日期：2026-06-30

## 1. 为什么需要这份解释

之前只说 `boundary075_prototype` 是一个 Pareto candidate，容易给人一种感觉：只是加了一个训练约束，然后看到结果变好。这个说法不够，因为它没有解释：

1. boundary 和 prototype 分别来自哪些前期失败分析；
2. 它们各自针对表征层、边界层和可靠性层的哪一类机制问题；
3. 干预后哪些机制变量真的发生了变化；
4. 这些机制变量变化如何对应最终 outcome 改善；
5. 它为什么属于多目标优化，而不是单一 loss trick。

本文件把 `boundary075_prototype` 拆成完整链条：

```text
前期机制分析
  -> 可干预训练约束
  -> 机制变量变化
  -> outcome 变化
  -> Pareto 多目标选择
```

## 2. 这个候选不是凭空来的

`boundary075_prototype` 的训练干预是：

```text
do(
  boundary_ce_weight = 0.75,
  prototype_center_weight = 0.02,
  prototype_margin_weight = 0.05,
  prototype_vtvf_margin = 1.0
)
```

它由两条前期分析线索合成：

| 来源分析 | 发现的问题 | 对应约束 | 作用位置 |
| --- | --- | --- | --- |
| VT/VF boundary 错误分析 | VT/VF 之间交叉错误集中，普通 CE 对高风险边界样本处理不足 | `boundary_ce_weight=0.75` | 样本权重 / 边界风险 |
| embedding / KNN / prototype 表征分析 | VT/VF 表征混叠、邻域不纯、prototype 歧义高 | `prototype_center_weight`, `prototype_margin_weight` | embedding geometry / prototype structure |

所以它不是“加一个约束试试看”，而是把前期两个最明确的失败机制转成可干预训练变量。

## 3. Boundary 部分来自什么

boundary 部分来自以下观察：

| 前期变量 / 现象 | 含义 | 为什么支持 boundary 干预 |
| --- | --- | --- |
| `vtvf_cross_errors` | VT 与 VF 相互误判 | 说明模型关键失败集中在 VT/VF 决策边界 |
| `softmax_vtvf_ambiguity` | VT/VF 概率接近 | 说明模型在 softmax 层已有边界不确定 |
| `boundary_score` | 边界 head 分数 | 可作为边界风险信号 |
| `validity_gate` / `gate_x_boundary` | gate 与 boundary 的联合可靠性信号 | 说明模型可能知道哪些样本不可靠，但需要训练约束校正 |
| `risk_targets` | 由 entropy、KNN、prototype、local instability、boundary ambiguity 等组成 | 可用于给高风险样本加权 |

因此，`boundary_ce_weight=0.75` 的逻辑是：

```text
高风险边界样本不应和普通样本等权训练
  -> 用 risk target 调整 CE 权重
  -> 让模型更重视 VT/VF 边界和高风险样本
```

为什么是 `0.75` 而不是 `1.0`：

```text
boundary 权重过强可能改善某些边界错误，
但也可能牺牲 macro-F1 或造成 error migration。
0.75 是在 boundary correction 和整体类别平衡之间的折中剂量。
```

## 4. Prototype 部分来自什么

prototype 部分来自表征层分析：

| 前期变量 / 现象 | 含义 | 为什么支持 prototype 干预 |
| --- | --- | --- |
| `silhouette_full` | embedding 全局类别分离度 | 分离度不足会让分类头更难稳定决策 |
| `sr_vt_norm_dist`, `sr_vf_norm_dist`, `vt_vf_norm_dist` | 类中心距离 / central distance | 类中心距离不足说明表征空间没有形成清晰结构 |
| `local_purity_k_mean` | KNN 邻域纯净度 | 邻域不纯说明样本处在混叠区域 |
| `knn_label_entropy_mean` | KNN 邻域标签熵 | 标签熵高说明局部类别不确定 |
| `knn_vtvf_mix_ventricular_mean` | VT/VF 邻域混合 | 直接对应 VT/VF 边界混叠 |
| `prototype_vtvf_ambiguity` | 样本距离 VT/VF 原型过近 | 原型层面支持 VT/VF ambiguity |
| `nearest_proto_is_pred` / representation conflict | 最近原型与模型预测冲突 | 说明分类头与 embedding 几何不一致 |

因此 prototype 约束的逻辑是：

```text
同类样本应该更紧
VT/VF 原型应该更分开
样本离 VT/VF 原型不应同时很近
```

对应训练目标：

| 约束 | 机制作用 |
| --- | --- |
| `prototype_center_weight=0.02` | 降低类内分散度，提高 embedding compactness |
| `prototype_margin_weight=0.05` | 强化类间原型距离，尤其 VT/VF |
| `prototype_vtvf_margin=1.0` | 给 VT/VF 原型分离设置最小 margin |

## 5. 多目标优化中每一部分负责什么

`boundary075_prototype` 不是单一目标，而是多目标分工：

| 组件 | 主要机制目标 | 主要 outcome 目标 | 潜在风险 |
| --- | --- | --- | --- |
| boundary CE | 让高风险边界样本得到更高训练权重 | 降低 VT/VF cross-error、total error、ECE | 权重太高会造成类别平衡破坏 |
| prototype center | 让同类 embedding 更紧 | 提高 macro-F1、降低 total error | 过强会压缩复杂类内部变化 |
| prototype VT/VF margin | 拉开 VT/VF 原型 | 降低 VT/VF boundary error | 过强可能造成 SR/ventricular 迁移错误 |
| Pareto guard | 同时检查 accuracy、macro-F1、ECE、VT/VF error、total error、migration | 避免单指标改进掩盖副作用 | 需要多 seed 验证 |

所以这个候选在设计上已经体现了多目标优化：

```text
不是只让 accuracy 高，
而是同时要求边界错误、校准错误、总错误和错误迁移都不能恶化。
```

## 6. 干预后机制变量发生了什么

`boundary075_prototype` 相对 same-seed baseline 的机制变量变化如下：

| 机制变量 | 平均变化 | 方向 | 解释 |
| --- | ---: | --- | --- |
| `silhouette_full` | +0.2067 | 3/3 seeds 增加 | embedding 全局分离度增强 |
| `sr_vt_norm_dist` | +0.9585 | 3/3 seeds 增加 | SR/VT 中心距离增加 |
| `sr_vf_norm_dist` | +1.3535 | 3/3 seeds 增加 | SR/VF 中心距离增加 |
| `vt_vf_norm_dist` | -0.0008 | 不稳定 | VT/VF 中心距离本身不是这轮最稳定解释变量 |
| `local_purity_k_mean` | +0.0204 | 3/3 seeds 增加 | KNN 邻域更纯 |
| `knn_label_entropy_mean` | -0.0128 | 3/3 seeds 降低 | 邻域标签更确定 |
| `knn_vtvf_mix_ventricular_mean` | -0.0277 | 2/3 seeds 降低 | VT/VF 邻域混合略降，但不如 purity 稳定 |
| `prototype_vtvf_ambiguity_ventricular_mean` | -0.0488 | 3/3 seeds 降低 | VT/VF 原型歧义降低 |
| `entropy_mean` | -0.0290 | 3/3 seeds 降低 | 整体预测不确定性下降 |
| `prob_margin_mean` | +0.0246 | 3/3 seeds 增加 | 概率边界更清楚 |
| `softmax_vtvf_ambiguity_ventricular_mean` | -0.0417 | 2/3 seeds 降低 | VT/VF softmax 歧义下降，但有 seed 波动 |
| `gate_x_boundary_any_error_auroc` | +0.6785 | 3/3 seeds 增加 | gate-boundary 组合更能识别错误 |

这说明它不是只改变最终分数，而是同时改变了多类中间机制：

```text
embedding separation
KNN neighborhood purity
prototype ambiguity
softmax margin / entropy
validity-boundary error separability
```

## 7. Outcome 发生了什么

`boundary075_prototype` 在 3 个 paired seeds 上 6 个 outcome 全部朝好方向变化：

| Outcome | 平均变化 | 好方向 seed 数 |
| --- | ---: | ---: |
| accuracy | +0.0317 | 3/3 |
| macro-F1 | +0.0429 | 3/3 |
| ECE | -0.0183 | 3/3 |
| VT/VF cross-errors | -20.33 | 3/3 |
| total errors | -135.00 | 3/3 |
| error migration penalty | -85.00 | 3/3 |

这就是为什么它被选为 full validation 候选。

## 8. 机制变量与 outcome 的关联

在 21 条 paired candidate-seed delta 上，多个机制变量与 outcome 改善方向一致：

| 机制变量 | Outcome | Spearman r | 解释 |
| --- | --- | ---: | --- |
| `gate_x_boundary_any_error_auroc` | error migration penalty | -0.709 | gate-boundary 错误识别越好，错误迁移越低 |
| `local_purity_k_mean` | error migration penalty | -0.701 | 邻域越纯，错误迁移越低 |
| `local_purity_k_mean` | ECE | -0.691 | 邻域越纯，校准越好 |
| `local_purity_k_mean` | total errors | -0.681 | 邻域越纯，总错误越少 |
| `knn_label_entropy_mean` | accuracy | -0.670 | 邻域标签熵越低，accuracy 越高 |
| `gate_x_boundary_any_error_auroc` | total errors | -0.666 | gate-boundary 信号越好，总错误越少 |
| `local_purity_k_mean` | accuracy | +0.665 | 邻域纯度越高，accuracy 越高 |
| `entropy_mean` | accuracy | -0.661 | 平均熵越低，accuracy 越高 |
| `gate_x_boundary_any_error_auroc` | accuracy | +0.660 | gate-boundary 错误识别越好，accuracy 越高 |
| `silhouette_full` | error migration penalty | -0.642 | embedding 分离越好，错误迁移越低 |

这一步非常重要：它把“我观察到表征层有问题”推进到“表征层机制变化与 outcome 改善方向一致”。

## 9. 为什么它不是单纯消融

普通消融只会说：

```text
加了 prototype，accuracy 变了多少。
```

本研究的因果式机制验证多了一层：

```text
do(boundary075 + prototype)
  -> 表征几何、KNN、prototype、softmax、validity 是否变化
  -> 这些变化是否与 outcome 改善一致
```

因此它更像：

```text
带机制中介证据的因果式消融
```

英文可以写作：

```text
mechanism-mediated causal-style ablation
```

## 10. 仍然没有完全解决什么

这份证据说明 `boundary075_prototype` 这个组合有效，但还没有完全回答：

1. prototype ambiguity 单独贡献多少；
2. KNN local purity 单独贡献多少；
3. softmax ambiguity 单独贡献多少；
4. validity gate alignment 单独贡献多少；
5. boundary CE 和 prototype margin 是否存在交互效应；
6. 为什么 `vt_vf_norm_dist` 本身在这轮并不稳定，但 VT/VF error 仍然改善。

因此下一步要做机制靶向消融，而不是再笼统地加一个大组合。

