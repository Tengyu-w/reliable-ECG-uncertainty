# 非表征证据 Usefulness Audit

## 为什么做这个实验

我们之前不只把表征分析写进模型，也把很多非表征分析信号写进过训练约束或更复杂的结构模型中，例如：

- risk target
- regularity features
- local instability / KNN
- VT/VF mixing
- softmax VT/VF ambiguity
- prototype / boundary geometry
- model disagreement
- validity gate / boundary score

但一个信号曾经被写进 loss 或模型结构，并不代表它在路由里一定有用。因此本实验做和 representation audit 类似的证据组消融：

> 每一类分析信号都必须被验证为 routing evidence，而不是默认可靠。

## 先回顾：两类训练方式

### 1. 约束式训练

代表：

- Risk-Pro
- Risk-Pro++
- Risk-Pro-readable
- prototype separation
- boundary contrastive
- regularity auxiliary
- risk/gate alignment

做法：

把分析信号变成 loss 或 auxiliary target。例如：

- 用 entropy、KNN、prototype、local instability、VT/VF mixing 组合出 `risk_target`；
- 让模型的 risk/gate head 拟合 risk target；
- 用 prototype loss 试图拉开类别中心；
- 用 boundary contrastive loss 强化 VT/VF 边界；
- 用 regularity auxiliary loss 让模型感知节律特征。

结果：

- 这些约束能改变风险信号或表征结构；
- 但不稳定减少 VT/VF cross-error；
- 因此它们不应被默认认为“可提升分类”。

### 2. 结构式训练

代表：

- CNN + TCN + Validity Bottleneck
- CNN + TCN + Validity v2
- CNN + Wavelet + TCN + Boundary Adapter

做法：

不是只加 loss，而是把分析逻辑写进模型路径：

```text
CNN local morphology
 + TCN rhythm branch
 + optional wavelet/time-frequency branch
 -> validity bottleneck
 -> validity gate / boundary score
 -> boundary adapter
 -> final logits
```

结果：

- validity/boundary gate 能识别高风险样本；
- wavelet 版本的 boundary detection 很强；
- 但识别风险不等于能正确修正分类方向；
- boundary adapter 可能产生方向性偏差，例如把 VT 大量推向 VF。

因此这些信号也应被审计为 routing evidence，而不是默认作为分类修正依据。

## 本次 audit 设计

脚本：

`src/nonrepresentation_evidence_usefulness_audit.py`

输出目录：

`results/evidence_informed_mechanism_routing_10seed_v4_20260627/nonrepresentation_evidence_usefulness_audit/`

审计信号组：

| family | 内容 |
|---|---|
| regularity | spectral entropy、dominant frequency、autocorrelation、line length 等 |
| risk target components | entropy、MSP、KNN、prototype、local instability、VT/VF mixing、softmax VT/VF ambiguity、risk target |
| softmax boundary | probability、entropy、margin、softmax VT/VF ambiguity |
| local instability / KNN | KNN distance、KNN entropy、KNN VT/VF mixing、local instability |
| latent cluster | cluster distance、cluster validation error rate |
| model disagreement | second model prediction/disagreement/entropy |
| historical diagnostics | risk target + prior calibration diagnostics |
| prototype geometry | prototype distance、nearest prototype、prototype margin |

每组做：

- `full`
- `without_<family>`
- `only_<family>`

本轮聚焦 10% 和 20% budget，因为这两个区间最能区分路由策略；30% 常接近饱和。

## Only-family 结果

### 10% budget

| only family | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|
| model disagreement | 0.5270 | 0.7661 | 0.0146 |
| softmax boundary | 0.5177 | 0.6722 | 0.0180 |
| regularity | 0.4313 | 0.6027 | 0.0207 |
| prototype geometry | 0.5165 | 0.5694 | 0.0219 |
| local instability / KNN | 0.5549 | 0.5547 | 0.0222 |
| risk target components | 0.5232 | 0.5277 | 0.0229 |
| historical diagnostics | 0.5082 | 0.5226 | 0.0232 |
| latent cluster | 0.4940 | 0.3044 | 0.0330 |

### 20% budget

| only family | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|
| model disagreement | 0.7871 | 0.9934 | 0.0005 |
| regularity | 0.7498 | 0.9380 | 0.0023 |
| prototype geometry | 0.8385 | 0.9372 | 0.0040 |
| local instability / KNN | 0.8394 | 0.8900 | 0.0072 |
| risk target components | 0.8021 | 0.8730 | 0.0045 |
| softmax boundary | 0.7798 | 0.8597 | 0.0099 |
| historical diagnostics | 0.7946 | 0.8564 | 0.0052 |
| latent cluster | 0.7833 | 0.7414 | 0.0149 |

## 关键解释

### 1. Model disagreement 是非常强的 VT/VF routing signal

Only model-disagreement:

- 10% budget 捕获 76.6% VT/VF cross-error；
- 20% budget 捕获 99.3% VT/VF cross-error。

这说明 second-opinion readable model 虽然作为分类模型不一定更强，但它与 Teacher 的分歧是很强的失败检测信号。

### 2. Regularity 信号不是装饰

Only regularity:

- 10% budget 捕获 60.3% VT/VF cross-error；
- 20% budget 捕获 93.8% VT/VF cross-error。

这说明节律/频域/自相关类特征对 VT/VF 边界和 atypical signal 有实际路由价值。

### 3. Prototype/local-instability 仍然有价值

Only prototype geometry 和 only local-instability/KNN 在 20% budget 下都很强：

- prototype geometry: 93.7% VT/VF capture；
- local instability/KNN: 89.0% VT/VF capture。

这支持之前 representation audit 的结论：表征相关信号不能直接当分类依据，但能作为 failure detector。

### 4. Risk target components 中等有用，但不是万能

Only risk-target components:

- 10% budget: 52.8% VT/VF capture；
- 20% budget: 87.3% VT/VF capture。

说明 risk target 作为综合风险信号有用，但它不是最强的 VT/VF 边界信号。

### 5. Latent cluster 单独较弱

Only latent cluster:

- 10% budget 只捕获 30.4% VT/VF cross-error；
- 20% budget 捕获 74.1%。

latent cluster 更适合作为辅助证据，不适合单独作为核心路由依据。

## 去掉某类信号后的变化

10% budget 下，full 相比 without-family：

| removed family | full - without in VT/VF capture | 解释 |
|---|---:|---|
| model disagreement | +0.1184 | 去掉分歧信号明显变差 |
| local instability / KNN | +0.0835 | 去掉局部不稳定信号变差 |
| prototype geometry | +0.0221 | 有小幅贡献 |
| latent cluster | +0.0144 | 小幅贡献 |
| historical diagnostics | -0.0006 | 基本无差异 |
| regularity | -0.0016 | full 在该 profile 下未充分利用 regularity |
| risk target components | -0.0037 | 基本无差异 |
| softmax boundary | -0.0146 | 去掉后反而略高，说明 full profile 目标仍需调整 |

20% budget 下，full 相比 without-family：

| removed family | full - without in VT/VF capture | 解释 |
|---|---:|---|
| regularity | +0.0181 | regularity 在 20% 有贡献 |
| prototype geometry | +0.0130 | prototype 有贡献 |
| softmax boundary | +0.0056 | 小幅贡献 |
| local instability / KNN | -0.0036 | 基本无差异 |
| historical diagnostics | -0.0075 | 基本无差异 |
| latent cluster | -0.0126 | 去掉后略好 |
| model disagreement | -0.0177 | 去掉后略好，但 only-model 很强，说明 profile 组合问题 |
| risk target components | -0.0236 | 去掉后略好，说明该综合信号可能稀释边界目标 |

## 对训练约束的解释

这些结果解释了为什么某些信号写进训练 loss 后没有稳定提升分类：

- 它们未必适合直接修正 logits；
- 但适合作为 routing evidence；
- 不同信号的用途不同：有些适合 VT/VF boundary，有些适合 all-error，有些适合 atypical review。

因此结论不是“这些约束没用”，而是：

> 它们更适合作为机制路由证据，而不是直接作为分类边界优化目标。

## 对结构式模型的解释

CNN+TCN+Validity 和 Wavelet+Boundary 的结果也可以这样解释：

- validity gate / boundary score 能识别风险；
- boundary adapter 直接修正方向时可能失败；
- 所以这些信号更应该进入 routing / prediction set，而不是直接强行改 VT/VF logits。

这与本 audit 一致：**风险识别信号适合做路由，不一定适合做分类修正。**

## 下一步

还需要做一个专门的 validity/boundary signal audit：

1. 把 CNN+TCN+Validity 的 `validity_gate` 和 `boundary_score` 接入 v4 router。
2. 把 Wavelet+TCN+Boundary 的 boundary score 接入 router。
3. 比较：
   - gate-only
   - boundary-score-only
   - full router + gate
   - full router + wavelet boundary
4. 判断它们是否应该作为 VT/VF boundary expert，而不是作为 logits adapter。

## 论文可用表述

英文：

> Analysis-derived signals that failed to consistently improve end-to-end classification can still provide useful routing evidence. Our evidence-family ablation shows that model disagreement, rhythm regularity, local instability, and prototype geometry each capture distinct failure modes, supporting their use as mechanism-specific reliability signals rather than direct label-correction constraints.

中文：

> 即使某些分析信号写入端到端训练后不能稳定提升分类，它们仍可能作为路由证据有效。证据组消融表明，模型分歧、节律 regularity、局部不稳定性和 prototype geometry 分别捕获不同失败模式，因此更适合作为机制特定的可靠性信号，而不是直接用于标签修正。
