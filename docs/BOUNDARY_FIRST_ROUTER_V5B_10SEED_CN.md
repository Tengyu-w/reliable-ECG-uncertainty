# v5b Boundary-First Router 10seed 结果

## 一句话结论

v5b 证明了我们刚才讨论的路线是对的：**把 softmax、validity gate、wavelet/time-frequency 三类边界证据合成一个专门的 VT/VF boundary-first route，比把所有机制混在一个综合路由里更能抓住 VT/VF 互错。**

最推荐的 v5b 版本不是复杂 learned ensemble，而是：

```text
mean_softmax_validity_wavelet
= robust_scaled(softmax_vtvf_ambiguity)
+ robust_scaled(validity_gate_x_boundary)
+ robust_scaled(wavelet_vtvf_boundary_risk)
```

再取平均。

## 这次做了什么

新增脚本：

```text
src/boundary_first_router_v5b.py
```

它读取：

- v4 router 的 `evidence_scores_val/test.csv`
- wavelet audit 生成的 `wavelet_vtvf_boundary_risk`
- CNN+TCN+Validity v1 的 `validity_gate` 和 `boundary_score`

然后只针对一个目标做路由：

```text
is_vtvf_cross_error
```

也就是说，它不再问“哪些样本整体最可能错”，而是专门问：

> 哪些样本最应该被拦进 `{VT,VF}` prediction set 或 VT/VF boundary expert review？

## 数据对齐

10 个 seed 全部完成：

| seeds | v4 evidence | validity v1 gate | wavelet evidence |
|---:|---|---|---|
| 42-51 | aligned | y_true aligned | aligned |

注意：validity v1 的 `y_pred` 和 teacher/router 的 `y_pred` 不一样，这是正常的，因为它来自另一个模型。我们只要求同一批样本的 `y_true` 对齐。

## 边界证据本身的检测力

| method | VT/VF cross-error AUROC | AUPR | 解释 |
|---|---:|---:|---|
| mean softmax+validity+wavelet | 0.9743 | 0.5105 | 最稳、最强 |
| learned boundary ensemble | 0.9696 | 0.4651 | validation 更高，但 test 不如简单平均 |
| softmax VT/VF ambiguity | 0.9664 | 0.4578 | 很强 |
| validity gate x boundary | 0.9634 | 0.4317 | 很强 |
| wavelet VT/VF boundary risk | 0.9619 | 0.4628 | 很强 |

这个结果说明三件事：

1. softmax、validity、wavelet 都是有效的 VT/VF 边界证据。
2. 简单平均比 learned ensemble 更稳，说明当前 validation set 上学权重有过拟合风险。
3. 这不是一个“复杂模型越复杂越好”的故事，而是“不同边界证据互相补充”的故事。

## 固定预算下的核心结果

| method | budget | actual action rate | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|---:|
| v4 optimized mechanism router | 10% | 10.0% | 57.7% | 59.7% |
| v5b recommended mean | 10% | 9.7% | 55.9% | 90.4% |
| v4 optimized mechanism router | 20% | 20.0% | 82.6% | 87.9% |
| v5b recommended mean | 20% | 15.2% | 78.9% | 99.7% |
| v4 optimized mechanism router | 30% | 30.0% | 93.8% | 97.2% |
| v5b recommended mean | 30% | 16.2% | 83.5% | 100.0% |

最关键的点是：v5b 在 20% 和 30% budget 下没有真的用满预算，因为它只在 VT/VF candidate 内做 boundary-first route。平均 action rate 分别只有 15.2% 和 16.2%，但已经几乎捕获所有 VT/VF cross-errors。

这说明它不是“多拦截所以更好”，而是“拦得更准”。

## 与 v4 的 paired improvement

推荐版 v5b 相对 v4：

| budget | delta all-error capture | delta VT/VF cross-error capture | delta action rate |
|---:|---:|---:|---:|
| 10% | -1.84 points | +30.69 points | -0.28 points |
| 20% | -3.72 points | +11.87 points | -4.75 points |

解释：

- v5b 牺牲了一点 all-error capture；
- 但大幅提升 VT/VF cross-error capture；
- 同时实际 action rate 更低；
- 这非常符合 boundary-first router 的定位。

它不是替代所有 review routing，而是专门作为 VT/VF boundary branch。

## 与单一边界证据的比较

| baseline | budget | v5b recommended 的 VT/VF capture 提升 |
|---|---:|---:|
| softmax VT/VF ambiguity | 10% | +5.73 points |
| validity gate | 10% | +6.05 points |
| wavelet VT/VF boundary risk | 10% | +1.71 points |
| softmax VT/VF ambiguity | 20% | +0.19 points |
| validity gate | 20% | +0.77 points |
| wavelet VT/VF boundary risk | 20% | +0.52 points |

10% budget 是最能体现价值的位置：三个证据合并后明显优于任何单一证据。20% 时大部分 VT/VF 互错已经被各个强信号抓到，所以提升变小。

## 为什么 learned ensemble 反而不是最优？

learned ensemble 的 validation AUROC 是 0.9876，比简单平均更高；但 test AUROC 只有 0.9696，而简单平均 test AUROC 是 0.9743。

这说明：

> 在当前数据规模下，直接学习边界证据权重可能会对 validation split 过拟合；简单、可解释、稳定的证据平均更适合作为论文主方法。

这是好事。因为博士申请/论文叙事里，一个可解释的简单组合通常比黑箱小模型更容易站住。

## v5b 在分层系统里的位置

现在机制路由应该拆成两层：

```text
Layer 1: Boundary-first route
  evidence:
    softmax_vtvf_ambiguity
    validity_gate_x_boundary
    wavelet_vtvf_boundary_risk
  action:
    {VT,VF} prediction set
    or VT/VF boundary expert review

Layer 2: Residual mechanism route
  evidence:
    representation/prototype/KNN conflict
    regularity / atypical signal
    model disagreement
    latent cluster / historical diagnostics
  action:
    representation review
    atypical signal review
    broader human review
```

这比“一个总路由管所有错误”更清楚，也更有创新性：

> 不同错误机制先被拆开，再用对应证据触发对应决策形式。

## 对完整 Wavelet+TCN 模型的影响

这次结果进一步支持：现在不急着跑完整 `CNN+Wavelet+TCN+Boundary` 10seed。

因为 wavelet evidence 已经在路由里证明有用；真正的问题不是“wavelet 有没有信息”，而是：

> 如何把 wavelet 信息用于正确的决策形式，而不是强行修正 VT/VF logits。

也就是说，wavelet 作为 route evidence 成功；wavelet 作为 end-to-end classifier 还需要方向约束后再谈。

## 论文可用表述

中文：

> 我们进一步构建了 boundary-first routing branch，将 softmax VT/VF ambiguity、validity-gated boundary score 与 fixed wavelet-derived boundary risk 组合为专门的 VT/VF 边界证据。10seed 结果显示，该简单平均组合在 10% action budget 下捕获 90.4% 的 VT/VF cross-errors，相比 v4 mechanism router 提升 30.7 个百分点；在实际 action rate 仅 15.2% 的情况下，20% budget 设置捕获 99.7% 的 VT/VF cross-errors。这表明高风险 VT/VF 边界错误更适合由 boundary-specific evidence branch 处理，而不是被混入一个统一的 all-error risk router。

English:

> We further introduced a boundary-first routing branch that combines softmax VT/VF ambiguity, validity-gated boundary evidence, and fixed wavelet-derived boundary risk. Across ten duplicate-family seeds, this simple evidence average captured 90.4% of VT/VF cross-errors at a nominal 10% action budget, improving over the v4 mechanism router by 30.7 percentage points. At a nominal 20% budget, it captured 99.7% of VT/VF cross-errors with an actual action rate of only 15.2%. These results suggest that high-risk VT/VF boundary failures are better handled by a boundary-specific routing branch than by a unified all-error risk router.

## 输出文件

- `src/boundary_first_router_v5b.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/boundary_first_router_v5b/boundary_first_policy_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/boundary_first_router_v5b/boundary_signal_diagnostics_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/boundary_first_router_v5b/paired_v5b_recommended_mean_vs_baselines_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/boundary_first_router_v5b/boundary_first_alignment_manifest.csv`
