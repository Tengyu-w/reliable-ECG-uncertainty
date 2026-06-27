# v4 与更强路由 baseline 的 10-seed 对比

## 目的

此前 v4 已经与 learned total-risk layered routing 对比，并在低/中 budget 下取得平均提升。但为了更接近论文级验证，还需要和更强、更简单的 baseline 对比：

- entropy
- MSP uncertainty
- low softmax margin
- softmax VT/VF ambiguity
- learned VT/VF boundary risk
- learned any-error risk
- KNN VT/VF mixing
- prototype VT/VF ambiguity
- latent-cluster risk
- model disagreement
- conformal set size
- fixed mechanism router v3b
- learned total-risk layered router
- optimized mechanism router v4

脚本：

`src/compare_routing_baselines_10seed.py`

输出目录：

`results/evidence_informed_mechanism_routing_10seed_v4_20260627/baseline_comparison/`

## 关键发现

v4 的强结论：

- v4 明确优于 fixed mechanism router v3b。
- v4 优于 learned total-risk layered router，尤其 5%-20% budget。
- v4 能稳定吃满预算。

但更严格的 baseline 暴露出一个重要事实：

**如果目标只看 VT/VF cross-error capture，`softmax_vtvf_ambiguity` 是非常强的单信号 baseline，并且在当前实验里强于 v4。**

这不是坏结果，而是告诉我们 v5 应该把 `softmax_vtvf_ambiguity` 明确作为 VT/VF boundary expert 的核心基线/核心输入，而不是只强调机制框架。

## 10% budget 结果

| method | action rate | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|---:|
| softmax VT/VF ambiguity | 0.1000 | 0.5232 | 0.8413 | 0.0106 |
| latent cluster VT/VF cross-rate | 0.1000 | 0.4782 | 0.7872 | 0.0123 |
| learned VT/VF boundary risk | 0.1000 | 0.4976 | 0.7664 | 0.0147 |
| KNN VT/VF mixing | 0.1000 | 0.3530 | 0.6585 | 0.0165 |
| learned any-error risk | 0.1000 | 0.6339 | 0.6367 | 0.0196 |
| optimized mechanism router v4 | 0.1000 | 0.5773 | 0.5969 | 0.0204 |
| learned total-risk layered | 0.0953 | 0.5606 | 0.5053 | 0.0230 |
| fixed mechanism router v3b | 0.0734 | 0.3733 | 0.3789 | 0.0278 |

解释：

- v4 比 learned total-risk router 更好。
- v4 比 fixed v3b 好很多。
- 但对于纯 VT/VF cross-error 捕获，softmax VT/VF ambiguity 明显更强。

## 20% budget 结果

| method | action rate | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|---:|
| softmax VT/VF ambiguity | 0.2000 | 0.8102 | 0.9921 | 0.0005 |
| latent cluster VT/VF cross-rate | 0.2000 | 0.7291 | 0.9916 | 0.0004 |
| learned VT/VF boundary risk | 0.2000 | 0.7787 | 0.9868 | 0.0008 |
| low softmax margin | 0.2000 | 0.8083 | 0.9377 | 0.0039 |
| learned any-error risk | 0.2000 | 0.8490 | 0.8959 | 0.0072 |
| optimized mechanism router v4 | 0.2000 | 0.8263 | 0.8786 | 0.0082 |
| learned total-risk layered | 0.1826 | 0.8039 | 0.8321 | 0.0084 |

解释：

- v4 对 all-error addressed 表现不错。
- 但 VT/VF-specific 单信号非常强，尤其 `softmax_vtvf_ambiguity`。
- 这说明 VT/VF boundary route 应该单独强化，而不能完全依赖机制 profile 的整体 utility。

## 30% budget 结果

30% budget 已经接近饱和，多个方法都能捕获几乎全部 VT/VF cross-error：

- softmax VT/VF ambiguity: 1.0000
- entropy/MSP/margin: 接近 1.0000
- v4: 0.9725
- learned total-risk: 0.9727

因此 30% budget 不是最能体现方法差异的区间。更有意义的是 5%-20%。

## Conformal baseline 限制

当前 conformal set-size baseline 只有 seed42 有可对齐列，因此 `n_seeds = 1`，不能作为可靠 10seed 结论。后续如果要正式比较 conformal，需要为所有 seed 重新生成 validation/test conformal outputs。

## 当前判断

v4 已经能作为机制化框架证明：

- 它比 fixed mechanism routing 好。
- 它比 learned total-risk routing 好。
- 它能把不同错误机制分开解释。

但如果论文目标是“最大化 VT/VF cross-error 捕获”，当前最强 baseline 是：

**softmax VT/VF ambiguity**

因此下一步 v5 应该做：

1. 保留机制分型框架。
2. 把 VT/VF boundary expert 改成强 boundary expert，以 `softmax_vtvf_ambiguity` 为核心。
3. validation objective 分开优化：
   - VT/VF cross-error capture
   - all-error capture
   - mechanism diversity / interpretability
4. 报告 v5 是否能同时保留解释结构，并接近或超过最强单信号 VT/VF baseline。

## 论文里应如何表述

谨慎说法：

> The optimized mechanism-aware router outperforms learned scalar total-risk routing across low-to-medium action budgets. However, a simple VT/VF ambiguity score remains a strong boundary-specific baseline, indicating that future mechanism profiles should explicitly privilege boundary ambiguity when VT/VF cross-error capture is the primary objective.

中文：

> 优化后的机制感知路由在低/中预算下优于单一总风险路由。但单独的 VT/VF ambiguity 分数仍是很强的边界错误 baseline，这说明后续机制组合应在 VT/VF 捕获目标下显式强化 boundary expert。
