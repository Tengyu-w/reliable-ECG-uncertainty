# v4 机制分型路由 10-seed 结果

## 目的

v3b 已经证明错误机制风险头可学，但固定预算分配没有打赢总风险路由。v4 的目标是验证：

- 机制分型是否不仅可解释，而且能提升实际路由效果；
- validation 是否可以自动选择机制门权重；
- 机制路由是否能吃满目标 action budget。

输出目录：

`results/evidence_informed_mechanism_routing_10seed_v4_20260627/`

核心脚本：

- `src/evidence_informed_recovery_routing.py`
- `src/run_mechanism_routing_10seed.py`

## v4 方法变化

v3b 使用手写机制预算权重，例如 VT/VF boundary、SR-ventricular、representation conflict、atypical signal 各自分到固定比例。

v4 改成 validation-optimized profile selection：

1. 每个机制头给出候选动作。
2. 预设一组可解释 profile，例如：
   - equal
   - VT/VF boundary heavy
   - representation heavy
   - atypical heavy
   - boundary + atypical
   - representation + atypical
   - single-mechanism profiles
3. 在 validation split 上选择效用最高的 profile。
4. 在 test split 上按该 profile 选择 top candidates，并强制接近目标 budget。

这一步保留了机制解释，同时避免把所有证据压成一个总风险分数。

## 机制风险头稳定性

| mechanism | AUROC mean | AUROC std | AUPR mean | AUPR std | enabled |
|---|---:|---:|---:|---:|---:|
| representation conflict | 0.9899 | 0.0076 | 0.6857 | 0.1682 | 10/10 |
| atypical signal | 0.9492 | 0.0380 | 0.5606 | 0.2343 | 10/10 |
| VT/VF boundary | 0.9539 | 0.0297 | 0.3872 | 0.1148 | 10/10 |
| SR-ventricular | 0.9125 | 0.0948 | 0.5092 | 0.1840 | 10/10 |
| hidden confident | 0.5000 | 0.0000 | 0.0079 | 0.0107 | 0/10 |

机制头稳定性仍然成立。`hidden_confident` 在当前数据/validation 规模下不能启用。

## v4 optimized mechanism routing

| budget | action rate | VT/VF set rate | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0500 | 0.0208 | 0.3179 | 0.2463 | 0.0335 |
| 0.10 | 0.1000 | 0.0478 | 0.5773 | 0.5969 | 0.0204 |
| 0.20 | 0.2000 | 0.0648 | 0.8263 | 0.8786 | 0.0082 |
| 0.30 | 0.3000 | 0.0765 | 0.9381 | 0.9725 | 0.0022 |

v4 已经解决 v3b 的最大问题：action rate 基本等于目标 budget。

## 与总风险路由对比

总风险路由：

| budget | action rate approx. | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---:|---:|---:|---:|---:|
| 0.05 | 0.0418 | 0.2731 | 0.2026 | 0.0357 |
| 0.10 | 0.0953 | 0.5606 | 0.5053 | 0.0230 |
| 0.20 | 0.1826 | 0.8039 | 0.8321 | 0.0084 |
| 0.30 | 0.2924 | 0.9158 | 0.9727 | 0.0021 |

Paired delta，v4 optimized mechanism minus total risk:

| budget | delta all-error addressed | delta VT/VF addressed | delta unresolved VT/VF |
|---:|---:|---:|---:|
| 0.05 | +0.0448 | +0.0437 | -0.0022 |
| 0.10 | +0.0168 | +0.0915 | -0.0026 |
| 0.20 | +0.0223 | +0.0465 | -0.0002 |
| 0.30 | +0.0223 | -0.0002 | +0.0001 |

解释：

- 5%、10%、20% budget 下，v4 平均优于总风险路由。
- 30% budget 下，VT/VF cross-error 捕获基本持平，但 all-error 捕获仍略高。
- v4 的提升最明显在 10% budget：VT/VF cross-error capture 从 0.5053 提升到 0.5969。

## Seed-level 稳定性

v4 optimized mechanism routing 相比总风险路由：

| budget | VT/VF capture 更好 seed 数 | all-error capture 更好 seed 数 | unresolved VT/VF 更好 seed 数 |
|---:|---:|---:|---:|
| 0.05 | 7/10 | 6/10 | 7/10 |
| 0.10 | 6/10 | 5/10 | 6/10 |
| 0.20 | 6/10 | 7/10 | 6/10 |
| 0.30 | 3/10 | 6/10 | 3/10 |

说明 v4 不是每个 seed 都赢，但在低到中等 budget 下，多数 seed 确实优于总风险路由。

## Validation 选择的 profile

不同 seed / budget 选择了不同机制组合：

- 5% budget：`vtvf_only`、`boundary_atypical`、`equal`、`representation_atypical`、`atypical_only` 都被选中过。
- 10% budget：`equal` 和 `boundary_atypical` 最常见。
- 20% budget：`equal`、`boundary_atypical`、`representation_heavy` 常见。
- 30% budget：`equal` 和 `representation_heavy` 常见。

这支持机制分型的必要性：不同 seed、不同预算下，最优机制组合不是固定的。

## 当前结论

已证实：

- 机制风险头稳定可学。
- validation-driven profile selection 能吃满 budget。
- v4 在 5%-20% budget 下平均优于总风险路由。
- 机制分型不只是解释结构，已经表现为实际路由收益。

仍需谨慎：

- v4 仍是内部 validation/test 研究验证，不是外部数据验证。
- profile set 是人工设计的，可继续扩展或用更正式的优化方法替代。
- seed-level 方差仍存在，尤其 30% budget 下收益趋于饱和。

## 可写成贡献点的表达

> We propose a mechanism-aware hierarchical reliability routing framework that decomposes ECG classification failures into distinct error mechanisms and uses validation-optimized mechanism profiles to allocate routing actions. Across 10 seeds, the optimized mechanism-aware router improves VT/VF cross-error capture over a scalar total-risk router at low-to-medium action budgets, while preserving mechanism-specific interpretability.

中文表达：

> 本研究不是将不确定性压缩为单一风险分数，而是将 ECG 分类错误分解为 VT/VF 边界混淆、SR-ventricular 方向错误、表征冲突、atypical signal 等机制，并通过 validation 优化机制组合，实现可解释且更有效的分层路由。
