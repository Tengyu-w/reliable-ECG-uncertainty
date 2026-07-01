# V5D / Stage1-Stage2 路由机制的因果-Pareto权重升级

对应脚本：

```text
python -m src.run_v5d_causal_pareto_weight_upgrade
```

本次已完成 20% budget 的十种子重点实验：

```text
python -m src.run_v5d_causal_pareto_weight_upgrade --budgets 0.20 --out results/v5d_causal_pareto_weight_upgrade_budget20_20260629
```

输出目录：

```text
results/v5d_causal_pareto_weight_upgrade_budget20_20260629/
```

## 1. 这次升级到底改了什么

这次不是另起一个新路由，也不是把 evidence head 和 V5D 比较。它保留原 V5D 的两阶段骨架：

```text
Stage 1: VT/VF boundary-first protection
Stage 2: residual mechanism recovery
```

可干预变量改为：

- `do(stage1_evidence_weights)`: 改变 stage1 边界证据权重；
- `do(stage2_residual_weights)`: 改变 stage2 residual 机制权重；
- `do(residual_reserve_fraction)`: 改变 stage1/stage2 的预算分配。

Stage1 纳入的证据包括：

- softmax VT/VF ambiguity；
- validity gate x boundary；
- wavelet VT/VF boundary risk；
- prototype VT/VF ambiguity；
- KNN VT/VF mixing。

Stage2 纳入的 residual 机制包括：

- `sr_ventricular`;
- `representation_conflict`;
- `atypical_signal`。

因此它是：

> V5D 内部权重升级，而不是 V5D 外部的新 policy。

## 2. 运行规模

| Item | Value |
| --- | ---: |
| Seeds | 10 |
| Budget | 20% |
| Stage1 profiles | 15 |
| Stage2 profiles | 10 |
| Reserve fractions | 0%, 10%, 20%, 30%, 40% |
| Candidate policies per budget per seed | 750 |
| Validation candidate rows | 7500 |
| Test candidate rows | 7500 |
| Aggregate Pareto rows | 31 |
| Stable aggregate Pareto rows | 4 |

## 3. 稳定 Pareto 候选

20% budget 下，验证集 Pareto 至少 5 个 seed 支持的稳定候选有 4 个：

| Candidate | Stage1 | Stage2 | Reserve | VT/VF capture | All-error capture | Auto VT/VF error | Auto error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stage1_softmax_boundary_only + stage2_sr_heavy` | softmax only | SR-heavy | 30% | 99.58% | 95.42% | 0.032% | 0.652% |
| `stage1_softmax_boundary_only + stage2_sr_representation_pair` | softmax only | SR+representation | 20% | 99.92% | 94.68% | 0.006% | 0.777% |
| `stage1_validity_heavy + stage2_sr_heavy` | validity-heavy | SR-heavy | 40% | 98.99% | 95.58% | 0.077% | 0.626% |
| `stage1_wavelet_heavy + stage2_atypical_only` | wavelet-heavy | atypical-only | 30% | 100.00% | 94.56% | 0.000% | 0.307% |

注意：这些数值是 validation-Pareto 支持 seed 子集上的 aggregate 结果，不能单独写成“全 10 seed 全面胜出”。

## 4. 全 10 seed 与原始 V5D 的真实比较

原始 V5D 在 20% budget 下：

| Original V5D | VT/VF capture | All-error capture | Auto VT/VF error | Auto error | Stage1 rate | Stage2 rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reserve 0% | 99.72% | 81.25% | 0.0195% | 2.20% | 15.25% | 3.61% |
| reserve 20% | 99.04% | 86.04% | 0.0673% | 1.79% | 13.54% | 6.46% |
| reserve 30% | 98.42% | 86.28% | 0.1054% | 1.69% | 12.53% | 7.47% |

全 10 seed 平均后，最稳妥的升级不是激进改 stage1，而是：

```text
stage1_original_softmax_validity_wavelet
+ stage2_sr_heavy
+ reserve_20pct
```

它相对原始 V5D reserve 20% 的变化为：

| Metric | Change |
| --- | ---: |
| VT/VF capture | 0.000 pp |
| All-error capture | +0.086 pp |
| Auto VT/VF error | 0.000 pp |
| Auto error reduction | +0.014 pp |

另一个相近候选：

```text
stage1_original_softmax_validity_wavelet
+ stage2_sr_ventricular_only
+ reserve_20pct
```

相对原始 V5D reserve 20%：

| Metric | Change |
| --- | ---: |
| VT/VF capture | 0.000 pp |
| All-error capture | +0.067 pp |
| Auto VT/VF error | 0.000 pp |
| Auto error reduction | +0.013 pp |

这说明：当前最可辩护的“V5D 升级”是小幅 stage2 权重校正，而不是大幅改掉 stage1。

## 5. 结论

本次实验得到的结论是：

> V5D 的 stage1 VT/VF boundary 设计本身很强，不宜轻易替换。因果-Pareto 权重搜索显示，真正稳妥的升级空间主要在 stage2 residual 权重，尤其是提高 SR-ventricular residual route 的优先级。在不降低 VT/VF capture、不增加自动残余 VT/VF 错误的前提下，可以获得很小但方向一致的 all-error capture 和 overall auto-error 改善。

换句话说：

- 如果目标是守住 VT/VF，原始 V5D 已经很强；
- 如果目标是多目标平衡，Pareto搜索揭示了 stage1/stage2 budget 的取舍；
- 如果目标是“保守升级 V5D”，当前证据支持 stage2 SR-ventricular 权重上调；
- 如果目标是“显著超越 V5D”，当前结果还不够。

## 6. 论文口径

建议写法：

> We further upgrade the existing V5D stage1-stage2 router by treating stage-wise evidence weights and residual reserve allocation as intervention variables. The Pareto search suggests that the original VT/VF boundary-first stage is already highly competitive, while modest improvements can be obtained by reweighting the residual stage toward SR-ventricular errors. The improvement is small but preserves VT/VF capture, supporting the proposed causal-Pareto framework as a mechanism-level tuning tool rather than a wholesale replacement of V5D.

中文写法：

> 本文进一步在既有 V5D 两阶段路由骨架内部进行因果-Pareto权重优化。结果显示，原 stage1 的 VT/VF boundary-first 设计已经较强，最稳妥的升级不是替换 stage1，而是在保持原 stage1 证据组合不变的条件下，对 stage2 residual 机制进行权重校正，尤其提高 SR-ventricular route 的优先级。该升级在不降低 VT/VF capture 的条件下带来小幅 all-error capture 和 overall residual risk 改善。

## 7. 限制与下一步

- 当前只完成 20% budget 十种子重点实验；
- 稳定 Pareto 候选在子集 seed 上表现很强，但全 10 seed 不一定全面胜出；
- 最稳妥升级收益很小，需要更多 budget、OOD-style shift 和 route-level error migration 支撑；
- 仍是内部 duplicate-family 验证，不是外部临床验证；
- 下一步应跑 5%、10%、30% budget，并检查 stage2 SR-ventricular 权重上调到底捕获了哪些错误。
