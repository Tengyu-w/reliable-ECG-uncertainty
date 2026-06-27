# 机制分型分层路由 10-seed 验证

## 目的

验证 `mechanism-aware hierarchical reliability routing` 是否真的有用，而不是只在 seed42 上看起来合理。

本次实验使用 seed42-51，共 10 个 seed。每个 seed 使用：

- 主模型：`core_regularity_injection`
- second-opinion：`core_risk_pro_readable`

输出目录：

`results/evidence_informed_mechanism_routing_10seed_20260627/`

运行脚本：

```powershell
python -m src.run_mechanism_routing_10seed `
  --out results\evidence_informed_mechanism_routing_10seed_20260627 `
  --budgets 0.05 0.10 0.20 0.30 `
  --force
```

## 核心结论

机制风险头本身是有用的，而且跨 seed 稳定；但当前 v3b 的固定预算分配策略没有整体打赢“总风险路由”。这说明：

- “按错误机制建专家头”是成立的。
- “怎么把预算分配给不同机制门”还没优化好。
- 下一步应该做 validation-driven budget allocation，而不是继续手写机制权重。

## 机制风险头稳定性

| mechanism | AUROC mean | AUROC std | AUPR mean | AUPR std | enabled |
|---|---:|---:|---:|---:|---:|
| representation conflict | 0.9899 | 0.0076 | 0.6857 | 0.1682 | 10/10 |
| atypical signal | 0.9492 | 0.0380 | 0.5606 | 0.2343 | 10/10 |
| VT/VF boundary | 0.9539 | 0.0297 | 0.3872 | 0.1148 | 10/10 |
| SR-ventricular | 0.9126 | 0.0948 | 0.5095 | 0.1841 | 10/10 |
| hidden confident | 0.5000 | 0.0000 | 0.0079 | 0.0107 | 0/10 |

解释：

- `representation_conflict` 是目前最强、最稳定的机制头。
- `atypical_signal` 和 `VT/VF boundary` 也稳定可学。
- `hidden_confident` 在 validation 中几乎没有正例，不能作为当前路由门，只能保留为失败模式分析。

## 机制路由 v3b 整体表现

| budget | action rate mean | VT/VF set rate mean | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0481 | 0.0286 | 0.2329 | 0.2070 | 0.0349 |
| 0.10 | 0.0734 | 0.0462 | 0.3733 | 0.3789 | 0.0278 |
| 0.20 | 0.1164 | 0.0797 | 0.6309 | 0.6889 | 0.0146 |
| 0.30 | 0.1491 | 0.1061 | 0.7625 | 0.8240 | 0.0095 |

注意：v3b 的 action rate 明显低于 nominal budget，尤其在 20%/30% 预算时只用了约 11.6%/14.9%。这说明固定机制阈值和候选约束让预算没有被充分使用。

## 与总风险路由对比

总风险路由：

| budget | review rate | VT/VF set rate | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0371 | 0.0047 | 0.2732 | 0.2020 | 0.0358 |
| 0.10 | 0.0921 | 0.0031 | 0.5607 | 0.5053 | 0.0230 |
| 0.20 | 0.1826 | 0.0002 | 0.8041 | 0.8317 | 0.0084 |
| 0.30 | 0.2920 | 0.0001 | 0.9159 | 0.9722 | 0.0022 |

Paired delta，机制路由减去总风险路由：

| budget | delta VT/VF addressed | delta all-error addressed | delta unresolved VT/VF |
|---:|---:|---:|---:|
| 0.05 | +0.0050 | -0.0403 | -0.0008 |
| 0.10 | -0.1265 | -0.1874 | +0.0048 |
| 0.20 | -0.1428 | -0.1732 | +0.0062 |
| 0.30 | -0.1482 | -0.1534 | +0.0073 |

解释：

- 当前 v3b 在 10%-30% budget 下，总捕获率不如总风险路由。
- 这不是机制头不可用，而是 v3b 的策略没有把预算吃满，且机制权重是手写的。
- 在 20%-30% 区间，机制路由的“每单位 action rate 捕获效率”反而更高，但因为 action rate 太低，总捕获量落后。

## 证据消融

VT/VF cross-error，10% budget：

| feature group | AUROC mean | AUPR mean | captured mean | unresolved VT/VF rate |
|---|---:|---:|---:|---:|
| softmax only | 0.9572 | 0.3987 | 0.7808 | 0.0157 |
| all evidence | 0.9559 | 0.3939 | 0.7660 | 0.0164 |
| regularity only | 0.9478 | 0.3633 | 0.7567 | 0.0147 |
| historical diagnostics | 0.9439 | 0.3429 | 0.7554 | 0.0167 |
| latent cluster only | 0.9501 | 0.3610 | 0.7518 | 0.0161 |
| model disagreement only | 0.9465 | 0.3648 | 0.7493 | 0.0171 |
| representation only | 0.9515 | 0.3912 | 0.7415 | 0.0183 |

这说明 VT/VF cross-error 不是只能靠表征层。softmax、regularity、historical diagnostics、latent cluster、representation 都有可用信号。这个支持机制分型的设计基础：不同证据组应该进入不同机制门，而不是全部混成一个解释不清的总分。

## 目前判断

已证实：

- 机制头可学，而且多 seed 稳定。
- 机制分型能把错误拆成不同可解释通道。
- 60-100 个证据特征确实不应该只塞进一个总风险分数。

尚未证实：

- 当前机制路由策略本身优于总风险路由。
- 手写预算权重是合理的。
- hidden-confident 机制能在当前 validation 规模下稳定学习。

## 下一步

最应该做的是 v4：

1. 用 validation split 自动学习机制预算分配，而不是手写 0.35/0.20/0.15/0.20。
2. 让机制路由吃满目标 action budget。
3. 同时优化两个目标：VT/VF cross-error capture 和 all-error capture。
4. 加入 paired 10-seed comparison，报告机制路由 v4 是否显著优于总风险路由。

这一步完成后，才能说“机制分型分层路由不仅可解释，而且有效”。
