# v5d Reserved Residual Budget 10seed 结果

## 一句话结论

v5d 解决了 v5c 的一个预算权衡问题：低预算时，如果 Stage 1 boundary-first 把预算全部用完，VT/VF capture 很高，但 all-error capture 会略低。v5d 强制给 Stage 2 residual mechanism route 预留一部分预算后，**10% 和 20% budget 下可以同时超过 v4 的 all-error capture 和 VT/VF capture。**

最推荐的主系统版本是：

```text
v5d reserve 10%-20% residual budget
```

其中：

- 10% reserve 更偏 VT/VF safety；
- 20% reserve 更平衡 all-error 与 VT/VF。

## 做了什么

新增脚本：

```text
src/hierarchical_router_v5d_reserved_budget.py
```

v5d 仍然是两阶段系统：

```text
Stage 1: boundary-first route
  evidence = softmax + validity + wavelet
  action = {VT,VF}

Stage 2: residual mechanism route
  evidence = SR-ventricular / representation / atypical / hidden-confident
  action = mechanism-specific review
```

与 v5c 的区别是：

```text
v5c: Stage 1 先用预算，Stage 2 用剩下的
v5d: 先强制给 Stage 2 预留一部分 budget
```

本次扫描了：

```text
reserve fraction = 0%, 10%, 20%, 30%
```

## 主要结果：10% budget

| method | reserve | action rate | Stage 1 | Stage 2 | all-error capture | VT/VF capture |
|---|---:|---:|---:|---:|---:|---:|
| v4 optimized router | - | 10.0% | - | - | 57.7% | 59.7% |
| v5d | 0% | 9.7% | 9.7% | 0.0% | 55.9% | 90.4% |
| v5d | 10% | 10.0% | 8.9% | 1.1% | 57.9% | 87.2% |
| v5d | 20% | 10.0% | 8.0% | 2.0% | 59.2% | 83.9% |
| v5d | 30% | 10.0% | 7.0% | 3.0% | 60.3% | 79.8% |

解释：

- reserve 0% 最保守地保护 VT/VF，VT/VF capture 最高。
- reserve 10%-20% 开始补回 all-error capture。
- reserve 30% all-error 最高，但 VT/VF capture 明显下降。

相对 v4：

| reserve | delta all-error capture | delta VT/VF capture |
|---:|---:|---:|
| 0% | -1.84 points | +30.69 points |
| 10% | +0.19 points | +27.47 points |
| 20% | +1.48 points | +24.24 points |
| 30% | +2.53 points | +20.09 points |

所以 10% budget 下，推荐：

```text
reserve 10% if VT/VF safety is primary
reserve 20% if all-error balance is also important
```

## 主要结果：20% budget

| method | reserve | action rate | Stage 1 | Stage 2 | all-error capture | VT/VF capture |
|---|---:|---:|---:|---:|---:|---:|
| v4 optimized router | - | 20.0% | - | - | 82.6% | 87.9% |
| v5d | 0% | 18.9% | 15.2% | 3.6% | 81.2% | 99.7% |
| v5d | 10% | 20.0% | 14.5% | 5.5% | 84.2% | 99.4% |
| v5d | 20% | 20.0% | 13.5% | 6.5% | 86.0% | 99.0% |
| v5d | 30% | 20.0% | 12.5% | 7.5% | 86.3% | 98.4% |

相对 v4：

| reserve | delta all-error capture | delta VT/VF capture |
|---:|---:|---:|
| 0% | -1.38 points | +11.87 points |
| 10% | +1.56 points | +11.49 points |
| 20% | +3.42 points | +11.18 points |
| 30% | +3.65 points | +10.56 points |

20% budget 下，v5d 非常漂亮：所有 reserve 版本都大幅提升 VT/VF capture；10%-30% reserve 同时提升 all-error capture。

最平衡的是：

```text
reserve 20%
```

因为它保留 99.0% VT/VF capture，同时 all-error 比 v4 高 3.42 points。

## 主要结果：30% budget

30% budget 下，所有版本都接近：

| method | all-error capture | VT/VF capture |
|---|---:|---:|
| v4 optimized router | 93.8% | 97.2% |
| v5d reserve 0%-20% | 94.9% | 100.0% |
| v5d reserve 30% | 94.8% | 100.0% |

30% 时预算足够大，Stage 1 已经抓完 VT/VF，Stage 2 也有足够空间处理 residual errors，所以 reserve fraction 影响不大。

## 推荐主方法

如果只选一个主系统，我建议写：

```text
v5d with 20% residual budget reserve
```

理由：

1. 10% budget：
   - all-error 比 v4 高 +1.48 points；
   - VT/VF capture 比 v4 高 +24.24 points。
2. 20% budget：
   - all-error 比 v4 高 +3.42 points；
   - VT/VF capture 比 v4 高 +11.18 points。
3. 它不像 0% reserve 那样只偏向 VT/VF；
4. 也不像 30% reserve 那样牺牲太多 VT/VF capture。

也可以在论文里把 v5d-10% reserve 写成 safety-prioritized variant，把 v5d-20% reserve 写成 balanced variant。

## 方法学意义

v5d 把系统从：

```text
高风险边界优先
```

推进到：

```text
高风险边界优先 + 残余机制预算保障
```

这比 v5c 更像正式系统，因为它避免了低预算下 Stage 2 完全没有机会启动。

最终结构可以写成：

```text
Mechanism-Separated Hierarchical Routing with Reserved Residual Budget
```

中文：

```text
带残余预算保障的机制分离式分层可靠路由
```

## 论文可用表述

中文：

> 为避免 boundary-first route 在低预算下完全占用 review/action budget，我们进一步引入 residual-budget reservation。v5d 将总预算的一部分固定保留给 Stage 2 residual mechanism routing，其余预算用于 Stage 1 VT/VF boundary route。10seed paired comparison 显示，在 20% budget 下，v5d-20% residual reserve 同时提升 all-error capture 与 VT/VF cross-error capture，分别比 v4 router 高 3.42 和 11.18 个百分点。这表明分层可靠路由不仅需要高风险边界优先，还需要为非边界错误机制保留显式决策容量。

English:

> To prevent the boundary-first route from consuming the entire action budget under low-budget settings, we introduced residual-budget reservation. In v5d, a fixed fraction of the budget is reserved for Stage-2 residual mechanism routing, while the remaining budget is allocated to the Stage-1 VT/VF boundary route. Across ten paired duplicate-family seeds, v5d with a 20% residual reserve improved both all-error capture and VT/VF cross-error capture at a 20% budget, exceeding the v4 router by 3.42 and 11.18 percentage points, respectively. These results suggest that reliable hierarchical routing requires not only high-risk boundary prioritization but also explicit decision capacity for residual failure mechanisms.

## 与 v5c / stress test 的关系

- v5c 证明两阶段结构成立；
- stress test 证明 v5c 的 boundary score 对 validation downsample 不敏感，但 VT/VF errors 存在 cluster concentration；
- v5d 进一步解决 v5c 在低预算下 all-error 略低的问题。

因此最终叙事可以是：

```text
v5b: boundary-first branch 有效
v5c: 两阶段层级系统有效
stress test: 内部稳定但 cluster concentration 需要限制 claim
v5d: residual-budget reservation 让系统更均衡
```

## 输出文件

- `src/hierarchical_router_v5d_reserved_budget.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5d_reserved/v5d_reserved_policy_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5d_reserved/paired_v5d_reserved_vs_baselines_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5d_reserved/v5d_alignment_manifest.csv`
