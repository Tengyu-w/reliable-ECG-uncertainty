# v5c 两阶段机制分离式路由 10seed 结果

## 一句话结论

v5c 把我们前面讨论的结构真正串起来了：

```text
Stage 1: boundary-first VT/VF route
Stage 2: residual mechanism route
```

它不是把五六类分类方法压成两个分类器，而是把多类机制专家按优先级组织起来：先处理 VT/VF 高风险边界，再用剩余预算处理 SR-ventricular、representation conflict、atypical signal 等残余机制。

## v5c 做了什么

新增脚本：

```text
src/hierarchical_router_v5c.py
```

流程：

```text
Input ECG sample
  -> base classifier prediction
  -> Stage 1 boundary-first route
       evidence = mean(softmax VT/VF ambiguity,
                       validity gate x boundary,
                       wavelet VT/VF boundary risk)
       action = {VT,VF}
  -> Stage 2 residual mechanism route, only for remaining samples
       evidence = sr_ventricular risk
                + representation conflict risk
                + atypical signal risk
                + hidden confident risk if enabled
       action = mechanism-specific review
```

关键设计：v5c 保持总 action budget 不变。
如果 nominal budget 是 20%，Stage 1 先用掉 boundary route 需要的部分，Stage 2 只能使用剩余预算。

## 数据对齐

10 个 seed 全部完成，v4 evidence、wavelet evidence、validity evidence 的 `y_true` 均对齐。

| seeds | status | val/test aligned |
|---:|---|---|
| 42-51 | completed | yes |

这是同一 test set 上的 paired comparison。

## 主结果

| method | budget | action rate | Stage 1 boundary | Stage 2 residual | all-error capture | VT/VF capture |
|---|---:|---:|---:|---:|---:|---:|
| v4 optimized router | 10% | 10.0% | - | - | 57.7% | 59.7% |
| v5b boundary-only | 10% | 9.7% | 9.7% | 0.0% | 55.9% | 90.4% |
| v5c hierarchical | 10% | 9.7% | 9.7% | 0.0% | 55.9% | 90.4% |
| v4 optimized router | 20% | 20.0% | - | - | 82.6% | 87.9% |
| v5b boundary-only | 20% | 15.2% | 15.2% | 0.0% | 78.9% | 99.7% |
| v5c hierarchical | 20% | 18.9% | 15.2% | 3.6% | 81.2% | 99.7% |
| v4 optimized router | 30% | 30.0% | - | - | 93.8% | 97.2% |
| v5b boundary-only | 30% | 16.2% | 16.2% | 0.0% | 83.5% | 100.0% |
| v5c hierarchical | 30% | 30.0% | 16.2% | 13.8% | 94.9% | 100.0% |

## 与 v4 的 paired delta

| budget | delta action rate | delta all-error capture | delta VT/VF capture | delta unresolved VT/VF rate |
|---:|---:|---:|---:|---:|
| 5% | +0.00 points | -2.60 points | +38.12 points | -1.24 points |
| 10% | -0.28 points | -1.84 points | +30.69 points | -1.43 points |
| 20% | -1.14 points | -1.38 points | +11.87 points | -0.80 points |
| 30% | +0.00 points | +1.05 points | +2.75 points | -0.22 points |

解释：

- 低预算时，v5c 明确牺牲一点 all-error capture，换取大幅 VT/VF cross-error capture。
- 20% 时，Stage 2 开始补 residual errors，但 all-error 仍略低于 v4。
- 30% 时，Stage 1 已经捕获全部 VT/VF，Stage 2 有足够预算处理剩余错误，所以 all-error 和 VT/VF 都超过 v4。

## 为什么 10% 时 Stage 2 没启动？

因为 v5c 是总预算守恒的。10% nominal budget 下，Stage 1 boundary-first 已经用掉了 validation 预算，所以 Stage 2 没有剩余 quota。

这不是 bug，而是当前策略的一个清晰特性：

> 在低预算下，v5c 把有限预算优先给 VT/VF boundary risk。

如果论文目标强调 high-risk VT/VF safety，这个选择合理。
如果目标强调 all-error capture，可以在下一版做一个 reserved-residual-budget variant，例如：

```text
80% budget for Stage 1 boundary
20% budget reserved for Stage 2 residual
```

这可以作为后续 v5d。

## v5c 的机制意义

v5c 终于把“多类分类方法”变成了真正的分层系统：

```text
Stage 1 横向并列：
  - softmax boundary classifier
  - validity-domain boundary classifier
  - wavelet/time-frequency boundary classifier

Stage 2 横向并列：
  - SR-ventricular classifier
  - representation/prototype/KNN conflict classifier
  - atypical-signal classifier
  - hidden-confident classifier when available
```

两个 stage 不是两个分类方法，而是两个优先级层。
每个 stage 里面仍然保留多个机制专家。

## 论文可用表述

中文：

> 我们进一步构建了两阶段机制分离式路由。第一阶段使用 softmax、validity-domain 与 wavelet/time-frequency 三类边界证据优先识别 VT/VF boundary failures，并输出 `{VT,VF}` prediction set；第二阶段仅在剩余样本与剩余预算上执行 residual mechanism routing。10seed paired comparison 显示，在 10% action budget 下，v5c 相比 v4 router 将 VT/VF cross-error capture 从 59.7% 提升至 90.4%；在 30% budget 下，v5c 同时提升 all-error capture 与 VT/VF capture，分别达到 94.9% 与 100.0%。这表明高风险边界错误应当优先由 boundary-specific evidence branch 处理，而剩余错误再由机制特定路由分流。

English:

> We further implemented a two-stage mechanism-separated router. The first stage prioritizes VT/VF boundary failures using softmax ambiguity, validity-domain evidence, and wavelet/time-frequency boundary risk, producing a `{VT,VF}` prediction set. The second stage applies residual mechanism routing only to the remaining samples and remaining budget. Across ten paired duplicate-family seeds, v5c improved VT/VF cross-error capture from 59.7% to 90.4% at a 10% action budget compared with the v4 router. At a 30% budget, it improved both all-error capture and VT/VF capture, reaching 94.9% and 100.0%, respectively.

## 下一步

v5c 已经证明两阶段系统是成立的。下一步有两个选择：

1. 做 v5d：预留 residual budget，让 10%/20% 下也能兼顾 all-error。
2. 做 final method figure 和 ablation table，把 v5b/v5c 整理成论文方法图。

如果目标是继续冲结果，做 v5d。
如果目标是准备写博士申请/论文材料，先做方法图和 ablation 表。

## 输出文件

- `src/hierarchical_router_v5c.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5c/v5c_hierarchical_policy_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5c/paired_v5c_vs_baselines_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/hierarchical_router_v5c/v5c_alignment_manifest.csv`
