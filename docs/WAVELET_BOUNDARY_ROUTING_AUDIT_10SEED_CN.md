# Wavelet/Time-Frequency Boundary Evidence 路由审计

## 一句话结论

这次没有把 `CNN+Wavelet+TCN+Boundary` 当成完整分类器去重跑，而是把它背后的 wavelet/time-frequency 分析方法拆出来，作为机制路由里的一个新证据专家。10seed 结果显示：**wavelet boundary risk 对 VT/VF 交叉错误非常敏感，适合作为 VT/VF boundary route，而不是先作为端到端分类器。**

## 做了什么

新增脚本：

```text
src/wavelet_boundary_routing_audit.py
```

它做的是：

1. 读取 v4 router 已经生成的 `evidence_scores_val.csv` 和 `evidence_scores_test.csv`。
2. 用相同 seed、相同 duplicate-family split 重建原始 ECG windows。
3. 使用模型里已有的 `FixedWaveletFilterBank1D`，提取固定多尺度 wavelet-like 响应。
4. 不训练深度模型，只在 validation set 上训练两个轻量风险头：
   - `wavelet_any_error_risk`
   - `wavelet_vtvf_boundary_risk`
5. 在 test set 上评估它们作为 review/routing score 的能力。
6. 把 `wavelet_vtvf_boundary_risk` 加进机制路由，形成第一版 `v5_wavelet_boundary_router`。

## 数据对齐

10 个 seed 全部完成，并且 val/test 的 `y_true` 都与 v4 router 对齐。

| seeds | val aligned | test aligned |
|---:|---|---|
| 42-51 | yes | yes |

这说明这次比较没有换测试集，也没有混用不同 split。

## Wavelet 风险头本身学到了什么

| risk head | target | AUROC mean | AUPR mean | 解释 |
|---|---|---:|---:|---|
| `wavelet_any_error_risk` | any error | 0.8886 | 0.4308 | 对总体错误有较强识别力 |
| `wavelet_vtvf_boundary_risk` | VT/VF cross-error | 0.9619 | 0.4628 | 对 VT/VF 交叉错误非常强 |

这说明 wavelet/time-frequency 证据不是“多加一个复杂模型”的噱头，而是真的能捕获 VT/VF boundary failure mode。

## 固定 budget 下的 routing/review 效果

| method | budget | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|
| `wavelet_vtvf_boundary_risk` | 5% | 29.0% | 59.4% |
| `wavelet_vtvf_boundary_risk` | 10% | 49.1% | 85.2% |
| `wavelet_vtvf_boundary_risk` | 20% | 69.6% | 97.1% |
| `wavelet_vtvf_boundary_risk` | 30% | 82.2% | 98.6% |

与之前很强的 `softmax_vtvf_ambiguity` 对比：

| method | budget | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|
| softmax VT/VF ambiguity | 10% | 52.3% | 84.1% |
| wavelet VT/VF boundary risk | 10% | 49.1% | 85.2% |
| softmax VT/VF ambiguity | 20% | 81.0% | 99.2% |
| wavelet VT/VF boundary risk | 20% | 69.6% | 97.1% |

解释：

- 10% budget 下，wavelet boundary risk 的 VT/VF 捕获略高于 softmax ambiguity。
- 20% budget 下，softmax ambiguity 略高，但 wavelet risk 仍然接近。
- wavelet 的 all-error capture 低于 softmax，说明它不是通用错误检测器，而是更专门的 VT/VF boundary evidence。

## 加入机制路由后的第一版 v5 结果

| method | budget | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|
| v4 optimized mechanism router | 10% | 57.7% | 59.7% |
| v5 wavelet boundary router | 10% | 61.4% | 68.0% |
| v4 optimized mechanism router | 20% | 82.6% | 87.9% |
| v5 wavelet boundary router | 20% | 84.7% | 93.5% |
| v4 optimized mechanism router | 30% | 93.8% | 97.2% |
| v5 wavelet boundary router | 30% | 95.1% | 99.7% |

paired mean improvement:

| comparison | budget | delta all-error capture | delta VT/VF capture |
|---|---:|---:|---:|
| v5 - v4 | 10% | +3.65 points | +8.34 points |
| v5 - v4 | 20% | +2.06 points | +5.59 points |

这说明把 wavelet boundary evidence 加进机制路由是有用的。

## 但也暴露了一个重要问题

`wavelet_vtvf_boundary_risk` 单独排序时：

- 10% budget 捕获 85.2% VT/VF cross-errors；
- 20% budget 捕获 97.1% VT/VF cross-errors。

但 v5 综合路由里：

- 10% budget 捕获 68.0%；
- 20% budget 捕获 93.5%。

这不是 wavelet 没用，而是第一版 v5 的 optimization 同时奖励：

- all-error capture
- VT/VF cross-error capture
- mechanism target capture

所以 wavelet 的 VT/VF 专门性被 all-error 目标稀释了。这个结果反而支持我们的机制路由假设：

> Wavelet/time-frequency evidence 应该作为一个专门的 VT/VF boundary route，而不是被混进一个泛化总风险路由里。

## 对分层决策系统的更新

当前合理结构应当是：

```text
ECG sample
 -> base classifier SR/VT/VF
 -> evidence layer
    -> softmax VT/VF ambiguity
    -> validity gate / boundary score
    -> embedding/prototype/KNN conflict
    -> regularity / atypical morphology
    -> model disagreement
    -> wavelet/time-frequency boundary risk
 -> mechanism router
    -> automatic single label
    -> {VT,VF} prediction set
    -> VT/VF boundary expert review
    -> representation conflict review
    -> atypical signal review
```

Wavelet 这条不应该输出“这是 VT 还是 VF”，而应该输出：

```text
这个样本是否具有时频尺度上的 VT/VF boundary risk？
```

如果是，它触发：

```text
{VT,VF} prediction set
```

或：

```text
VT/VF boundary expert review
```

## 对完整模型是否需要跑 10seed 的判断

现在不建议立刻跑完整 `CNN+Wavelet+TCN+Boundary` 10seed。原因是：

1. seed42 端到端模型已经暴露方向性失衡：大量 `VT -> VF`。
2. 当前 10seed 路由审计已经证明 wavelet evidence 本身很有价值。
3. 下一步更应该先做 boundary-first router，而不是把 wavelet 又塞回端到端分类器。

更合理的顺序是：

1. 先做 `v5b boundary-first router`：
   - `softmax_vtvf_ambiguity`
   - `validity_gate`
   - `boundary_score`
   - `wavelet_vtvf_boundary_risk`
2. 让它只优化 VT/VF boundary route，而不是同时优化所有错误。
3. 如果 v5b 在 10seed 下稳定优于 softmax-only 和 v4，再考虑跑完整 wavelet 模型。

## 论文可用表述

中文：

> 我们没有直接将 wavelet/time-frequency 分支作为更复杂的端到端分类器，而是首先将其作为机制路由证据进行审计。10seed 结果显示，固定 wavelet-like 响应训练得到的 `wavelet_vtvf_boundary_risk` 在 10% review budget 下捕获 85.2% 的 VT/VF cross-errors，在 20% budget 下捕获 97.1%。将该证据加入机制路由后，v5 router 相比 v4 router 在 10% budget 下提升 8.34 个百分点的 VT/VF cross-error capture，在 20% budget 下提升 5.59 个百分点。这说明 wavelet/time-frequency 信息更适合作为 VT/VF boundary-specific routing evidence，而不一定适合作为直接标签修正器。

English:

> Rather than using the wavelet/time-frequency branch as a more complex end-to-end classifier, we first audited it as mechanism-specific routing evidence. Across ten duplicate-family seeds, a fixed wavelet-derived boundary risk score captured 85.2% of VT/VF cross-errors at a 10% review budget and 97.1% at a 20% budget. Incorporating this evidence into the mechanism router improved VT/VF cross-error capture over the v4 router by 8.34 percentage points at 10% budget and 5.59 percentage points at 20% budget. These results support using wavelet/time-frequency cues as VT/VF boundary-specific routing evidence rather than direct label-correction constraints.

## 输出文件

- `src/wavelet_boundary_routing_audit.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/wavelet_boundary_routing_audit/wavelet_boundary_policy_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/wavelet_boundary_routing_audit/wavelet_risk_head_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/wavelet_boundary_routing_audit/paired_v5_wavelet_vs_baselines_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/wavelet_boundary_routing_audit/wavelet_boundary_alignment_manifest.csv`
