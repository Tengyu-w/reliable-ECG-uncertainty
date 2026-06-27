# v5c 内部 Stress Test 结果

## 一句话结论

这次 stress test 给出的结论比较平衡：

> v5c 的强效果不是单纯由 validation 数据量不足造成的；但当前 VT/VF cross-errors 确实高度集中在少数 latent clusters 中，因此结果可能被当前数据分布放大，不能直接外推为外部泛化能力。

换句话说：

- **好消息**：validation downsample 到 25% 后，VT/VF 捕获率几乎不掉。
- **需要谨慎的地方**：VT/VF 错误本身高度集中，top 3 latent clusters 平均覆盖 98.4% 的 VT/VF cross-errors。

## 做了什么

新增脚本：

```text
src/internal_stress_test_v5c.py
```

做了两类内部 stress test：

1. **Cluster concentration stress**
   - 检查 v5c 是否只是在少数 latent clusters 上表现好。
   - 指标包括 top1/top3 cluster 错误占比、captured-error cluster concentration、去掉最大错误 cluster 后的捕获率。

2. **Validation downsample stress**
   - 把 validation evidence 缩小到：

```text
25%, 50%, 75%, 100%
```

   - 每个非 100% fraction 重复 30 次。
   - 重新估计 boundary-first 的 robust scaling，再在 test set 上评估。

注意：这是内部 stress test，不是外部数据库验证。

## Stress Test 1：Cluster Concentration

### 主要结果

| budget | VT/VF capture | all-error capture | top1 VT/VF-error cluster share | top3 VT/VF-error cluster share |
|---:|---:|---:|---:|---:|
| 5% | 62.8% | 29.2% | 70.7% | 98.4% |
| 10% | 90.4% | 55.9% | 70.7% | 98.4% |
| 20% | 99.7% | 81.2% | 70.7% | 98.4% |
| 30% | 100.0% | 94.9% | 70.7% | 98.4% |

### 这代表什么

VT/VF cross-errors 在当前数据里高度集中：

- 最大 latent cluster 平均包含 70.7% 的 VT/VF cross-errors；
- top 3 latent clusters 平均包含 98.4% 的 VT/VF cross-errors。

这说明 v5c 的高捕获率可能部分来自一个事实：

> 当前数据集里的 VT/VF 失败模式不是完全分散的，而是集中在少数 latent validity domains 里。

这对方法不是坏事，因为 reliability routing 本来就是要找 validity domains；但它限制了 claim：

> 不能直接说这个效果会在所有外部 ECG 数据集上同样强。

### 是否完全由最大 cluster 撑起来？

不是完全。去掉最大 VT/VF-error cluster 后：

- 10% budget 下，剩余 VT/VF capture 平均约 76.6%；
- 20% budget 下，剩余 VT/VF capture 平均约 99.3%；
- 30% budget 下，剩余 VT/VF capture 为 100%。

所以结果不是只靠一个 cluster。但是当前错误分布确实高度集中，必须在 limitation 里写。

## Stress Test 2：Validation Downsample

### 主要结果

| validation fraction | budget | VT/VF capture mean | VT/VF capture min | all-error capture mean |
|---:|---:|---:|---:|---:|
| 25% | 10% | 90.3% | 73.0% | 55.9% |
| 100% | 10% | 90.4% | 74.0% | 55.9% |
| 25% | 20% | 99.7% | 97.5% | 78.9% |
| 100% | 20% | 99.7% | 98.3% | 78.9% |

### 这代表什么

validation 数据减少到 25% 后，v5b/v5c Stage 1 的 boundary-first score 基本没有崩。

这说明当前强结果不太像是：

> 因为 validation set 刚好太小，所以 scaling 或 route threshold 偶然调中了。

更像是：

> softmax、validity、wavelet 三类边界证据本身在当前数据分布中确实能稳定识别 VT/VF boundary failures。

但注意，这只验证了内部 validation-size sensitivity，不验证外部泛化。

## 最稳的解释

确认事实：

1. v5c 在 10seed duplicate-family split 下稳定提高 VT/VF cross-error capture。
2. validation downsample 到 25% 后，boundary-first 捕获率基本保持。
3. VT/VF errors 在 latent clusters 上高度集中。

合理解释：

1. v5c 可能找到了当前数据中的 latent validity domains。
2. 这些 domains 被 softmax ambiguity、validity gate 和 wavelet boundary risk 同时识别。
3. 因此 boundary-first routing 在当前 split 上非常有效。

不能证明的事：

1. 不能证明它已经外部泛化。
2. 不能证明在更大、更复杂、多类别 ECG 数据上仍有同等效果。
3. 不能证明它具有临床验证意义。

## 对论文/申请材料的影响

这个 stress test 反而让叙事更严谨：

我们可以说：

> v5c is internally robust to validation downsampling, but its strong VT/VF capture is partly enabled by concentrated latent boundary-failure domains in the current dataset.

中文：

> v5c 对 validation 下采样具有内部稳定性，但其强 VT/VF 捕获效果部分依赖于当前数据集中较集中的 latent boundary-failure domains。

这比单纯说“效果很好”更像顶刊/博士申请里的成熟表达。

## 建议的 limitation 表述

中文：

> 尽管 v5c 在 10 个 duplicate-family split 中稳定提升了 VT/VF cross-error capture，内部 stress test 显示当前 VT/VF boundary failures 高度集中于少数 latent clusters。Validation downsampling 表明该方法对内部 calibration sample size 并不敏感，但 cluster concentration 结果提示其高捕获率可能部分受当前数据分布影响。因此，本结果应被视为内部可靠性证据，而非外部泛化或临床验证。

English:

> Although v5c consistently improved VT/VF cross-error capture across ten duplicate-family splits, internal stress testing showed that the VT/VF boundary failures were highly concentrated in a small number of latent clusters. Validation downsampling suggested that the boundary-first score was not sensitive to the internal calibration sample size, but the cluster concentration analysis indicates that the high capture rate may partly reflect concentrated failure domains in the current dataset. These findings should therefore be interpreted as internal reliability evidence rather than external generalization or clinical validation.

## 下一步

下一步最值得做的是 v5d 或 stress extension：

1. **v5d reserved residual budget**
   - 在 10%/20% budget 下强制给 Stage 2 留 10%-20% 的 quota；
   - 看能不能减少 all-error 损失，同时保留 VT/VF capture。

2. **harder cluster stress**
   - 在 validation 阶段模拟 leave-one-high-risk-cluster-out scaling；
   - 测试如果最大 risk cluster 没出现在 validation 中，boundary route 是否还稳。

3. **最终写法**
   - 主结果写 v5c；
   - limitation 写 cluster concentration；
   - stress table 放 supplemental/internal validation。

## 输出文件

- `src/internal_stress_test_v5c.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/internal_stress_v5c/cluster_concentration_stress_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/internal_stress_v5c/validation_downsample_stress_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/internal_stress_v5c/internal_stress_manifest.csv`
