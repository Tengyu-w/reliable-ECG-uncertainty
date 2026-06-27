# v5d 主方法收束、Frozen SSL 对照与解释可靠性审计

## 一句话结论

这次补完了三个最值得继续做的升级：

1. **升级 3：Frozen self-supervised encoder comparison**
   已完成一个轻量自监督冻结编码器对照。它不是外部 ECG foundation model 验证，但可以作为 foundation-model readiness baseline。

2. **升级 5：Explanation reliability audit**
   已完成 10seed 解释可靠性量化，不再只是展示 embedding、regularity、waveform 图，而是检查每类解释证据是否真的对上对应错误机制。

3. **升级 2：v5d 设为最终主方法**
   已更新最终叙事：RISK 是 evidence layer 的核心分数之一，v5d mechanism-separated hierarchical router 才是最终 decision policy。

最终主线应写成：

```text
RISK / softmax / validity / wavelet / representation / regularity
    -> evidence layer
v5d mechanism-separated hierarchical router
    -> final decision policy
```

## 1. Frozen self-supervised encoder comparison

新增脚本：

```text
src/frozen_ssl_encoder_comparison.py
```

输出目录：

```text
results/frozen_ssl_encoder_comparison_20260627/
```

### 实验设计

这个实验不是下载外部 foundation model，而是先做一个轻量 frozen self-supervised encoder baseline：

- 使用 RHYTHMS 内部训练 split；
- 用 Barlow Twins 风格的自监督目标训练小型 1D CNN encoder；
- 不使用标签训练 encoder；
- 冻结 encoder；
- 在 frozen embedding 上训练浅层 logistic classifier；
- 在 validation set 上训练 any-error risk head 和 VT/VF-boundary risk head；
- 在 test set 上评估 accuracy、macro-F1、ECE、error AUROC、VT/VF capture、unresolved VT/VF rate。

运行设置：

```text
seeds = 42, 43, 44
split = duplicate-family
self-supervised epochs = 4
max SSL training windows = 4096
budgets = 10%, 20%, 30%
```

### 主要结果

| 指标 | mean | 解释 |
|---|---:|---|
| accuracy | 0.8937 | 分类能力不如当前最强 supervised backbone |
| macro-F1 | 0.6678 | VT/VF 类别仍有明显困难 |
| ECE | 0.0471 | 校准一般 |
| any-error AUROC | 0.9451 | frozen embedding 对错误检测很有用 |
| VT/VF-error AUROC | 0.9676 | 对 VT/VF boundary risk 很强 |

20% action budget 下：

| 方法 | all-error capture | VT/VF capture | unresolved VT/VF rate |
|---|---:|---:|---:|
| SSL any-error risk | 0.9164 | 0.9282 | 0.0036 |
| SSL VT/VF-boundary risk | 0.8249 | 0.9979 | 0.0001 |

### 解释

这个结果说明 frozen self-supervised encoder 有很强的 risk-ranking 潜力，尤其是 VT/VF boundary risk；但是它的分类性能并不够强，所以不应该替代当前 supervised backbone 或 v5d。

最稳妥的写法是：

> A lightweight frozen self-supervised encoder was used as a foundation-model readiness baseline. Although its classification performance was weaker than the supervised backbone, its frozen embeddings supported strong error and VT/VF-boundary risk ranking, suggesting that a real external ECG foundation model could be evaluated using the same frozen-head protocol in future work.

中文写法：

> 我们加入了一个轻量 frozen self-supervised encoder 对照，用于评估 foundation-model 方向是否值得继续。结果显示，该冻结编码器本身分类性能不如当前 supervised backbone，但其 embedding 对错误检测和 VT/VF boundary risk 排序具有较强价值。因此它更适合作为 foundation-ready evidence baseline，而不是替代 v5d 主方法。

## 2. Explanation reliability audit

新增脚本：

```text
src/explanation_reliability_audit.py
```

输出目录：

```text
results/evidence_informed_mechanism_routing_10seed_v4_20260627/explanation_reliability_audit/
```

### 实验设计

这个实验回答的问题是：

> 解释证据是否真的对上了它声称解释的错误机制？

因此它不再继续堆图，而是把解释证据分为几类：

- boundary explanation；
- representation explanation；
- regularity / atypicality explanation；
- second-opinion explanation；
- hidden-confidence explanation；
- SR-ventricular explanation。

然后分别评估它们对不同 error type 的：

- AUROC；
- AUPR；
- top 10% capture；
- top 20% capture；
- lift；
- v5d route-level precision。

### 主要结果

| evidence family | intended target | AUROC | AUPR | top 10% capture | top 20% capture |
|---|---|---:|---:|---:|---:|
| boundary explanation | VT/VF cross-error | 0.9646 | 0.4266 | 0.8249 | 0.9949 |
| representation explanation | representation conflict | 0.9701 | 0.4540 | 0.9011 | 0.9951 |
| second-opinion explanation | any error | 0.8180 | 0.4105 | 0.4881 | 0.7118 |
| SR-ventricular explanation | SR-ventricular error | 0.8843 | 0.3836 | 0.5973 | 0.7836 |
| regularity/atypicality explanation | atypical signal error | 0.6364 | 0.1600 | 0.2181 | 0.3959 |
| hidden-confidence explanation | hidden confident error | 0.8821 | 0.0335 | 0.1328 | 0.2274 |

### v5d route-level alignment

20% budget / 20% residual reserve 下：

| route | mean selected | all-error precision | target precision | VT/VF precision |
|---|---:|---:|---:|---:|
| boundary_first | 588.9 | 0.5117 | 0.3197 | 0.3197 |
| representation_conflict | 188.0 | 0.6170 | 0.5957 | 0.0000 |
| sr_ventricular | 286.6 | 0.2701 | 0.2545 | 0.0155 |

### 解释

最强的是 boundary explanation 和 representation explanation。它们不是“看起来合理”的图，而是真的能捕获对应 failure mechanism。

regularity/atypicality 和 hidden-confidence 证据要更谨慎：

- regularity 对 atypical/signal error 有方向性，但独立识别能力中等；
- hidden-confidence AUROC 高但 AUPR 很低，说明这类错误基率太低，不能单独当强证据；
- 它们适合作为 v5d residual route 的辅助 evidence，不适合作为单独主贡献。

论文可用写法：

> We evaluated explanation reliability by testing whether each evidence family preferentially identifies the failure mechanism it is intended to support. This turns interpretability from a visual plausibility claim into an error-mechanism alignment test. Boundary and representation explanations showed the strongest mechanism alignment, whereas regularity and hidden-confidence evidence were useful but weaker as stand-alone route justifications.

## 3. v5d 正式成为最终主方法

项目主线现在应从：

```text
RISK is the final contribution
```

改为：

```text
RISK is a core evidence score.
v5d is the final decision policy.
```

原因：

- RISK 只是把多源证据压成 review-priority score；
- v5d 才真正决定样本进入 `{VT,VF}` prediction set、residual mechanism review，还是 automatic single label；
- v5d 明确区分 VT/VF boundary failure 和 residual failure；
- v5d 还通过 reserved residual budget 解决了 v5c 低预算时 Stage 1 吃光预算的问题。

主结果：

| method | budget | all-error capture | VT/VF capture | unresolved VT/VF rate |
|---|---:|---:|---:|---:|
| v4 optimized router | 20% | 82.6% | 87.9% | 0.82% |
| v5d reserve 20% | 20% | 86.0% | 99.0% | 0.07% |

最终英文主贡献可以写成：

> We propose a mechanism-separated hierarchical reliability router for SR/VT/VF ECG classification. Multi-source reliability evidence, including RISK, softmax ambiguity, validity-domain evidence, wavelet/time-frequency boundary risk, representation conflict, and regularity signals, is first organized into an evidence layer. The final v5d policy then routes samples through a VT/VF boundary-first branch and a reserved-budget residual mechanism branch, producing either a `{VT,VF}` prediction set, mechanism-specific review, or automatic single-label output.

## 4. 现在三条升级的最终定位

| 升级 | 是否完成 | 最终定位 |
|---|---|---|
| Frozen/self-supervised encoder comparison | 已完成轻量内部版 | foundation-model readiness baseline，不替代主方法 |
| Explanation reliability audit | 已完成 10seed | 把 interpretability 从“图像展示”升级成 mechanism-alignment evidence |
| v5d as main method | 已完成叙事更新 | 最终 decision policy；RISK 是 evidence layer 的核心分数 |

## 5. 还不能过度声称什么

- frozen SSL 不是外部 ECG foundation model；
- 3seed frozen SSL 只是快速内部对照，不等于最终 foundation validation；
- explanation reliability 是内部机制一致性证据，不是医生解释一致性实验；
- v5d 仍然缺 external validation；
- 当前所有结果仍然不能写成 clinical validation 或 diagnostic claim。
