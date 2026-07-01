# 机制定向因果式量化全量结果与第三幕说明

日期：2026-07-01

这份文档用于把 2026-06-30 至 2026-07-01 完成的模型层机制实验整理成一个可以给导师解释的版本。它不是再给项目增加一个新名字，而是把已经分析过的 embedding、KNN 邻域、prototype、boundary、gate、ECG 波形规则性等机制变量，放进一条更清楚的证据链：

```text
do(training constraint)
  -> measured mechanism variable change
  -> model outcome change
```

核心结论是：模型层因果式多目标优化已经找到了比较清楚的有效机制，其中最稳的是 embedding/KNN 局部纯净度链条；prototype compactness、boundary ambiguity、gate-boundary joint 也有可量化支持。ECG 波形规则性这一支不能写成“训练改变了原始 ECG 波形”，而应写成“训练改变了模型对不可干预波形属性的编码能力和错误敏感性”。这就是本轮所谓的“第三幕”。

## 1. 这轮实验到底回答什么

导师指出的问题可以表述为：

```text
前期做了很多表征层、波形、模型结构、约束和错误机制分析，
但这些分析为什么能影响模型结果，需要定量证据链。
```

因此，本轮实验不是单纯比较哪个模型分数更高，也不是把模型分类器和 V5D 路由器混在一起比较。它回答的是：

1. 哪些训练约束是真正可干预变量；
2. 它们是否改变了预先定义的机制变量；
3. 这些机制变量变化是否和 outcome 同方向变化；
4. 哪些机制可以进入论文主线，哪些只能作为辅助或负结果。

## 2. 变量角色必须分清楚

### 2.1 可干预变量

本轮真正被 `do(...)` 改变的是训练约束或训练权重，而不是直接手动改 embedding、KNN purity 或 ECG 波形。

| candidate | 可干预训练约束 | 预期机制 |
| --- | --- | --- |
| `proto_center_only` | prototype center compactness loss | 类内紧凑性、整体表征几何 |
| `proto_center_margin` | prototype center + VT/VF margin | prototype geometry、VT/VF 边界 |
| `proto_margin_only` | VT/VF prototype margin | VT/VF prototype ambiguity |
| `contrastive_vtvf_light` | VT/VF boundary-aware contrastive loss | KNN local purity、VT/VF mixing |
| `boundary050` | boundary CE weight = 0.50 | softmax VT/VF boundary ambiguity |
| `boundary075` | boundary CE weight = 0.75 | softmax VT/VF boundary ambiguity |
| `gate_boundary_joint` | risk/validity gate 与 boundary 对齐 | gate-boundary alignment |
| `prototype_plus_contrastive` | prototype + contrastive joint loss | prototype-KNN joint mechanism |
| `boundary075_prototype` | boundary075 + prototype constraint | boundary-prototype joint mechanism |
| `regularity_aux_medium` | ECG regularity auxiliary loss | waveform-aware encoding / sensitivity |

### 2.2 机制变量

机制变量是训练后被测量出来的中间证据，不是直接被设置的旋钮。

| 机制家族 | 代表变量 | 回答的问题 |
| --- | --- | --- |
| embedding geometry | `silhouette_full`, `davies_bouldin_full`, `vt_vf_norm_dist` | 表征空间是否把 SR/VT/VF 分开 |
| KNN neighborhood | `local_purity_k_mean`, `knn_label_entropy_mean`, `knn_vtvf_mix_ventricular_mean` | 一个样本附近的邻居是否支持当前类别 |
| prototype ambiguity | `prototype_vtvf_ambiguity_ventricular_mean`, `prototype_vtvf_ambiguity_auroc` | VT/VF prototype 是否边界混淆 |
| softmax boundary | `entropy_mean`, `prob_margin_mean`, `softmax_vtvf_ambiguity_ventricular_mean` | 概率输出是否处在 VT/VF 模糊边界 |
| gate / validity | `validity_gate_any_error_auroc`, `boundary_score_any_error_auroc`, `gate_x_boundary_any_error_auroc` | gate 是否能识别不可靠或边界样本 |
| waveform regularity | `spectral_entropy`, `dominant_frequency`, `autocorr_peak`, `zero_crossing_rate`, `line_length` 等 | 原始 ECG 的波形属性如何影响模型编码和错误分布 |

### 2.3 不可干预变量

ECG 波形规则性特征是原始输入属性，不是训练能直接改变的东西。

包括：

```text
spectral_entropy
dominant_frequency
dominant_frequency_concentration
spectral_centroid
spectral_bandwidth
autocorr_peak
autocorr_peak_lag_s
zero_crossing_rate
line_length
```

正确写法：

```text
regularity auxiliary training changes how strongly model embeddings encode ECG regularity attributes,
and changes error sensitivity across waveform strata.
```

不能写成：

```text
regularity auxiliary training changes the ECG waveform itself.
```

### 2.4 Outcome

模型层 outcome 只和模型层比较：

```text
accuracy
macro_f1
ece
vtvf_cross_errors
total_errors
error_migration_penalty
```

V5D 是分类之后的 review/recover 路由机制，因此不能直接拿 V5D 的路由结果和一个纯分类模型的 accuracy 混比。公平比较必须分层：

| 层级 | 可以比较谁 |
| --- | --- |
| model-only | CNN vs CNN-LSTM vs PRO vs constrained / causal-Pareto model |
| evidence-head-only | entropy vs KNN vs prototype vs gate vs waveform evidence |
| router-only | V5D router vs optimized mechanism router |
| fixed-router downstream | 同一个 V5D router 接不同模型后的 downstream 结果 |

## 3. 全量实验完成情况

本轮全量机制定向实验已经完成：

| 项目 | 数值 |
| --- | ---: |
| completed runs | 33 / 33 |
| candidates | 11 |
| seeds | 3 (`42`, `43`, `44`) |
| paired candidate-seed deltas | 30 |
| measured mechanism variables | 32 |
| mechanism-outcome association rows | 192 |

主要结果目录：

```text
results/mechanism_targeted_causal_ablation_full_20260630/
results/mechanism_targeted_causal_quantification_full_20260630/
results/waveform_regularity_encoding_audit_20260701/
```

主要量化文件：

```text
run_level_mechanism_outcome_table.csv
paired_candidate_seed_mechanism_outcome_deltas.csv
intervention_to_mechanism_effects.csv
intervention_to_outcome_effects.csv
mechanism_to_outcome_association.csv
mediation_or_path_effect_summary.csv
causal_mechanism_variable_dictionary.csv
mechanism_targeted_verdict_table.csv
top_mechanism_outcome_associations.csv
```

## 4. 第一幕：哪些训练约束真的改善 outcome

与 baseline 同 seed 配对比较后，主要 candidate 的 outcome delta 如下。正数表示 accuracy/macro-F1 变好；负数表示 ECE、错误数或迁移惩罚变少。

| candidate | accuracy | macro-F1 | ECE | VT/VF cross | total errors | migration penalty | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `contrastive_vtvf_light` | +0.0443 | +0.0568 | -0.0263 | -14.67 | -189.33 | -106.83 | 强核心候选 |
| `proto_center_only` | +0.0415 | +0.0471 | -0.0244 | -20.67 | -178.00 | -109.33 | 强核心候选 |
| `proto_center_margin` | +0.0372 | +0.0442 | -0.0277 | -15.33 | -160.33 | -94.00 | 强核心候选 |
| `boundary075_prototype` | +0.0318 | +0.0430 | -0.0183 | -20.33 | -135.00 | -85.00 | 强核心候选 |
| `boundary075` | +0.0318 | +0.0230 | -0.0153 | -2.67 | -137.33 | -77.50 | 核心候选 |
| `prototype_plus_contrastive` | +0.0288 | +0.0186 | -0.0150 | +2.00 | -123.67 | -68.00 | 整体好，但 VT/VF cross 不干净 |
| `gate_boundary_joint` | +0.0232 | +0.0276 | -0.0065 | -1.33 | -96.67 | -50.67 | 核心候选，但要写成 joint |
| `boundary050` | +0.0180 | +0.0243 | -0.0097 | -15.33 | -77.67 | -54.00 | 核心候选 |
| `proto_margin_only` | +0.0099 | +0.0019 | -0.0071 | 0.00 | -43.67 | -26.83 | outcome 有改善，但目标机制不稳 |
| `regularity_aux_medium` | -0.0016 | +0.0039 | +0.0081 | -5.67 | +9.00 | -1.67 | 辅助证据，不能做核心模型提升 |

这里的结论是：本轮不是只找到一个“最好分数”，而是发现了不同机制有不同强度的证据。对于论文主线，`contrastive_vtvf_light`、`proto_center_only`、`proto_center_margin`、`boundary075_prototype` 更适合写成模型层有效机制；`regularity_aux_medium` 适合作为 ECG 波形结构解释的第三幕，而不是作为主模型最优项。

## 5. 第二幕：哪些机制变量解释了 outcome 改善

最强的机制-outcome 证据来自 KNN local purity。

| mechanism variable | outcome | Spearman r | p-value | 解释 |
| --- | ---: | ---: | ---: | --- |
| `local_purity_k_mean` | `total_errors` | -0.874 | 2.83e-10 | 局部纯净度越高，总错误越少 |
| `local_purity_k_mean` | `accuracy` | +0.853 | 2.21e-09 | 局部纯净度越高，准确率越高 |
| `local_purity_k_mean` | `error_migration_penalty` | -0.850 | 2.79e-09 | 局部纯净度越高，错误迁移惩罚越低 |
| `local_purity_k_mean` | `ece` | -0.844 | 4.74e-09 | 局部纯净度越高，校准误差越低 |
| `knn_label_entropy_mean` | `accuracy` | -0.735 | 3.71e-06 | 邻居标签熵越低，准确率越高 |
| `knn_label_entropy_mean` | `total_errors` | +0.728 | 5.12e-06 | 邻居越混乱，总错误越多 |
| `entropy_mean` | `accuracy` | -0.663 | 6.53e-05 | softmax 越不确定，准确率越低 |
| `entropy_mean` | `total_errors` | +0.644 | 1.22e-04 | softmax 越不确定，总错误越多 |

这给论文提供了最强的机制句子：

```text
The strongest internal mechanism evidence was observed for KNN local purity:
interventions that increased local neighborhood purity consistently reduced total errors,
error migration penalty and calibration error, while improving accuracy.
```

中文可以写成：

```text
最稳定的内部机制证据来自 KNN 邻域纯净度。能够提高局部邻域纯净度的训练约束，
通常同步降低总错误、错误迁移惩罚和校准误差，并提高分类准确率。
```

## 6. 机制链条逐条解释

### 6.1 KNN / representation chain

代表 candidate：

```text
contrastive_vtvf_light
proto_center_only
proto_center_margin
boundary075_prototype
```

证据链：

```text
do(contrastive / prototype compactness constraint)
  -> local_purity_k_mean increases
  -> knn_label_entropy_mean or VT/VF mixing decreases
  -> total errors, migration penalty, ECE decrease
```

这条链最适合作为论文主机制，因为它对应你前期做的大量 embedding 表征层分析：center distance、prototype distance、KNN 邻域密集度、KNN label entropy、VT/VF mixing 都属于同一类结构性证据。

### 6.2 Prototype compactness / geometry chain

代表 candidate：

```text
proto_center_only
proto_center_margin
boundary075_prototype
```

已观察到的目标机制变化包括：

```text
silhouette_full increases
local_purity_k_mean increases
prototype_vtvf_ambiguity_ventricular_mean decreases
```

解释：

```text
prototype / center constraints do not manually move one sample.
They reshape the learned embedding space so that samples from the same class are more compact,
and VT/VF prototype ambiguity becomes lower.
```

中文表达：

```text
prototype 类约束不是直接设置某个中心距离变量，而是把前期发现的“类中心不稳定、
VT/VF 原型边界模糊、类内分散”等失败机制，抽象成可干预训练目标。
训练后再测量 silhouette、local purity、prototype ambiguity，
用于验证该约束是否真的改变了目标机制。
```

### 6.3 Softmax boundary chain

代表 candidate：

```text
boundary050
boundary075
boundary075_prototype
```

目标机制变化：

```text
entropy_mean decreases
prob_margin_mean increases
softmax_vtvf_ambiguity_ventricular_mean decreases
```

证据链：

```text
do(boundary CE weighting)
  -> softmax VT/VF boundary ambiguity decreases
  -> confidence margin improves
  -> total errors and migration penalty decrease
```

这里可以和你前面关于“错误类型、边界样本、VT/VF 互相混淆”的分析衔接起来。它不是凭空加了一个 boundary loss，而是把前期观察到的错误集中区转化成训练中的可干预权重。

### 6.4 Gate-boundary chain

代表 candidate：

```text
gate_boundary_joint
```

需要谨慎写。结果显示 `gate_x_boundary_any_error_auroc` 明显改善，但 `validity_gate_any_error_auroc` 本身下降。因此不能写成“单独 gate 一定更好”，而应写成：

```text
The joint gate-boundary interaction became more informative,
although the standalone validity gate was not uniformly improved.
```

中文表达：

```text
gate-boundary joint 的证据支持“gate 与 boundary 的交互项”能更好定位错误，
但不支持把 validity gate 单独宣称为稳定增强机制。
```

### 6.5 Prototype plus contrastive chain

`prototype_plus_contrastive` 整体 accuracy、macro-F1、ECE、total errors 都改善，但 VT/VF cross-errors 平均增加 2。因此，如果论文目标强调 VT/VF 高风险混淆，它不应被写成最干净的核心方案。

更稳妥写法：

```text
prototype_plus_contrastive improved most global outcomes, but introduced a small adverse shift in VT/VF cross-errors,
suggesting that joint objectives require safety-oriented Pareto filtering rather than single-metric selection.
```

这反而能体现多目标优化的必要性：不能只看 accuracy 或 total errors，还必须同时看 VT/VF cross-errors、ECE 和 migration penalty。

### 6.6 Proto-margin-only negative / unstable result

`proto_margin_only` 的 outcome 有一些改善，但它没有稳定改变预设目标机制，因此 verdict 是 `negative_or_unstable`。

这说明：

```text
不是所有听起来合理的机制约束都能真正打中机制变量。
```

这类负结果对论文有价值，因为它能证明你不是事后挑一个好结果，而是在机制层面验证“约束是否真的改变目标机制”。

## 7. 第三幕：ECG 波形规则性的正确位置

### 7.1 为什么要单独做第三幕

你提出的关键问题是：如果 `spectral_entropy`、`dominant_frequency`、`autocorr_peak`、`line_length` 这些是 ECG 数据本身的属性，训练约束怎么可能“改变”它们？

这个问题非常重要。正确答案是：

```text
训练不能改变原始 ECG 波形属性；
训练只能改变模型如何编码、利用、或在这些波形属性分层上犯错。
```

因此我新增了一个波形规则性审计：

```text
python -m src.audit_waveform_regularity_encoding \
  --manifest results/mechanism_targeted_causal_ablation_full_20260630/mechanism_targeted_ablation_manifest_20260630_212510.csv \
  --out results/waveform_regularity_encoding_audit_20260701
```

该审计把 regularity features 明确定义为：

```text
non-intervenable ECG input attributes
```

同时测量两类模型侧结果：

```text
encoding metrics:
  model representation / logits ability to encode waveform attributes

error sensitivity metrics:
  model error behavior stratified by waveform attributes
```

### 7.2 Regularity auxiliary 是否改变了模型编码

对于 `regularity_aux_medium`，final embedding 对 9 个 ECG 波形属性的 Ridge R2 编码能力全部提高，并且每个属性都是 3/3 seeds 同方向为正。

| ECG regularity feature | final embedding R2 delta | positive seeds |
| --- | ---: | ---: |
| `line_length` | +0.1139 | 3/3 |
| `dominant_frequency` | +0.1089 | 3/3 |
| `autocorr_peak` | +0.0951 | 3/3 |
| `spectral_entropy` | +0.0674 | 3/3 |
| `spectral_centroid` | +0.0659 | 3/3 |
| `spectral_bandwidth` | +0.0656 | 3/3 |
| `dominant_frequency_concentration` | +0.0642 | 3/3 |
| `zero_crossing_rate` | +0.0602 | 3/3 |
| `autocorr_peak_lag_s` | +0.0447 | 3/3 |

这说明：

```text
regularity auxiliary loss does what it is designed to do at the representation level:
it makes final embeddings more informative about ECG waveform regularity attributes.
```

中文论文写法：

```text
regularity auxiliary 约束没有改变原始 ECG 波形，而是提高了模型最终表征对 ECG 波形规则性属性的可解码性。
这为“波形结构分析被模型利用”提供了内部定量证据。
```

### 7.3 Regularity auxiliary 是否改善错误分布

错误敏感性结果更复杂。它在若干高复杂度波形分层中降低了 VT/VF cross-rate：

| ECG stratum | high-q75 VT/VF cross-rate delta | high-minus-low VT/VF cross-rate delta | 判断 |
| --- | ---: | ---: | --- |
| `spectral_bandwidth` | -0.0658 | -0.0481 | 有利 |
| `line_length` | -0.0285 | -0.0518 | 有利 |
| `zero_crossing_rate` | -0.0265 | -0.0853 | 有利 |
| `dominant_frequency` | -0.0225 | +0.0343 | mixed |
| `autocorr_peak` | -0.0218 | -0.0186 | 有利 |
| `spectral_entropy` | -0.0059 | +0.0063 | weak / mixed |
| `spectral_centroid` | -0.0064 | +0.0116 | weak / mixed |
| `autocorr_peak_lag_s` | +0.0091 | +0.0162 | 不利 |

但全局 outcome 不强：

```text
accuracy: -0.0016
macro_f1: +0.0039
ece: +0.0081
vtvf_cross_errors: -5.67
total_errors: +9.00
error_migration_penalty: -1.67
```

因此第三幕的结论不是“regularity_aux 是最佳模型”，而是：

```text
regularity_aux_medium reliably increases waveform-attribute encoding in learned embeddings,
but this stronger encoding does not consistently translate into global classification or calibration improvement.
It should be treated as auxiliary waveform-structure evidence rather than a core model-improvement mechanism.
```

中文结论：

```text
regularity_aux_medium 证明了模型能够更强地编码 ECG 波形结构信息，
并在部分高复杂度波形分层中减少 VT/VF 混淆。
但它没有稳定改善全局 accuracy、ECE 和 total errors，
因此应作为“ECG 波形结构被模型利用的辅助证据”，而不是作为最终主模型。
```

## 8. 现在可以给老师怎么讲

可以用下面这段话：

```text
老师之前指出的问题是：我不能只是分析了很多 embedding、KNN、prototype、boundary 和 ECG 波形特征，
然后直接把它们放进模型或路由里，而应该证明这些机制真的会影响模型。

所以我现在把问题改成了机制定向的因果式量化：
在同一个 seed、同一个数据划分下，把 baseline 和只改变一个训练约束的 candidate 配对比较。
我先看这个训练约束是否改变了目标机制变量，比如 KNN local purity、prototype ambiguity、
softmax VT/VF ambiguity、gate-boundary alignment；
然后再看这些机制变量的变化是否和 accuracy、macro-F1、ECE、VT/VF cross-error、total errors 同步改变。

全量 33 次训练结果显示，KNN local purity 是最强的机制证据。
local_purity_k_mean 与 total errors 的 Spearman 相关为 -0.874，
与 accuracy 的相关为 +0.853，与 ECE 的相关为 -0.844。
这说明改善 embedding 邻域纯净度不是一个主观解释，而是与模型 outcome 改善高度一致。

此外，prototype compactness、boundary ambiguity 和 gate-boundary joint 也有配对干预证据。
ECG 波形规则性这一部分我没有把它写成可干预变量，
而是作为不可干预输入属性，验证训练是否改变模型对这些波形属性的编码和错误敏感性。
结果显示 regularity auxiliary 能提高 final embedding 对 9 个波形属性的可解码性，
但全局 outcome 改善不稳定，所以它是辅助机制证据，不是最终主模型。
```

## 9. 论文中建议采用的最终表述

### 中文

```text
本文将可靠 ECG 分类中的模型失败归因问题建模为机制定向的因果式多目标优化问题。
首先，基于前期错误分析定义 embedding geometry、KNN 邻域结构、prototype ambiguity、
softmax boundary ambiguity、validity gate 和 ECG waveform regularity 等机制变量。
其次，在相同数据划分和随机种子下构造 paired do-intervention proxy，
通过改变训练约束来观察机制变量与分类 outcome 的同步变化。
最后，采用多目标准则同时考察 accuracy、macro-F1、ECE、VT/VF cross-errors、
total errors 和 error migration penalty，从而避免单一准确率导向的选择偏差。

实验结果表明，KNN local purity 是最稳定的内部机制证据；
提高局部邻域纯净度的训练约束同步降低总错误、错误迁移惩罚和校准误差，并提高准确率。
prototype compactness 与 softmax boundary ambiguity 也表现出一致的机制-outcome 链条。
对于 ECG waveform regularity，本文将其视为不可干预输入属性，而非训练可改变的中介变量；
regularity auxiliary loss 能增强模型表征对波形属性的编码能力，但其全局 outcome 改善不稳定，
因此被定位为辅助解释证据。
```

### English

```text
We formulate reliable ECG classification as a mechanism-targeted causal-style multi-objective optimization problem.
Mechanism variables are first derived from failure analyses, including embedding geometry,
KNN neighborhood purity, prototype ambiguity, softmax boundary ambiguity, validity-gate alignment,
and ECG waveform regularity. Under matched random seeds and data splits, we then construct paired
do-intervention proxies by changing one training constraint at a time and measuring whether the intended
mechanism variables and downstream outcomes change coherently. Candidate interventions are selected under
multiple objectives, including accuracy, macro-F1, calibration error, VT/VF cross-errors, total errors,
and error migration penalty.

The strongest internal mechanism evidence is observed for KNN local purity: interventions that increase
local neighborhood purity consistently reduce total errors, error migration penalty, and calibration error,
while improving accuracy. Prototype compactness and softmax boundary ambiguity also provide coherent
mechanism-outcome chains. ECG waveform regularity is treated as a non-intervenable input attribute rather
than a manipulable mediator. The regularity auxiliary loss increases the decodability of waveform attributes
from learned embeddings, but does not consistently improve global classification and calibration outcomes;
therefore, it is used as auxiliary structural evidence rather than the primary model-improvement mechanism.
```

## 10. 仍然要保留的限制

这部分必须写，尤其是硕士论文或投稿时：

1. 目前是内部 paired evidence，不是外部临床验证。
2. 只有 3 个 seeds，能支持机制趋势，但不能过度声称稳定泛化。
3. 机制变量是 post-training measured mediators，因此应写作 causal-style evidence 或 paired do-intervention proxy，而不是严格自然科学因果证明。
4. 没有外部 ECG 数据集，后续只能用 OOD/stress split、record-level split 和 ECG-structure-preserving shift 来补充鲁棒性。
5. 波形规则性变量是不可干预输入属性，只能用于解释模型编码和错误分层，不能写成训练改变了原始 ECG。
6. 本项目是可靠性研究原型，不能表述为医学诊断系统或临床验证模型。

## 11. 下一步

第三幕后，最合理的下一步不是继续盲目训练新模型，而是把现有证据转成论文结构：

1. 方法章节：变量角色、paired do-intervention proxy、多目标选择；
2. 结果章节：全量 33-run 表格、核心 candidate、机制-outcome 相关；
3. 机制分析章节：KNN local purity、prototype geometry、boundary ambiguity、gate-boundary、waveform regularity；
4. 限制章节：内部数据、3 seeds、无外部验证、机制变量非严格中介证明；
5. 图表：一张总流程图、一张 outcome delta 表、一张 mechanism-outcome association heatmap、一张 waveform regularity encoding audit 图。

