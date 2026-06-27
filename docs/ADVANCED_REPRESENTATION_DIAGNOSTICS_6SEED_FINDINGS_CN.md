# 高级表征层诊断：6-seed CNN vs CNN-LSTM

本文件记录新增的六类高级表征分析。它们不是已有的 PCA、LDA、embedding centroid、KNN mixing 或 prototype ambiguity 复述，而是为了把项目从普通 embedding 可视化推进到更强的机制证据。

重要边界：以下结果是 ECG 可靠性研究原型证据，不是临床验证或医疗器械性能声明。

## 实验范围

- 数据划分：`duplicate_family`
- 模型：CNN vs CNN-LSTM
- paired seeds：42、43、44、45、46、47
- 每个 seed 对 CNN 和 CNN-LSTM 使用相同 split
- 输出目录：`results/cnn_lstm_baseline_20260626/advanced_representation_diagnostics_6seed/`

新增脚本：

```bash
python -m src.advanced_representation_diagnostics --mat RHYTHMS.mat --run-dir <run-dir> --model <model>
python -m src.aggregate_advanced_representation_diagnostics --run <seed> <model> <run-dir> --out <out-dir>
python -m src.plot_advanced_representation_diagnostics --root results/cnn_lstm_baseline_20260626/advanced_representation_diagnostics_6seed
```

可视化输出：

`results/cnn_lstm_baseline_20260626/advanced_representation_diagnostics_6seed/figures/`

| Figure | 作用 |
|---|---|
| `01_layerwise_vtvf_probe_auroc.png` | 看 VT/VF 信息在 CNN/CNN-LSTM 哪些层可线性读出 |
| `02_layerwise_multiclass_probe_macro_f1.png` | 看 SR/VT/VF 三分类信息随层级如何变化 |
| `03_layerwise_vtvf_probe_delta.png` | 看 CNN-LSTM 相对 CNN 在对应层是否增强 VT/VF 可读性 |
| `04_cross_seed_cka_svcca.png` | 看同一模型跨 seed 的表征一致性 |
| `05_same_seed_model_pair_cka_heatmap.png` | 看相同 seed 下 CNN 和 CNN-LSTM 表征相似性 |
| `06_vtvf_distribution_geometry.png` | 看 VT/VF 的 Mahalanobis、Fisher、Euclidean 几何 |
| `07_perturbation_stability_summary.png` | 看扰动下 embedding shift、prediction flip、prototype flip 和邻域保持 |
| `08_concept_alignment_correlation_heatmap.png` | 看 embedding 维度和 ECG regularity concepts 的相关性 |
| `09_concept_alignment_r2_heatmap.png` | 看 regularity concepts 能否由 embedding 线性重构 |
| `contact_sheet.png` | 所有图的总览 |

## 六个新增 target

1. **CKA/SVCCA 表征相似性**
   比较 CNN 和 CNN-LSTM 在相同 test windows 上是否学到相似表征，也比较同一模型跨 seed 是否稳定。

2. **Linear probing / VT-vs-VF probing**
   冻结表征，训练轻量 probe，检查 SR/VT/VF 和 VT/VF 信息是否能从表征中线性读出。

3. **Class-conditional covariance / Mahalanobis / Fisher geometry**
   不只看中心距离，还看类内协方差、Mahalanobis 距离和 Fisher ratio，判断“中心拉开”是否真的代表边界变好。

4. **Cross-seed representation consistency**
   统计同一模型跨 seed 的指标均值、方差和 CKA/SVCCA 相似性，判断表征机制是否稳定。

5. **Perturbation representation stability**
   对同一 test window 加轻量扰动，观察 embedding shift、prediction flip、prototype flip、neighbourhood preservation。

6. **Regularity concept alignment**
   检查 final embedding 是否线性携带 ECG regularity concepts，例如 spectral entropy、dominant frequency、autocorrelation、line length。

## 1. CKA/SVCCA 表征相似性

整体汇总：

| Comparison | Representation | CKA mean | SVCCA mean | SVCCA top directions |
|---|---:|---:|---:|---:|
| same-seed CNN vs CNN-LSTM | final embedding | 0.720 | 0.672 | 0.698 |
| same-seed CNN vs CNN-LSTM | classifier logits | 0.752 | 0.731 | 0.731 |
| same-model cross-seed | final embedding | 0.750 | 0.688 | 0.763 |
| same-model cross-seed | classifier logits | 0.762 | 0.716 | 0.716 |

按模型看跨 seed 表征一致性：

| Model | Representation | Cross-seed CKA | Cross-seed SVCCA |
|---|---|---:|---:|
| CNN | final embedding | 0.900 | 0.842 |
| CNN-LSTM | final embedding | 0.599 | 0.535 |
| CNN | classifier logits | 0.875 | 0.789 |
| CNN-LSTM | classifier logits | 0.650 | 0.643 |

解释：

- CNN 的 final embedding 跨 seed 更一致。
- CNN-LSTM 的表征子空间跨 seed 变化更大，说明组合时序结构并没有带来更稳定的内部机制。
- 相同 seed 下 CNN 和 CNN-LSTM 的表征相似性中等偏高，但不是同一表征；CNN-LSTM 的变化没有稳定转化为 VT/VF 可靠性提升。

## 2. Linear probing / VT-vs-VF probing

6-seed 均值：

| Model | Representation | Probe | Macro-F1 / AUROC | VT/VF cross-errors |
|---|---|---|---:|---:|
| CNN | final embedding | SR/VT/VF multiclass | 0.636 Macro-F1 | 213.5 |
| CNN-LSTM | final embedding | SR/VT/VF multiclass | 0.623 Macro-F1 | 211.0 |
| CNN | final embedding | VT vs VF binary | 0.769 AUROC | - |
| CNN-LSTM | final embedding | VT vs VF binary | 0.723 AUROC | - |
| CNN | classifier logits | VT vs VF binary | 0.775 AUROC | - |
| CNN-LSTM | classifier logits | VT vs VF binary | 0.763 AUROC | - |

解释：

- CNN-LSTM 没有让 VT/VF 信息更容易线性读出。
- final embedding 的 VT-vs-VF probe AUROC 反而从 CNN 的 0.769 降到 CNN-LSTM 的 0.723。
- 这支持“CNN-LSTM 不是稳定的 VT/VF 表征修复方案”。

### 2.1 真正的 layer-wise probing

追加脚本：

```bash
python -m src.layerwise_linear_probe --mat RHYTHMS.mat --run-dir <run-dir> --model cnn
python -m src.layerwise_linear_probe --mat RHYTHMS.mat --run-dir <run-dir> --model cnn_lstm
```

输出：

- 每个 run：`layerwise_linear_probe_summary.csv`
- 聚合表：`results/cnn_lstm_baseline_20260626/advanced_representation_diagnostics_6seed/layerwise_linear_probe_summary_6seed.csv`
- 成对差值：`layerwise_linear_probe_delta_summary.csv`

这一步和前面的 final embedding probe 不同。它直接取网络内部层：

- CNN：`conv1`、`pool1`、`conv2`、`pool2`、`conv3`、`pre_embedding_pool`、`final_embedding`、`classifier_logits`
- CNN-LSTM：`cnn_conv1`、`cnn_pool1`、`cnn_conv2`、`cnn_pool2`、`cnn_conv3`、`cnn_sequence`、`lstm_last_state`、`final_embedding`、`classifier_logits`

每一层都做两个 probe：

1. SR/VT/VF multiclass linear probe；
2. VT-vs-VF binary linear probe。

6-seed layer-wise 关键均值：

| Model | Layer | SR/VT/VF probe Macro-F1 | VT-vs-VF probe AUROC | VT/VF cross-errors |
|---|---|---:|---:|---:|
| CNN | conv1 | 0.593 | 0.721 | 213.0 |
| CNN | conv2 | 0.622 | 0.757 | 227.0 |
| CNN | conv3 | 0.649 | 0.765 | 220.5 |
| CNN | final_embedding | 0.635 | 0.769 | 213.3 |
| CNN | classifier_logits | 0.629 | 0.775 | 218.0 |
| CNN-LSTM | cnn_conv1 | 0.601 | 0.716 | 209.3 |
| CNN-LSTM | cnn_conv2 | 0.618 | 0.751 | 230.2 |
| CNN-LSTM | cnn_conv3 | 0.640 | 0.766 | 220.7 |
| CNN-LSTM | lstm_last_state | 0.621 | 0.729 | 209.8 |
| CNN-LSTM | final_embedding | 0.623 | 0.723 | 211.5 |
| CNN-LSTM | classifier_logits | 0.614 | 0.763 | 213.5 |

成对解释：

- 早期卷积层基本相近：CNN-LSTM 的 CNN front-end 没有明显比 CNN 更差，也没有明显更强。
- CNN-LSTM 的 `cnn_conv3` / `cnn_sequence` 层有接近 CNN `conv3` 的 VT-vs-VF AUROC，说明局部卷积表征中已经有 VT/VF 信息。
- 但是进入 `lstm_last_state` 和 `final_embedding` 后，VT-vs-VF AUROC 平均下降到 0.729 和 0.723。
- CNN 的 final embedding VT-vs-VF AUROC 为 0.769；CNN-LSTM final embedding 为 0.723。

这说明：

> CNN-LSTM 并不是没有学到 VT/VF 信息，而是 LSTM/后端整合没有稳定保留或放大 VT/VF 可线性读出的边界信息。它有时改善总体序列稳定性，但没有稳定改善 VT/VF decision-relevant representation。

## 3. Covariance / Mahalanobis / Fisher geometry

final embedding 的 VT/VF 几何：

| Model | VT/VF Mahalanobis distance | VT/VF Fisher ratio | Euclidean center distance |
|---|---:|---:|---:|
| CNN | 2.359 | 95.019 | 3.567 |
| CNN-LSTM | 1.990 | 95.798 | 3.414 |

解释：

- CNN-LSTM 没有增加 VT/VF Mahalanobis 分离，反而更低。
- Fisher ratio 基本持平，说明局部类内形状和边界结构没有被稳定改善。
- 这比单纯 center distance 更有说服力：即使某些 seed 的距离看起来变好，也不能说明分布形状和边界风险更好。

## 4. Cross-seed consistency

关键跨 seed 稳定性：

| Model | Metric | Mean | CV abs |
|---|---|---:|---:|
| CNN | VT/VF probe AUROC | 0.769 | 0.167 |
| CNN-LSTM | VT/VF probe AUROC | 0.723 | 0.303 |
| CNN | embedding probe Macro-F1 | 0.636 | 0.123 |
| CNN-LSTM | embedding probe Macro-F1 | 0.623 | 0.184 |
| CNN | final embedding cross-seed CKA | 0.900 | - |
| CNN-LSTM | final embedding cross-seed CKA | 0.599 | - |

解释：

- CNN-LSTM 的 probe 和表征相似性都更不稳定。
- 这和 6-seed 分类结论一致：CNN-LSTM 有时赢，但不是稳定机制。

## 5. Perturbation representation stability

对 test windows 加轻量扰动后的均值：

| Model | Embedding shift | Prediction flip | Prototype flip | Neighbour Jaccard |
|---|---:|---:|---:|---:|
| CNN | 4.349 | 0.153 | 0.210 | 0.043 |
| CNN-LSTM | 2.583 | 0.134 | 0.167 | 0.101 |

解释：

- CNN-LSTM 的扰动稳定性更好：embedding shift 更小，prediction/prototype flip 更少，KNN neighbourhood 保留更多。
- 但这个稳定性没有转化为更好的 VT/VF 单标签可靠性。
- 这给论文提供了一个更细的结论：**temporal modelling may improve perturbation stability, but perturbation stability alone is insufficient for VT/VF boundary reliability.**

## 6. Regularity concept alignment

final embedding 与 ECG regularity concepts 的对齐：

| Model | Max dim-feature correlation | Mean dim-feature correlation | Best ridge R2 | Mean ridge R2 |
|---|---:|---:|---:|---:|
| CNN | 0.841 | 0.557 | 0.033 | -0.032 |
| CNN-LSTM | 0.812 | 0.481 | 0.025 | -0.035 |

解释：

- final embedding 中有少量维度和 regularity concepts 高相关。
- 但整体线性可重构性很弱，mean R2 为负，说明 embedding 不是简单编码这些手工 regularity 特征。
- CNN-LSTM 的 concept alignment 没有更强，反而略弱。

## 综合结论

六类高级表征分析共同支持一个更强的机制结论：

> CNN-LSTM 并没有稳定产生更可读、更一致、更能分离 VT/VF 的表征。它在扰动稳定性上有优势，但在 VT/VF linear readability、Mahalanobis separation、cross-seed representation consistency 和 regularity concept alignment 上没有稳定优势。因此，VT/VF 可靠性问题不应继续被表述为“需要更复杂 backbone”。更合理的论文主线是：识别表征层局部混叠，并把这些证据转化为 frozen-backbone boundary calibration、prediction set 和 review routing。

## 对顶刊叙事的意义

这组结果可以把论文从工程式结论推进到机制式结论：

1. 更复杂时序模型并不等于更稳定的 VT/VF 表征。
2. 全局扰动稳定性和高风险边界可靠性不是同一个目标。
3. 表征层里存在可诊断的 VT/VF 局部混叠与跨 seed 不稳定。
4. 冻结 backbone 后做 boundary risk calibration 不是工程妥协，而是由表征机制证据推出的决策层方案。

## 仍需谨慎

- 当前高级诊断只覆盖 CNN vs CNN-LSTM 的 6 seeds。
- RISK-PRO++、Teacher、PRO 的高级诊断还可以追加，用来证明“几何增强不等于决策可靠”的机制链。
- 仍缺少外部数据验证，不能做临床泛化声明。
