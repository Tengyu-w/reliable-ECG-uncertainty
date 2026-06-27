# 四个顶刊级可靠性方向的 10seed 分析总结

日期：2026-06-26

## 分析对象

本轮分析不是重新训练模型，而是用已有输出对三类模型体系做统一 post-hoc 机制分析：

- 传统 CNN：10 个 paired seeds，seed42 到 seed51。
- CNN-LSTM：同一批 paired seeds，和 CNN 成对比较。
- ProRisk：这里指当前 Risk-Pro-readable/ProRisk 版本，并与 frozen-backbone Teacher 成对比较。

输出目录：

- `results/top_journal_reliability_directions_20260626/`
- 主脚本：`src/top_journal_reliability_directions.py`

本轮覆盖 40 个 run：CNN 10、CNN-LSTM 10、Teacher 10、ProRisk 10。

## 方向一：Representation Underspecification Atlas

这个方向回答的问题是：不同模型是否只是 accuracy 不同，还是在表征空间、VT/VF 边界、错误稳定性之间存在欠定性？

核心输出：

- `direction1_representation_underspecification_atlas.csv`
- `direction1_atlas_model_summary.csv`
- `direction1_paired_delta_summary.csv`
- `direction1_representation_boundary_correlations.csv`

关键结果：

| 对比 | 指标 | mean delta | bootstrap 95% CI | 解释 |
|---|---:|---:|---:|---|
| CNN-LSTM - CNN | accuracy | -0.0131 | [-0.0367, 0.0113] | 没有稳定 accuracy 提升 |
| CNN-LSTM - CNN | macro-F1 | +0.0216 | [-0.0250, 0.0639] | 有改善趋势但不确定 |
| CNN-LSTM - CNN | VT/VF cross-errors | -49.3 | [-105.0, -5.3] | 最稳定的收益是减少 VT/VF 互错 |
| CNN-LSTM - CNN | embedding silhouette | +0.1280 | [0.0689, 0.1953] | 表征聚类结构明显更规整 |
| CNN-LSTM - CNN | softmax VT/VF boundary AUROC | -0.0361 | [-0.0904, 0.0125] | 边界可检测性没有同步提高 |
| ProRisk - Teacher | accuracy | -0.0046 | [-0.0271, 0.0178] | 没有稳定 accuracy 提升 |
| ProRisk - Teacher | VT/VF cross-errors | +8.5 | [-12.9, 32.8] | 没有减少 VT/VF 互错 |
| ProRisk - Teacher | embedding silhouette | +0.0865 | [0.0329, 0.1370] | 表征几何更规整 |
| ProRisk - Teacher | KNN VT/VF boundary AUROC | -0.0283 | [-0.0749, 0.0217] | 局部边界风险信号变弱 |

确认事实：

1. CNN-LSTM 和 ProRisk 都提升了 embedding silhouette，说明表征空间更规整。
2. 这种几何规整没有稳定转化为 accuracy、ECE 或边界风险可读性的全面提升。
3. CNN-LSTM 的主要收益集中在 VT/VF cross-errors 减少，而不是整体错误减少。
4. ProRisk 的主要变化是表征更规整、校准略改善，但 VT/VF cross-errors 和 KNN 边界可检测性没有改善。

论文层面的解释：

这支持“representation stability is not boundary reliability”。表征层变得更紧、更平滑，不等于决策边界更可靠。尤其 ProRisk 证明了一个关键负结果：几何正则可以同时规整正确样本和错误样本，所以模型会形成更稳定但未必更正确的错误区域。

## 方向二：Hidden Rhythm Stratification

这个方向回答的问题是：错误是不是均匀分布的，还是集中在某些隐含节律/记录簇中？

核心输出：

- `direction2_hidden_rhythm_strata_clusters.csv`
- `direction2_hidden_rhythm_strata_summary.csv`
- 每个 run 的样本级 cluster 文件：`latent_strata_samples_*.csv`

关键 cluster-level 结果：

| 模型 | mean cluster error rate | max cluster error rate | mean high-conf error rate | max high-conf error rate |
|---|---:|---:|---:|---:|
| CNN | 0.2315 | 1.0000 | 0.0687 | 1.0000 |
| CNN-LSTM | 0.3165 | 1.0000 | 0.0781 | 0.9970 |
| Teacher | 0.1815 | 0.7872 | 0.0584 | 0.6099 |
| ProRisk | 0.2050 | 0.9869 | 0.0925 | 0.8668 |

确认事实：

1. 四类模型都存在高错误率 latent clusters。
2. ProRisk 的平均 cluster error rate 低于 CNN/CNN-LSTM，但 high-confidence error cluster 更明显。
3. CNN-LSTM 虽然减少 VT/VF cross-errors，但仍存在错误高度集中的 cluster，说明序列平滑没有消除隐含子群风险。
4. 某些 cluster 的错误率接近 1，且 confidence 很高，这类区域比普通平均错误更适合成为论文中的 failure-domain evidence。

论文层面的解释：

这说明模型失败不是随机噪声，而是存在 latent validity domains：某些隐含节律/记录簇天然落在模型有效域之外。由于没有患者 ID，这里不能说“患者亚群”，只能说“latent record/rhythm clusters”或“duplicate-family-controlled latent strata”。

## 方向三：Validity-Domain Map

这个方向回答的问题是：我们能不能用表征层和不确定性信号，训练一个 frozen-feature 的有效域检测器，判断哪些样本更可能不可靠？

方法：

用训练集 frozen embeddings 和模型自身 logits 构建 post-hoc validity classifier，在测试集评估。特征包括：

- confidence、margin、entropy；
- softmax VT/VF ambiguity；
- prototype VT/VF ambiguity；
- KNN entropy、KNN VT/VF mixing；
- centroid distance、centroid margin；
- ventricular-neighbor fraction、KNN prediction agreement。

核心输出：

- `direction3_validity_domain_scores.csv`
- `direction3_validity_domain_review_curves.csv`
- `direction3_validity_domain_summary.csv`
- 每个 run 的样本级 risk score：`validity_scores_*.csv`

关键结果：

| 模型 | target | AUROC mean | AUPR mean | 解释 |
|---|---|---:|---:|---|
| CNN | any error | 0.9308 | 0.6944 | 任意错误很可检测 |
| CNN-LSTM | any error | 0.9370 | 0.7266 | 任意错误检测最强 |
| Teacher | any error | 0.9128 | 0.5353 | 可检测，但 AUPR 较弱 |
| ProRisk | any error | 0.9268 | 0.5838 | 比 Teacher 略强 |
| CNN | VT/VF boundary error | 0.7144 | 0.5637 | 边界错误中等可检测 |
| CNN-LSTM | VT/VF boundary error | 0.6910 | 0.4474 | 边界错误更少，但不更可读 |
| Teacher | VT/VF boundary error | 0.6991 | 0.4675 | 中等 |
| ProRisk | VT/VF boundary error | 0.6754 | 0.4589 | 边界有效域没有改善 |
| CNN-LSTM | confident VT/VF boundary error | 0.8815 | 0.3074 | 高置信危险错可检测，但基率低 |
| Teacher | confident VT/VF boundary error | 0.8731 | 0.3422 | 高置信危险错可检测 |

确认事实：

1. 任意错误比 VT/VF boundary error 更容易被表征/不确定性特征检测。
2. CNN-LSTM 对 any-error validity map 很强，但对 VT/VF boundary error 的 AUROC 不如 CNN。
3. ProRisk 没有提升 VT/VF boundary validity map，甚至 KNN 边界信号弱于 Teacher。
4. 高置信 VT/VF boundary error 的 AUROC 较高，但 AUPR 不高，说明这类错误少、检测难度仍在。

论文层面的解释：

这给下一步模型升级一个很清楚的方向：不要继续把很多约束混进主分类器里，而是保留 frozen backbone/teacher 表征，在其上训练一个独立 validity/approval head。这个 head 的目标不是提高原始 accuracy，而是显式识别模型在哪些区域不可靠。

## 方向四：Model Disagreement / Synthetic Second Opinion

这个方向回答的问题是：两个机制不同的模型在同一 split 上的分歧，能不能作为“第二意见”信号捕获错误？

核心输出：

- `direction4_model_disagreement_second_opinion.csv`
- `direction4_model_disagreement_scores.csv`
- `direction4_model_disagreement_summary.csv`

关键结果：

| 对比 | disagreement rate | any-error AUROC | VT/VF boundary AUROC | top 10% error capture | top 10% VT/VF-boundary capture |
|---|---:|---:|---:|---:|---:|
| CNN vs CNN-LSTM | 0.1033 | 0.9082 | 0.7619 | 0.4791 | 0.1775 |
| Teacher vs ProRisk | 0.0646 | 0.8230 | 0.6920 | 0.5870 | 0.2191 |

确认事实：

1. 模型分歧是很强的 error-risk signal，尤其 CNN vs CNN-LSTM。
2. CNN vs CNN-LSTM 的分歧对 VT/VF boundary error 的 AUROC 达到 0.7619，比单模型 softmax/KNN 边界信号更有潜力。
3. Teacher vs ProRisk 分歧率较低，但 top 10% review budget 能捕获较高比例的 any-model errors。
4. 仍存在 both-agree-wrong：CNN vs CNN-LSTM 平均 371.7，Teacher vs ProRisk 平均 293.8。这说明第二意见有帮助，但不能完全替代 validity-domain analysis。

论文层面的解释：

这支持一个更强的论文叙事：不是单个模型“更好”，而是不同机制模型暴露不同的有效域边界。模型间分歧可以作为 synthetic second opinion，用来定位隐含高风险区域。

## 总体结论

本轮四个方向把 CNN、CNN-LSTM、ProRisk 放在同一套可靠性框架下后，结论比单纯性能比较更清晰：

1. CNN-LSTM 的优势是真实存在的，但主要是减少 VT/VF cross-errors，而不是全面提升 accuracy、ECE 或总错误。
2. CNN-LSTM 让表征更平滑、更聚类化，但边界可读性没有稳定增强。
3. ProRisk 证明“更规整的表征”本身不是答案；它可以让错误也变得更稳定、更高置信。
4. Hidden strata 显示错误集中在某些 latent record/rhythm clusters，说明可靠性问题具有结构性。
5. Validity map 显示 frozen-feature risk head 是更合理的下一步：显式学习模型有效域，而不是继续把所有正则都塞进主分类器。
6. Model disagreement 提供了一个强的第二意见信号，可以作为 validity head 的输入或对照基线。

## 对下一步模型升级的启发

建议下一步不是再做一个更复杂的 ProRisk，而是做：

**Frozen Backbone + Validity-Domain Approval Head**

核心思想：

- 保留当前最强或最稳定的 backbone/teacher 表征；
- 不再用多种正则共同拉主分类边界；
- 用本轮四方向分析得到的特征训练一个独立 approval/review head；
- 输出不是简单三分类，而是：分类结果 + 是否处于有效域 + 是否需要 review；
- 评估重点放在 VT/VF boundary error capture、high-confidence error suppression、coverage-risk curve，而不是只看 accuracy。

这会把论文从“我们尝试改善模型”提升为：

> ECG models are representation-boundary underspecified. Representation compactness can stabilize both correct and incorrect decisions. Reliable ECG classification therefore requires explicit validity-domain mapping rather than geometric regularization alone.

## 限制

- 这是内部数据上的研究原型结果，不是临床验证。
- 没有患者 ID，因此不能做患者级结论，只能做 record/duplicate-family controlled split 下的 latent strata 分析。
- CNN/CNN-LSTM 与 ProRisk/Teacher 是两组不同实验体系，适合机制对照，不应混成一个最终排行榜。
- Validity classifier 是 post-hoc frozen-feature 检测器，下一步需要独立成正式模型组件并做 paired review-policy evaluation。
