# ECG 不确定性项目组会口播版

这版不是代码索引，是给你开组会时顺着说的讲稿。你可以按这个顺序讲，老师问细节时再打开原来的索引版或公开图集。

## 1. 开场怎么说

老师，我这个项目做的不是一个单纯的 ECG 三分类模型，而是一个可靠性分析原型。模型先判断一个 ECG 窗口属于 SR、VT 还是 VF，但我真正关心的不只是它整体准确率有多高，而是它能不能识别自己什么时候不可靠。

这个任务里最重要的风险点是 VT 和 VF 的边界。SR 和室性节律往往更容易分开，但 VT 和 VF 在模型表示空间里更接近，也更容易互相混淆。所以我后面的分析都围绕一个问题展开：如果只能人工复核一小部分 ECG 窗口，模型能不能把最容易出错、最值得复核的 VT/VF 边界样本排到前面？

我会把整个项目讲成一条证据链：数据防泄漏、模型训练、不确定性评估、embedding 几何分析、VT/VF 边界分析、corruption 稳健性、review routing，最后是 RISK 复核优先级分数。

## 2. 数据部分怎么讲

本地数据来自 `RHYTHMS.mat`，里面有三类 ECG 记录：SR、VT、VF。我做了只读检查，确认本地文件里有 SR 488 条、VT 46 条、VF 60 条原始记录。

这些原始记录长度不一样，所以项目先把 ECG 切成固定长度窗口。默认是 100 Hz 采样率，5 秒一个窗口，也就是 500 个点。每个窗口会做 z-score 标准化，让不同幅值尺度的 ECG 更可比。

这里最重要的是 split。不能把同一条 ECG 记录切出来的窗口随机分到训练集和测试集，因为相邻窗口很像，这会造成数据泄漏。项目用 record-level split，后期还用更严格的 duplicate-family split，把重复或高度相似的 source records 放到同一个 split 里。这样做是为了避免模型看起来很准，但其实只是见过相似片段。

这部分对应的数据代码主要是 `src/data.py` 和 `src/inspect_data.py`。公开统计表是 `results_public/tables/dataset_split_statistics.csv`。

## 3. 训练部分怎么讲

训练入口是 `src/train.py`。它做的事情可以概括成：读数据、切窗口、做 split、选择模型、训练、保存预测概率和 embedding。

模型不是只有一个 CNN，而是比较了多种时序模型，包括 CNN、TCN、ResNet1D、InceptionTime、BiGRU，还有融合 ECG regularity 特征的 RegularityFusion 和 ReliabilityGatedFusion。

这一步的输出很关键。它会保存 `metrics.json`、`test_predictions.csv`、`embeddings_train.npz`、`embeddings_val.npz` 和 `embeddings_test.npz`。后面所有不确定性、PCA、边界分析和复核路由，都要基于这些输出。

公开图 `results_public/figures/00_summary/model_performance_summary.png` 可以用来讲模型分类表现。你可以先说：模型整体分类性能不错，但项目不止停留在 accuracy。因为分类强不代表模型知道自己什么时候不可靠。

## 4. Embedding 和 PCA 怎么讲

Embedding 可以理解成模型内部对 ECG 窗口的压缩理解。PCA 图就是把这个高维空间投影到二维或三维，让我们观察 SR、VT、VF 在模型内部是不是分得开。

这里要注意，PCA 图不是统计证明，它只是诊断图。它的价值在于帮助我们看到结构：SR 和室性节律通常更容易分开，而 VT 和 VF 更容易靠近或混合。

公开图 `results_public/figures/00_summary/embedding_geometry_distances.png` 很适合讲这个点。图里可以看到 VT/VF 的 normalized centroid distance 往往比 SR-VT、SR-VF 小，所以 VT/VF 是最紧的边界。

这就自然引出下一步：我不能只看 overall error，而要把 VT->VF 和 VF->VT 这种 cross-error 单独分析。

## 5. 不确定性怎么讲

不确定性部分问的是：模型错的时候，它有没有表现出“不确定”？

项目比较了几类分数。第一类是 softmax 决策不确定性，比如 MSP、entropy、temperature scaling、energy。第二类是 embedding 空间的 atypicality，比如 prototype distance、Mahalanobis distance、kNN distance。

公开图 `results_public/figures/00_summary/uncertainty_error_detection.png` 可以说明：有些分数对普通错误检测很强，比如 MSP 和 entropy；但有些分数并不稳定，比如 energy 在一些模型上很弱。这个负结果很重要，因为它说明可靠性不能只靠一个分数。

这部分对应代码是 `src/uncertainty.py` 和 `src/evaluate_uncertainty.py`。

## 6. VT/VF 边界怎么讲

这个项目不是把所有错误混在一起看，而是专门分析 VT/VF 边界错误。因为在这个任务里，VT 和 VF 互相混淆比普通错误更值得关注。

`src/ambiguity_analysis.py` 里计算了 ventricular ambiguity index。它综合了几个信息：到 VT 和 VF 原型的距离是否接近，softmax 里 VT 和 VF 的概率是否接近，近邻标签是否混乱，以及近邻里 VT/VF 是否混合。

这部分可以这样说：我希望模型不只是输出一个类别，还能告诉我们这个窗口是不是处在 VT/VF 模糊边界上。如果它处在边界上，就更应该进入专家复核。

对应图组是 `results_public/figures/02_uncertainty_calibration/`。里面的 ambiguity distribution、atypicality-vs-ambiguity map 和 review curve 都在支持这个逻辑。

## 7. ECG regularity 怎么讲

为了不让整个项目只是黑盒深度学习，我还引入了 ECG-domain regularity 特征。比如频谱熵、主频、主频集中度、自相关峰、过零率、line length 等。

这些特征的意义是把模型可靠性和 ECG 信号结构联系起来。比如 VT 和 VF 的节律和频域结构可能不同，边界错误可能和 regularity 异常有关。

图组 `results_public/figures/03_regularity_interpretability/` 主要就是讲这件事。你不用逐张背，只要说它们比较了不同 case group 的 autocorr、dominant frequency concentration、sample entropy 和 spectral entropy。最后的 permutation importance 图说明某些 regularity 特征确实对分类或可靠性有贡献。

对应代码是 `src/regularity_analysis.py`、`src/feature_only_analysis.py`、`src/regularity_feature_ablation.py` 和 `src/gate_analysis.py`。

## 8. OOD 和 corruption 怎么讲

干净测试集表现好还不够。ECG 信号可能有噪声、基线漂移、遮挡、尖峰、饱和、时间缩放等问题，所以项目做了 corruption severity 测试。

图组 `results_public/figures/04_ood_corruption/` 每张图对应一种 corruption，横轴是严重程度，纵轴是 OOD AUROC。这个图组想回答的是：当信号越来越坏时，不确定性或 embedding 分数能不能反应出来？

这里不要说某一个分数永远最好。更稳妥的说法是：不同分数对不同 corruption 敏感性不同，所以需要多种可靠性视角。

对应代码是 `src/evaluate_ood.py`、`src/evaluate_corruption_severity.py` 和 `src/monotonicity_analysis.py`。

## 9. Review routing 是中心

这是项目最重要的一步。因为不确定性分数本身只是一个数字，真正有意义的是它能不能改变决策。

项目把测试窗口按风险分数排序。高风险窗口交给专家复核，低风险窗口自动接受模型预测。然后统计在 10%、20%、30% 复核预算下，能抓住多少错误，尤其是多少 VT/VF 边界错误。

公开图 `results_public/figures/00_summary/review_routing_vtvf_capture.png` 是这部分最适合放 PPT 的图。公开表 `results_public/tables/review_routing_boundary_lrii.csv` 里可以看到：CNN-10 在 20% 复核预算下抓住约 91.7% 的 VT/VF 边界错误，TCN-20 抓住约 93.7%，RegularityFusion-12 抓住约 96.5%。

还有一个很关键的负结果：ResNet1D-12 分类并不弱，但 10% 复核预算下只抓住约 38.3% 的 VT/VF 边界错误。这说明强分类器不等于强复核排序器，所以 review routing 必须单独评估。

对应代码是 `src/review_efficiency_analysis.py`、`src/reliability_map.py`、`src/ambiguity_routing_policy.py` 和 `src/runtime_supervisor.py`。

## 10. PRO 怎么讲

PRO 是 prototype separation，可以理解成一种表示空间边界干预。它希望让 embedding 里的类别结构更清楚，尤其让 VT/VF 边界更可控。

早期 V4/V5 图里，PRO 确实能改变 embedding geometry，也可能减少某些 automatic-route VT/VF errors。所以它有研究价值。

但最终 V6 duplicate-family 证据更严格，PRO 不能作为稳定提升方法来讲。它在不同 seed 上表现不一致，而且会出现 error migration，也就是某些错误方向减少了，另一些方向可能增加。

所以老师如果问 PRO 是否成功，你可以回答：PRO 不是最终 deployable solution，但它是一个重要的 boundary-structure 实验。它帮助我们理解表示空间干预会怎样改变错误分布，也提醒我们不能只看平均提升。

对应图组是 `results_public/figures/06_pro_geometry/`、`09_pro_boundary_mitigation/` 和最终更关键的 `10_v6_pro_error_migration/`。

## 11. RISK 怎么讲

RISK 是当前最适合讲的主贡献。

它的想法是：部署时不可能每次都跑一堆复杂 post-hoc 分析，所以我把多种可靠性证据压缩成一个 risk score。这个分数不是为了提高分类 accuracy，而是为了给专家复核排序。

RISK 的证据来源包括 entropy、kNN atypicality、local instability、VT/VF mixing、softmax VT/VF ambiguity 等。`src/select_deployable_risk_weights.py` 会在 validation set 上选择最符合复核目标的权重，避免手工平均把关键 VT/VF 信号稀释掉。

最终 V6 duplicate-family 结果里，validation-selected RISK 在 10% review burden 下抓住约 82.8% 的 VT/VF cross-errors，在 20% review burden 下抓住 100%。这是当前项目最稳妥的结论。

但一定要补一句：这是内部数据、三种子、窗口级实验结果，不是外部临床验证。

对应图组是 `results_public/figures/11_v6_risk_distillation/`，对应表是 `results_public/tables/duplicate_family_selected_risk_review_aggregate.csv`。

## 12. 图片怎么顺着讲

不要一张张机械讲。你可以按图组讲：

`00_summary` 是总览，四张图串起分类、embedding、不确定性和 review routing。

`01_embedding_pca` 讲模型内部表示，重点是 VT/VF 为什么是最紧边界。

`02_uncertainty_calibration` 讲 confidence、entropy、ambiguity、coverage-risk 和 review curve。

`03_regularity_interpretability` 讲 ECG regularity 特征如何解释难样本。

`04_ood_corruption` 讲信号变坏时不确定性是否反应。

`05_risk_supervisor_ablation` 讲 risk head、supervisor、ablation 和 stability-aware routing。

`06_pro_geometry`、`09_pro_boundary_mitigation` 和 `10_v6_pro_error_migration` 要一起讲：PRO 有边界结构价值，但最终要谨慎。

`08_risk_corruption_robustness` 和 `11_v6_risk_distillation` 支持 RISK 作为最终 review-priority 方向。

## 13. 老师可能问什么

如果老师问：为什么不只看 accuracy？

你答：因为总体 accuracy 会掩盖 VT/VF 这种高风险边界错误。我的目标是让模型知道什么时候不该自动接受预测。

如果老师问：PCA 能证明什么？

你答：PCA 不证明统计显著性，它是诊断图，用来观察 embedding 结构，尤其是 VT/VF 是否混合。

如果老师问：RISK 和普通 uncertainty 有什么区别？

你答：普通 uncertainty 是单个分数，RISK 是把多源可靠性证据蒸馏成一个直接面向复核排序的分数。

如果老师问：PRO 是不是失败了？

你答：不是简单失败。它不是最终稳定方法，但它揭示了边界干预和 error migration，所以是有研究价值的混合结果。

如果老师问：最大限制是什么？

你答：数据是内部受限数据，没有外部验证；最终 evidence 只有三个 seed；corruption 是合成扰动；窗口级分类不等于患者级诊断；项目不是医疗器械。

## 14. 最后一页怎么收

最后可以这样说：

这个项目的贡献不是一个最强 ECG 分类器，而是一套可靠性分析流程。它先控制 record-level 和 duplicate-family leakage，再训练多种模型，然后用不确定性、embedding、regularity、corruption 和 review routing 来判断模型什么时候不可靠。最终，RISK 把这些可靠性证据蒸馏成一个复核优先级分数，在内部三种子 duplicate-family 结果中能用有限复核预算捕获大部分 VT/VF cross-errors。

保守结论是：RISK 可以作为内部研究场景下的 expert-review priority score，但还需要外部数据、更多 seed 和更明确的临床复核流程，才能讨论更强的泛化和部署。
