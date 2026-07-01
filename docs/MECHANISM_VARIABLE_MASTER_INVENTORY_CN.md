# ECG 可靠性机制变量总表

日期：2026-06-30

## 1. 这份表的作用

本文件整理项目中已经分析过、使用过或证据化过的机制变量。目的不是声称每一个变量都已经完成同等级因果验证，而是把后续“机制靶向因果消融实验”需要考虑的变量完整列出来。

当前变量可以分为七大类：

1. 表征几何与中心距离；
2. KNN 邻域结构；
3. prototype / centroid 边界歧义；
4. softmax / confidence / calibration；
5. validity / gate / boundary 可靠性信号；
6. ECG 波形规则性与小波时频结构；
7. 路由、解释、风险头和 review policy。

## 2. 表征几何与中心距离

这部分回答的问题是：模型的 embedding 空间是否把 SR、VT、VF 分开了，尤其 VT/VF 边界是否清楚。

| 变量 | 含义 | 已有证据状态 | 下一步可干预方式 |
| --- | --- | --- | --- |
| `silhouette_full` | 全 embedding 的类别分离度 | 已进入模型层 paired 机制验证 | contrastive / center loss |
| `davies_bouldin_full` | 类簇紧密度与分离度综合指标 | 已进入模型层 paired 机制验证 | contrastive / center loss |
| `sr_vt_norm_dist` | SR 与 VT 中心距离，按类内散度归一化 | 已进入模型层 paired 机制验证 | SR/VT margin 或 class-aware contrastive |
| `sr_vf_norm_dist` | SR 与 VF 中心距离 | 已进入模型层 paired 机制验证 | SR/VF margin 或 class-aware contrastive |
| `vt_vf_norm_dist` | VT 与 VF 中心距离 | 已进入模型层 paired 机制验证 | VT/VF prototype margin |
| `euclidean_centroid_distance` | 欧氏中心距离 | 早期高级表征诊断中使用 | centroid-margin intervention |
| `mahalanobis_centroid_distance` | 考虑协方差后的中心距离 | 早期高级表征诊断中使用 | covariance-aware margin 或 diagnostic-only |
| `within_scatter` | 类内分散程度 | PRO/geometry 分析中使用 | center compactness loss |
| `centroid_distance_margin` | 最近中心与第二近中心的距离差 | 决策边界分析中使用 | margin loss |
| `nearest_centroid_distance` | 到最近类别中心的距离 | 决策边界/风险分析中使用 | distance-aware risk target |

这部分最接近你说的 central distance / center distance。注意：这些变量本身通常不是直接可调旋钮，而是通过 center loss、prototype margin、contrastive loss 等训练约束间接干预。

## 3. KNN 邻域结构

这部分回答的问题是：一个样本在 embedding 空间附近的邻居是否支持当前类别，还是处在混杂区域。

| 变量 | 含义 | 已有证据状态 | 下一步可干预方式 |
| --- | --- | --- | --- |
| `local_purity_k_mean` | KNN 邻域纯净度 | 已进入模型层 paired 机制验证 | neighborhood-purity proxy loss |
| `knn_distance_mean` / `knn_mean_distance` | 到邻居的平均距离 | 已进入模型层 paired 机制验证 / 路由层使用 | local compactness loss |
| `knn_min_distance` | 到最近邻居距离 | 路由特征中使用 | diagnostic 或 local compactness |
| `knn_label_entropy_mean` | 邻居标签熵 | 已进入模型层 paired 机制验证 | supervised contrastive / purity loss |
| `knn_label_entropy_any_error_auroc` | KNN 标签熵识别错误的 AUROC | 已进入模型层 paired 机制验证 | risk target / review score |
| `knn_vtvf_mix_ventricular_mean` | VT/VF 邻域混合程度 | 已进入模型层 paired 机制验证 | VT/VF separation loss |
| `knn_vtvf_mix_auroc` | KNN VT/VF mixing 对边界错误的识别能力 | 已进入模型层 paired 机制验证 | risk target / routing score |
| `knn_vt_fraction`, `knn_vf_fraction`, `knn_sr_fraction` | 邻域中各类别比例 | 路由特征中使用 | diagnostic / routing evidence |
| `knn_pred_is_model_pred` | KNN 预测是否和模型预测一致 | 路由冲突机制中使用 | representation-conflict target |
| `neighbor_jaccard_mean` | 扰动前后邻域稳定性 | 高级表征诊断中使用 | stability / embedding consistency |

这部分就是你说的“邻域纯净度、KNN 密集度、混合程度”。它很适合下一步做机制靶向消融，因为 KNN purity / entropy / mixing 可以分别对应不同训练约束。

## 4. Prototype / centroid 边界歧义

这部分回答的问题是：样本离 VT 原型和 VF 原型是否太接近，是否存在 prototype 层面的边界混淆。

| 变量 | 含义 | 已有证据状态 | 下一步可干预方式 |
| --- | --- | --- | --- |
| `prototype_vtvf_ambiguity_ventricular_mean` | VT/VF 样本的原型歧义均值 | 已进入模型层 paired 机制验证 | VT/VF prototype margin |
| `prototype_vtvf_ambiguity_auroc` | 原型歧义识别 VT/VF 错误的 AUROC | 已进入模型层 paired 机制验证 | risk target / routing score |
| `proto_vtvf_ambiguity` | 路由层原型歧义特征 | 路由和 calibration 分析中使用 | fixed-router evidence |
| `proto_margin` | 最近和次近原型距离差 | 路由特征中使用 | prototype margin loss |
| `abs_proto_vtvf_margin` | VT/VF 原型 margin 绝对值 | calibration/decision 分析中使用 | VT/VF margin intervention |
| `proto_vtvf_prefers_vf` | 原型距离是否更偏向 VF | 路由特征中使用 | bias / directional error audit |
| `nearest_proto_is_pred` | 最近原型是否等于模型预测 | 表征冲突分析中使用 | representation-conflict target |
| `prototype_flip_rate` | 扰动后最近原型是否改变 | 高级表征诊断中使用 | stability / prototype consistency |

这部分与 PRO/prototype separation 直接相关。当前最成功的 `boundary075_prototype` 就主要利用了 boundary weighting 与 prototype geometry 的组合。

## 5. Softmax / confidence / calibration

这部分回答的问题是：模型的置信度、概率边界和校准是否能识别错误，尤其是否存在高置信错误。

| 变量 | 含义 | 已有证据状态 | 下一步可干预方式 |
| --- | --- | --- | --- |
| `confidence_mean` | 平均最大 softmax 概率 | 已进入模型层 paired 机制验证 | anti-confident risk loss |
| `entropy_mean` | 平均预测熵 | 已进入模型层 paired 机制验证 | risk entropy alignment |
| `entropy_any_error_auroc` | 熵识别错误的能力 | 已进入模型层 paired 机制验证 | entropy-risk target |
| `prob_margin_mean` | 第一和第二类别概率差 | 已进入模型层 paired 机制验证 | margin-aware calibration |
| `low_margin_any_error_auroc` | 低 margin 识别错误能力 | 已进入模型层 paired 机制验证 | margin loss / review score |
| `softmax_vtvf_ambiguity_ventricular_mean` | VT/VF 概率边界歧义 | 已进入模型层 paired 机制验证 | boundary CE / entropy alignment |
| `softmax_vtvf_ambiguity_auroc` | softmax VT/VF 歧义识别边界错误能力 | 已进入模型层 paired 机制验证 | routing score / risk target |
| `energy`, `msp`, `temperature_msp`, `temperature_entropy` | 传统 uncertainty scores | uncertainty error detection 表中使用 | calibration baseline |
| `ECE` | expected calibration error | 模型 outcome | calibration objective |

这部分要谨慎：置信度变量常常和错误相关，但不一定天然可靠，所以需要结合 calibration、anti-confident loss 和 OOD stress。

## 6. Validity / gate / boundary 可靠性信号

这部分回答的问题是：模型是否知道自己处在不可靠或边界区域。

| 变量 | 含义 | 已有证据状态 | 下一步可干预方式 |
| --- | --- | --- | --- |
| `validity_gate_mean` | validity gate 平均强度 | 已进入模型层 paired 机制验证 | gate target / risk gate loss |
| `boundary_score_mean` | boundary head 平均分数 | 已进入模型层 paired 机制验证 | boundary auxiliary loss |
| `gate_x_boundary_mean` | gate 与 boundary score 的组合 | 已进入模型层 paired 机制验证 | joint gate-boundary objective |
| `boundary_score_any_error_auroc` | boundary score 识别任意错误能力 | 已进入模型层 paired 机制验证 | boundary risk alignment |
| `validity_gate_any_error_auroc` | gate 识别错误能力 | 已进入模型层 paired 机制验证 | risk gate alignment |
| `gate_x_boundary_any_error_auroc` | gate x boundary 识别任意错误能力 | 已进入模型层 paired 机制验证 | joint alignment |
| `boundary_score_vtvf_cross_auroc` | boundary score 识别 VT/VF 交叉错误 | 已进入模型层 paired 机制验证 | VT/VF boundary target |
| `gate_x_boundary_vtvf_cross_auroc` | gate x boundary 识别 VT/VF 错误 | 已进入模型层 paired 机制验证 | joint boundary target |
| `gate_minus_confidence`, `low_validity_model_confidence` | gate 与低置信度组合 | validity audit 中使用 | diagnostic / routing score |

这里不能写成“gate 天然正确”。早期 gate-target 版本表现不稳定，所以它应写成可干预 reliability component，而不是已经自证正确的解释机制。

## 7. ECG 规则性波形变量

这部分回答的问题是：ECG 节律和频谱结构是否能解释 VT/VF 边界、atypical signal 或模型不稳定。

基础 regularity features：

| 变量 | 含义 | 机制类型 |
| --- | --- | --- |
| `spectral_entropy` | 频谱复杂度/混乱程度 | frequency / complexity |
| `dominant_frequency` | 主导频率 | frequency |
| `dominant_frequency_concentration` | 主频集中度 | frequency / periodicity |
| `spectral_centroid` | 频谱重心 | frequency |
| `spectral_bandwidth` | 频谱带宽 | frequency / complexity |
| `autocorr_peak` | 自相关峰值，反映周期性 | periodicity |
| `autocorr_peak_lag_s` | 自相关峰滞后时间 | periodicity |
| `zero_crossing_rate` | 零交叉率，反映快速振荡/复杂度 | complexity |
| `line_length` | 波形线长，反映形态复杂度 | complexity |

机制库中已经按 feature family 做过证据化：

| family | 例子 |
| --- | --- |
| `frequency_only` | dominant frequency, spectral centroid, spectral bandwidth 等 |
| `periodicity_only` | autocorr peak, autocorr lag |
| `complexity_only` | zero crossing, line length, entropy |
| `without_frequency / without_periodicity / without_complexity` | 消融某一类 regularity 特征 |
| `regularity_feature_injection` | 将规则性特征注入模型或路由 |

regularity 很重要，但当前更适合写成辅助 ECG waveform evidence，而不是所有模型改善的唯一主因。

## 8. 小波 / 多尺度时频变量

这部分回答的问题是：VT/VF 边界是否需要从多尺度时频结构中观察。

小波特征由 3 个尺度、3 类原子和多个统计量组成：

| 维度 | 内容 |
| --- | --- |
| scale | `s2`, `s4`, `s8` |
| atom | `mexican_hat`, `slope`, `oscillation` |
| statistic | `mean`, `std`, `p95`, `max`, `entropy` |
| derived ratios | `osc_to_shape_energy`, `slope_to_shape_energy`, `fine_to_coarse_energy`, `mid_to_coarse_energy`, `oscillation_fraction`, `slope_fraction`, `shape_fraction` |

机制库中最重要的两个小波 risk heads：

| 变量 | target | 已有证据 |
| --- | --- | --- |
| `wavelet_any_error_risk` | any error | 10-seed AUROC/AUPR + budget routing |
| `wavelet_vtvf_boundary_risk` | VT/VF cross-error | 10-seed AUROC/AUPR + budget routing |

小波证据很强，但主要属于 evidence head / routing evidence。它不能单独拿来和完整 V5D router 比，必须组成完整 policy 后比较。

## 9. Risk target / local instability / LRI 变量

这部分回答的问题是：哪些样本应被判为高风险，进入 review 或 recover。

| 变量 | 含义 | 用途 |
| --- | --- | --- |
| `risk_target_entropy` / `entropy` | 熵风险 | risk target / review score |
| `risk_target_knn` / `knn` | KNN 距离风险 | risk target |
| `risk_target_prototype` / `prototype` | prototype distance risk | risk target |
| `local_instability` | 局部扰动或邻域不稳定性 | risk target / review score |
| `softmax_vtvf_ambiguity` | VT/VF softmax boundary 风险 | risk target / routing |
| `lrii` | local rhythm instability index | review / routing |
| `boundary_lrii` | 边界导向 LRI | VT/VF boundary review |
| `atypicality_lrii` | atypical signal 导向 LRI | atypical review |
| `vtvf_mixing` | VT/VF 邻域混合风险 | review / routing |

这些变量是从诊断变量转向 review/recover policy 的关键桥梁。

## 10. 机制风险头与解释变量

这部分回答的问题是：错误是否可以被分成不同机制类型，并给出对应解释。

机制风险头：

| 机制头 | 含义 | 当前状态 |
| --- | --- | --- |
| `vtvf_boundary` | VT/VF 边界错误 | 强证据 |
| `representation_conflict` | 表征冲突错误 | 强证据 |
| `sr_ventricular` | SR 与 ventricular 混淆 | 中等/可用证据 |
| `atypical_signal` | 非典型波形错误 | 强机制头，但解释需谨慎 |
| `hidden_confident` | 隐藏高置信错误 | 当前为负结果，不能写成主机制 |

解释变量：

| explanation | 对应机制 |
| --- | --- |
| `boundary_explanation` | VT/VF boundary |
| `representation_explanation` | representation conflict |
| `sr_ventricular_explanation` | SR-ventricular |
| `regularity_atypicality_explanation` | atypical signal |
| `hidden_confidence_explanation` | hidden confident |
| `second_opinion_explanation` | model disagreement / second opinion |

解释变量更适合作为 explanation reliability，不应替代 outcome 验证。

## 11. Review / routing outcome 变量

这部分回答的问题是：机制证据用于兜底以后，是否减少自动残余错误或提高 review capture。

| outcome | 含义 |
| --- | --- |
| `all_error_addressed` / `all_error_captured` | 所有错误被 review/recover 覆盖比例 |
| `vtvf_cross_error_addressed` / `vtvf_error_captured` | VT/VF 交叉错误捕获比例 |
| `automatic_unresolved_error_rate` / `auto_error_rate` | 未进入 review 的自动错误率 |
| `automatic_unresolved_vtvf_cross_error_rate` / `auto_vtvf_error_rate` | 未进入 review 的 VT/VF 自动错误率 |
| `review_error_enrichment` | review 队列中错误富集程度 |
| `review_burden` / `budget` | review 成本 |

这些变量是路由层多目标优化的 outcome，不应和模型层 accuracy/macro-F1 混在一起比较。

## 12. 下一步机制靶向因果消融建议

建议不要一次验证所有变量，而是先选 5 个核心机制做 targeted intervention：

| 优先级 | 目标机制 | 干预方式 | 观察变量 | outcome |
| --- | --- | --- | --- | --- |
| 1 | prototype VT/VF ambiguity | prototype margin / center loss | `prototype_vtvf_ambiguity`, `vt_vf_norm_dist` | VT/VF cross-errors, total errors |
| 2 | KNN purity / mixing | supervised contrastive 或 neighborhood proxy | `local_purity_k_mean`, `knn_label_entropy`, `knn_vtvf_mix` | macro-F1, VT/VF errors |
| 3 | softmax boundary ambiguity | boundary CE / entropy alignment | `softmax_vtvf_ambiguity`, `prob_margin`, ECE | calibration, VT/VF errors |
| 4 | validity gate alignment | risk-gate / boundary-gate alignment | `gate_x_boundary_auroc`, `validity_gate_auroc` | error capture, ECE |
| 5 | ECG waveform regularity | regularity aux / feature-family ablation | frequency/periodicity/complexity features | atypical error, SR/VT/VF confusion |

小波和 routing evidence 建议放在第二轮，因为它们更偏向兜底路由层，不是模型 embedding 本体的第一批机制靶向训练干预。

