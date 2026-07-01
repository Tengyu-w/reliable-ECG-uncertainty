# 因果机制感知的多目标优化方法章节草稿

> 本文档用于论文方法章节，不作为临床诊断或医疗器械验证声明。这里的“因果”指项目内部的、同数据划分和同随机种子下的干预式证据链，用于解释训练约束、机制信号和路由策略为什么会影响可靠性结果。

## 1. 研究问题

本研究的核心任务不是单纯提高 SR、VT、VF 三分类准确率，而是在心电图分类模型发生不确定、边界混淆或隐藏性高置信错误时，建立一个可解释、可量化、可复现实验验证的可靠性机制。前期工作已经从三个层面发现了错误来源：

1. 表征层问题：不同类别在 embedding 空间中存在局部混叠、邻域纯度不足、原型距离不稳定和边界样本聚集等现象。
2. 心电波形结构问题：VT/VF 边界错误与节律规则性、频谱集中度、小波时频特征、周期性和线长等波形属性有关。
3. 模型可靠性问题：部分模型即使分类准确率较高，仍可能在 VT/VF 边界、隐藏高置信错误、校准误差和错误迁移上表现不稳定。

因此，本研究将“模型性能”扩展为多目标可靠性问题：不仅优化分类正确性，还同时约束校准、边界错误、总错误、错误迁移和路由可解释性。

## 2. 变量定义

为了避免把所有分析结果简单堆叠为经验规则，本研究将变量分为四类：不可干预变量、可干预变量、机制变量和结果变量。

### 2.1 不可干预变量

不可干预变量是研究对象本身或实验设计中固定不变的背景条件：

| 类型 | 示例 | 说明 |
| --- | --- | --- |
| ECG 波形事实 | 原始 SR、VT、VF 片段及其节律结构 | 不能为了优化模型而改变真实波形结构 |
| 标签与任务定义 | SR、VT、VF 分类标签 | 本研究不重新定义疾病标签 |
| 数据划分 | record-level split、固定 seed 下的 train/validation/test | 用于降低泄漏风险和保证 paired comparison |
| 历史模型输出 | 已训练模型的 logits、embedding、预测错误 | 在机制库分析中作为观测证据使用 |

### 2.2 可干预变量

可干预变量是本研究可以主动改变并观察其后果的设计选择：

| 层面 | 可干预变量 | 对应问题 |
| --- | --- | --- |
| 模型训练层 | prototype loss、boundary loss、risk loss、regularity constraint、stability constraint、embedding consistency 等 | 改变训练约束是否会改变表征结构和模型可靠性 |
| 机制证据层 | 使用或不使用 wavelet、regularity、KNN density、prototype ambiguity、softmax ambiguity、mechanism head、explanation score | 哪些机制信号真正能解释错误 |
| 路由策略层 | evidence weight、budget、risk score 组合、stage1/stage2 阈值和权重 | 如何在固定 review budget 下捕获更多关键错误 |
| 多目标选择层 | accuracy、macro-F1、ECE、VT/VF cross-error、total error、migration penalty 的权衡方式 | 哪个候选方案位于更合理的 Pareto 区域 |

### 2.3 机制变量

机制变量是连接“干预”和“结果”的中间证据。它们回答的问题是：为什么某个训练约束或路由策略会影响最终结果。

| 机制家族 | 变量示例 | 解释作用 |
| --- | --- | --- |
| 表征几何 | silhouette、类内距离、类间距离、原型距离、embedding dispersion | 判断类别是否在表征空间中更可分 |
| KNN 邻域结构 | local purity、boundary density、neighbor disagreement | 判断局部邻域是否支持当前预测 |
| 原型与 softmax 歧义 | prototype VT/VF ambiguity、softmax VT/VF ambiguity、margin | 判断 VT/VF 是否处于边界混淆 |
| validity 与 risk | gate x boundary AUROC、risk target、hidden confident risk | 判断模型是否知道自己何时不可靠 |
| 小波时频信号 | wavelet vtvf boundary risk、frequency band energy、spectral concentration | 判断波形层面是否存在边界风险 |
| 规则性波形信号 | periodicity、dominant frequency、spectral entropy、line length、autocorr peak | 判断 ECG 节律结构是否支持错误解释 |
| 解释一致性 | boundary explanation、representation explanation、hidden confidence explanation | 判断机制头给出的解释是否对齐真实错误类型 |

这些变量不能被省略，因为它们正是本项目区别于普通黑箱模型调参的核心证据来源。

### 2.4 结果变量

结果变量必须按层面分别定义，不能混用。

模型层结果变量包括：

| Outcome | 含义 |
| --- | --- |
| accuracy | 三分类总体准确率 |
| macro-F1 | 类别均衡后的分类性能 |
| ECE | calibration error，越低越好 |
| VT/VF cross-error | VT 与 VF 之间互相混淆的错误数量 |
| total error | 总错误数量 |
| error migration penalty | 一个机制减少某类错误时是否引入其他错误 |

路由层结果变量包括：

| Outcome | 含义 |
| --- | --- |
| all-error addressed | review 或 recover 策略覆盖全部错误的比例 |
| VT/VF capture | review 或 recover 策略捕获 VT/VF 边界错误的比例 |
| unresolved VT/VF risk | 路由后仍未覆盖的 VT/VF 风险 |
| budget efficiency | 在固定 10%、20%、30% budget 下的错误捕获效率 |
| explanation reliability | 路由原因是否与真实错误机制一致 |

## 3. 因果机制证据链

本研究不将因果推断写成“ECG 生理因果证明”，而是写成内部干预式机制证据链。核心逻辑是：

```text
do(可干预设计选择) -> 机制变量变化 -> 可靠性结果变化
```

### 3.1 模型层证据链

模型层关注训练约束是否通过表征机制改善模型本身：

```text
do(training constraint)
    -> embedding / KNN / prototype / softmax / validity mechanism
    -> accuracy, macro-F1, ECE, VT/VF error, total error, migration
```

例如，`boundary075_prototype` 不是只因为最终指标变好才被保留，而是因为同 seed paired comparison 显示它同时改变了多个机制变量：embedding silhouette 增加、KNN local purity 增加、prototype VT/VF ambiguity 下降、softmax VT/VF ambiguity 下降，并且 validity signal 对错误迁移的解释能力提高。也就是说，它存在“干预 -> 机制 -> outcome”的证据链。

### 3.2 机制库与路由层证据链

路由层关注机制信号是否能组成完整 routing policy：

```text
do(evidence family / policy weight / budget)
    -> wavelet / regularity / mechanism head / explanation signal
    -> error capture, unresolved risk, budget efficiency, explanation reliability
```

这里必须区分三件事：

1. evidence head：某个机制信号能不能识别某类错误。
2. routing policy：多个机制信号如何在预算和权重下组合成完整路由。
3. recovery action：路由之后选择 review、recover 或其他处理方式。

因此，不能用单个 wavelet head 或 validity head 直接和 V5D 这种完整路由机制比较。公平比较必须是完整 routing policy 对完整 routing policy。

## 4. 多目标优化定义

### 4.1 模型层目标向量

模型层候选方案的目标向量定义为：

```text
Y_model = [
    maximize accuracy,
    maximize macro-F1,
    minimize ECE,
    minimize VT/VF cross-error,
    minimize total error,
    minimize error migration penalty
]
```

一个候选模型只有在不牺牲关键可靠性指标的前提下提升分类性能，才可以被认为优于基线。若某方案提高 accuracy 但显著恶化 VT/VF cross-error 或 calibration，则不能简单称为总体改进。

### 4.2 路由层目标向量

路由层候选策略的目标向量定义为：

```text
Y_route = [
    maximize all-error addressed,
    maximize VT/VF capture,
    minimize unresolved VT/VF risk,
    minimize review budget,
    maximize explanation reliability
]
```

这意味着路由机制的目标不是重新训练一个分类器，而是在模型已经产生预测、风险和机制证据之后，决定哪些样本应该进入 review 或 recover 流程，以及为什么。

### 4.3 Pareto 选择原则

候选方案 A 相对于候选方案 B 的优势，不应只由单一指标决定，而应由 Pareto 支配关系判断：

```text
A 在所有关键目标上不差于 B，
并且至少在一个关键目标上优于 B，
则 A Pareto-dominates B。
```

对于不能完全支配的候选方案，本研究保留 trade-off 解释。例如，较高 review budget 可能带来更高 VT/VF capture，但牺牲人工审核效率；较强 prototype constraint 可能改善边界错误，但需要检查是否带来错误迁移。

## 5. 定量证据构建方法

### 5.1 同 seed paired intervention

模型层使用同 seed paired comparison，而不是把不同随机种子、不同模型和不同实验随意横向比较。对于每个候选干预，计算：

```text
Delta mechanism = M_intervention(seed_i) - M_baseline(seed_i)
Delta outcome   = Y_intervention(seed_i) - Y_baseline(seed_i)
```

这样可以回答：在同样数据划分和随机条件下，改变训练约束是否稳定改变了机制变量和结果变量。

### 5.2 机制变量到结果变量的关联

在 paired delta 之外，本研究进一步计算机制变量和结果变量之间的统计关联，例如 Spearman correlation。其作用不是证明严格随机因果效应，而是检查机制解释是否与结果变化方向一致。

如果一个机制变量被声称为重要原因，它至少应满足：

1. 干预后该机制变量发生可观察变化。
2. 该机制变量变化方向与 outcome 改善方向一致。
3. 该机制变量在不同 seed 或不同候选中不是只出现一次的偶然现象。
4. 该机制变量能解释具体错误类型，而不是只提升总分。

### 5.3 机制库证据强度

对于 wavelet、regularity、mechanism head 和 explanation signal，定量证据主要来自：

| 证据类型 | 作用 |
| --- | --- |
| AUROC / AUPRC | 判断机制信号是否能识别特定错误类型 |
| mean ± std across seeds | 判断机制是否稳定 |
| fixed-budget capture | 判断路由在固定审核预算下的错误捕获效率 |
| unresolved risk | 判断高风险错误是否仍被漏掉 |
| explanation alignment | 判断给出的原因是否对应真实错误类别 |

例如，wavelet VT/VF boundary risk 的作用不是“证明小波一定最优”，而是证明 ECG 时频结构能够为 VT/VF 边界风险提供稳定证据。hidden confident head 的表现若接近随机，则应作为负结果保留，不能强行写成有效机制。

## 6. 与既有模型和路由的公平比较

本研究采用分层比较原则：

| 比较对象 | 应比较对象 | 不应比较对象 |
| --- | --- | --- |
| 分类模型 | CNN、CNN-LSTM、PRO、constrained model、causal-Pareto model | V5D router |
| evidence head | 其他单一 evidence head | 完整 V5D router |
| routing policy | V5D stage1/stage2 router、causal-Pareto weighted router、fixed budget router | 单个分类模型 |
| recovery action | 其他 recovery action 或完整 recover pipeline | 单个 risk score |

因此，模型层的结论只能回答“哪种模型或训练约束更好”；路由层的结论只能回答“哪种完整路由策略在相同预算下更好”。这一步是为了避免前期错误版本中出现的层级混淆。

## 7. 当前证据支持的结论写法

### 7.1 模型层

当前证据支持的写法是：

`boundary075_prototype` 在同 seed paired comparison 中同时改善了分类、校准、VT/VF 边界错误和错误迁移，并且这些改善伴随 embedding silhouette、KNN local purity、prototype ambiguity、softmax ambiguity 和 validity signal 的方向性改善。因此，它可以作为模型层“训练约束 -> 表征机制 -> 可靠性 outcome”的主要候选证据。

边界说明：该结论仅支持模型可靠性机制解释，不扩展为 ECG 生理机制证明或临床验证声明。

### 7.2 路由层

当前证据支持的写法是：

wavelet、regularity、mechanism-specific heads 和 explanation scores 可以组成机制证据库，其中部分机制对 VT/VF boundary、representation conflict 和 atypical signal 等错误类型具有较强识别能力。完整 routing policy 需要在固定 budget 下比较 error capture、VT/VF capture 和 unresolved risk，而不是把单个 evidence head 直接拿来和 V5D 比。

边界说明：路由层比较单位是完整 routing policy，单个机制信号只作为 routing policy 的组成证据。

## 8. 可复现文件与实验入口

本方法章节对应的主要实验和结果文件如下：

| 目的 | 脚本或结果 |
| --- | --- |
| 模型层机制量化 | `src/run_causal_mechanism_quantification.py` |
| 模型层结果目录 | `results/causal_mechanism_quantification_20260630/` |
| 机制变量字典 | `causal_mechanism_variable_dictionary.csv` |
| paired delta 表 | `paired_candidate_seed_mechanism_outcome_deltas.csv` |
| 机制到 outcome 关联 | `mechanism_to_outcome_association.csv` |
| 机制库证据链 | `src/build_mechanism_library_evidence_chain.py` |
| 机制库结果目录 | `results/mechanism_library_evidence_chain_20260630/` |
| 机制信号强度表 | `mechanism_signal_strength_inventory.csv` |
| 路由 outcome 表 | `mechanism_policy_outcome_inventory.csv` |

这些文件共同支持论文中的方法逻辑：不是先验指定某个机制一定有效，而是把每个机制都放到定量证据链中检验。

## 9. 论文中的限制说明

本研究仍有以下限制，论文中需要主动说明：

1. 当前因果证据是内部 paired intervention 和机制关联证据，不是外部随机对照实验。
2. 模型层关键结果目前主要来自有限 seed，适合写作“preliminary but structured evidence”，不能夸大为最终定论。
3. 项目缺少外部独立 ECG 数据集，因此泛化能力只能通过 record-level split、OOD-style stress 和机制稳定性间接支持。
4. ECG 波形机制解释仍是模型可靠性解释，不等价于临床病理机制解释。
5. 多目标优化提供的是 trade-off selection，不保证存在单一全局最优模型。
6. 路由策略改善的是 review/recover 决策，不应被描述为直接提高原始分类器本身的分类能力。

## 10. 可直接放入论文的英文表述

The proposed framework formulates reliable ECG classification as a causal-mechanism-aware multi-objective optimization problem. Instead of optimizing classification accuracy alone, we define a set of intervention variables, mechanism variables, and reliability outcomes. At the model layer, training constraints such as prototype, boundary, and reliability-guided losses are treated as interventions, while embedding geometry, local neighborhood purity, prototype ambiguity, softmax ambiguity, and validity signals are quantified as mediating mechanisms. At the routing layer, wavelet time-frequency features, regularity descriptors, mechanism-specific risk heads, and explanation-alignment scores are treated as evidence families for constructing complete review-routing policies. Candidate models and routing policies are evaluated by Pareto-style objectives, including accuracy, macro-F1, calibration error, VT/VF cross-errors, total errors, error migration, fixed-budget error capture, unresolved VT/VF risk, and explanation reliability. This design allows the study to test not only whether a candidate improves performance, but also through which measurable reliability mechanism the improvement occurs.
