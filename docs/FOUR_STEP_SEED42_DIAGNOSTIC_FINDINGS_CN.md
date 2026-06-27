# 四步可靠性诊断：Seed 42 研究发现

本文件记录当前 ECG 不确定性项目在 seed 42 上的结构化诊断结果。核心目的不是证明模型已经具备临床可用性，而是回答一个研究问题：

> 当表征空间看起来被拉开以后，为什么 SR/VT/VF 的最终决策可靠性并没有同步改善？

结论需要谨慎理解：这里是单 seed 的机制诊断证据，不是最终统计显著性结论，也不是临床验证。

## Step 1. 表征层诊断

目标是观察模型内部表征是否沿着网络逐步区分 SR、VT、VF，以及 VT/VF 混叠主要出现在哪些阶段。

已新增脚本：

```bash
python -m src.layerwise_representation_diagnosis --mat RHYTHMS.mat --run-dir <run-dir>
```

主要输出：

- `layerwise_representation_summary.csv`
- `layerwise_representation_diagnosis.json`
- `layerwise_centroid_distances.png`

### 关键发现

Teacher 模型的 fused embedding 在测试集上有较好的整体 silhouette（0.452），但 VT/VF 测试中心距离只有 0.680，明显低于训练中心距离 1.893。这说明模型在训练表征中可以形成一定的 VT/VF 几何分离，但这种分离在测试集上收缩。

更细的逐层结论如下：

| Model | Layer | Silhouette | SR-VT test dist | SR-VF test dist | VT-VF test dist | VT/VF mixing |
|---|---|---:|---:|---:|---:|---:|
| Teacher | waveform embedding | 0.412 | 2.418 | 2.374 | 0.688 | 0.297 |
| Teacher | regularity feature embedding | 0.063 | 1.154 | 1.236 | 0.463 | 0.336 |
| Teacher | fused embedding | 0.452 | 2.284 | 2.669 | 0.680 | 0.316 |
| Teacher | classifier logits | 0.571 | 3.455 | 4.107 | 0.698 | 0.309 |
| PRO | fused embedding | 0.547 | 2.494 | 3.453 | 0.761 | 0.347 |
| PRO | classifier logits | 0.628 | 3.251 | 4.720 | 0.841 | 0.341 |
| RISK-PRO++ | fused embedding | 0.612 | 2.746 | 3.845 | 0.866 | 0.354 |
| RISK-PRO++ | classifier logits | 0.693 | 3.291 | 4.890 | 0.954 | 0.346 |

这张表回答了四个具体问题：

1. **是否逐层分开？**
   整体上是逐层分开的。Teacher 从 waveform embedding 的 silhouette 0.412 提升到 classifier logits 的 0.571；PRO 提升到 0.628；RISK-PRO++ 提升到 0.693。说明模型后层确实让整体三分类结构更清晰。

2. **哪一类最容易混淆？**
   一直是 VT/VF。以 Teacher 为例，fused embedding 中 SR-VT 距离为 2.284、SR-VF 距离为 2.669，但 VT-VF 只有 0.680。也就是说，SR 与 ventricular rhythm 分得开，但 VT 和 VF 之间仍然贴得很近。

3. **哪一层开始暴露 VT/VF 混叠？**
   regularity feature embedding 最明显。Teacher 的 regularity feature embedding silhouette 只有 0.063，VT-VF test distance 只有 0.463，说明手工节律特征本身对 SR/VT/VF 的整体三分类分离能力较弱，尤其无法单独把 VT/VF 稳定分开。融合后整体分离提升，但 VT/VF 局部混叠没有消失。

4. **PRO/RISK-PRO++ 是否真的改善了表征？**
   它们改善了全局几何指标。RISK-PRO++ 的 fused silhouette 从 Teacher 的 0.452 提高到 0.612，classifier logits silhouette 从 0.571 提高到 0.693，VT-VF test distance 也从 0.680 提高到 0.954。但它没有降低 VT/VF mixing，反而从 Teacher 的 0.316 上升到 0.346-0.354。因此改善的是全局 separation，不是局部 boundary reliability。

### 错误样本是否真的靠近 VT/VF 边界？

是的。Teacher 的 fused embedding 中，正确 VT/VF 样本的 prototype VT/VF ambiguity 为 0.710，而 VT/VF 交叉错误样本为 0.712，二者都很高，说明大量 ventricular 样本本来就位于 VT/VF prototype 边界附近。

PRO 和 RISK-PRO++ 会降低这个 ambiguity 数值，但并没有降低交叉错误：

| Model | Correct VT/VF ambiguity | VT/VF error ambiguity | VT/VF cross-error rate |
|---|---:|---:|---:|
| Teacher | 0.710 | 0.712 | 0.046 |
| PRO | 0.506 | 0.565 | 0.067 |
| RISK-PRO++ | 0.398 | 0.455 | 0.069 |

这进一步说明：中心距离和 prototype ambiguity 不是充分指标。模型可以让错误样本看起来离 prototype boundary 更远，但最终 classifier 仍然产生更多 VT/VF 互错。这也是为什么后续必须进入决策边界诊断，而不能只停在 PCA/embedding 图。

PRO 和 RISK-PRO++ 都把训练集 VT/VF 中心距离推得更大：

| Model | Fused VT/VF train distance | Fused VT/VF test distance | VT/VF mixing | VT/VF cross-error rate |
|---|---:|---:|---:|---:|
| Teacher | 1.893 | 0.680 | 0.316 | 0.046 |
| PRO | 3.471 | 0.761 | 0.347 | 0.067 |
| RISK-PRO++ | 4.643 | 0.866 | 0.354 | 0.069 |

这是一条重要负结果：更大的训练表征中心距离没有带来更低的 VT/VF 交叉错误。RISK-PRO++ 的整体几何分离更强，但局部 VT/VF 邻域混合仍然更高，最终错误也更多。

## Step 2. 决策边界诊断

目标是判断错误来自两类机制：

- 表征重叠：样本在 embedding 空间中已经更接近错误类别原型；
- 分类头边界错位：样本更接近真实类别原型，但 classifier head 仍然判错。

已新增脚本：

```bash
python -m src.decision_boundary_diagnosis --run-dir <run-dir>
```

主要输出：

- `decision_boundary_diagnosis.csv`
- `decision_boundary_summary.csv`
- `decision_boundary_mechanism_counts.csv`
- `decision_boundary_vtvf_margin_map.png`

### 关键发现

三组模型的错误都主要来自表征重叠，而不是单纯 classifier head 边界放错。

| Model | Correct | Representation overlap | Classifier boundary mismatch | Mixed/outlying |
|---|---:|---:|---:|---:|
| Teacher | 3875 | 281 | 30 | 8 |
| PRO | 3806 | 368 | 7 | 13 |
| RISK-PRO++ | 3784 | 393 | 17 | 0 |

这说明“冻结 backbone 后只重新训练线性分类头”不能根本解决问题。实际验证也支持这一点：

| Model | Frozen plain macro-F1 | Frozen plain VT/VF cross-errors | Frozen balanced macro-F1 | Frozen balanced VT/VF cross-errors |
|---|---:|---:|---:|---:|
| Teacher | 0.704 | 259 | 0.702 | 276 |
| PRO | 0.694 | 285 | 0.694 | 299 |
| RISK-PRO++ | 0.685 | 303 | 0.692 | 307 |

class-balanced head 没有降低 VT/VF 交叉错误，说明 class imbalance 不是当前 seed 42 下的唯一主因。

## Step 3. 不确定性机制拆解

当前项目中每个不确定性信号对应不同风险含义：

| Signal | 捕获的问题 | 当前解释 |
|---|---|---|
| Entropy / MSP | softmax 犹豫 | 对错误检测有效，但可能漏掉高置信错误 |
| Temperature scaling | 置信度校准 | Teacher ECE 从 0.044 降到 0.016，但它不改变表征结构 |
| KNN mixing | 局部邻域标签混乱 | 对 VT/VF 结构性混叠更直接 |
| Prototype distance | 是否不像任何典型类 | 可解释，但中心距离不是充分条件 |
| Regularity features | 节律和频域异常 | 提供生理信号层面的辅助解释 |
| Conformal prediction | 输出集合不确定性 | 对 VT/VF 边界样本可输出 `{VT, VF}`，避免强制单类判断 |

Teacher 的 temperature scaling 明显改善 ECE（0.044 到 0.016），但这只是概率校准，不等于 VT/VF 边界变好。也就是说，概率可以更诚实，但 embedding 混叠仍然存在。

## Step 4. 从解释到改进

当前证据不支持继续把所有信号强行塞进 backbone loss 里。RISK-PRO++ 已经显示：

- 训练中心距离更大；
- 测试 silhouette 更高；
- 但 VT/VF 交叉错误更多；
- ECE 更差；
- representation-overlap 错误更多。

更合理的下一步是把研究贡献从“更强 backbone”转向“表征证据驱动的决策可靠性”：

1. 冻结 teacher backbone，保留稳定表征。
2. 单独训练或拟合轻量决策层，而不是继续大幅改 backbone。
3. 建立 VT/VF 专门边界校准器，输入可以包括 logit margin、prototype margin、KNN mixing、regularity features 和 entropy。
4. 对边界样本输出 `{VT, VF}` prediction set，而不是强制输出一个单一类别。
5. 对结构性高风险样本做 review routing，报告固定复核比例下能捕获多少 VT/VF 交叉错误。

## 可以写给导师的研究表述

本项目的研究价值不只是构建 ECG 分类器，而是系统分析深度模型在 VT/VF 等相近高风险节律之间为何产生不可靠决策。实验显示，单纯扩大 embedding 中的类别中心距离并不能保证决策边界更可靠；相反，部分表征强化方法虽然改善了全局几何指标，却增加了局部 VT/VF 混叠和交叉错误。这提示可靠性改进应从单一 accuracy 或中心距离优化，转向多机制不确定性诊断、边界校准、conformal prediction set 和复核路由。

## 当前局限

- 以上结论目前主要来自 seed 42，需要多 seed 复核。
- 仍需把各类不确定性信号在固定 review budget 下做成统一比较表。
- 数据来自现有数据集，缺少外部验证，因此不能声称临床泛化。
- VT/VF 边界的可解释性还需要结合更多错误样本图和 regularity feature 分布展示。

## 下一步建议

优先实现一个冻结 backbone 的 VT/VF boundary calibrator。它不直接改变原模型表征，而是用已有诊断信号学习什么时候应该：

- 保持原始单类预测；
- 输出 `{VT, VF}` 集合；
- 标记为需要人工复核的结构性高风险样本。

该闭环已经实现，详见 `docs/VT_VF_DECISION_CALIBRATION_ENDPOINT_CN.md`。对应脚本为：

```bash
python -m src.vtvf_decision_calibration --run-dir <run-dir>
```
