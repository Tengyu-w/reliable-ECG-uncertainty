# 2026-06-30 导师交流录音要点：因果优化与机制解释

录音转写文件：

```text
results/advisor_meeting_transcript_20260630/transcript_timestamped.txt
results/advisor_meeting_transcript_20260630/transcript_plain.txt
```

说明：自动转写对 ECG、VT/VF、embedding、KNN 等术语有误识别，以下是基于关键语义的人工整理。

## 1. 前段作业部分

录音前约 0-3 分钟主要是 coursework/assignment 评分解释。和当前 ECG 毕业论文关系不大，可忽略。

## 2. 老师的核心判断

老师认为你的方向是有价值的，尤其是：

- 不只是追求 benchmark/accuracy；
- 试图理解模型为什么错；
- 试图解释模型内部表征；
- 试图用这些解释设计改进策略；
- 这和 explainable AI / model reliability 是相关研究方向。

对应录音大意：

> 如果你能弄清楚模型为什么表现不好，理论上你就应该能提出策略去改善它。

## 3. 老师指出的主要问题

老师觉得目前还不够清楚的是：

1. 你到底是在展示一个好的 ECG classifier；
2. 还是在展示某种模型方法/机制方法更好；
3. 还是在评估模型内部机制如何影响可靠性。

这三者必须区分，否则论文主线会显得太大。

对应录音大意：

> At the moment it's not clear whether the classification is important, or whether the model approach is important, or whether you are evaluating the model approach.

## 4. 定量内证是关键

老师明确说需要：

```text
some sort of quantifiable measure
```

也就是：不能只说 embedding、KNN、central distance、signal features 很重要，而要定量证明它们为什么重要。

在本项目中应转化为：

```text
do(training constraint)
    -> mechanism variable changes
        -> model outcome changes
```

例如：

| Intervention | Mechanism variable | Outcome |
| --- | --- | --- |
| prototype margin | VT/VF center distance, prototype distance, KNN mixing | VT/VF cross-errors |
| boundary CE | high-risk boundary weighting | total errors, migration penalty |
| regularity auxiliary | rhythm/morphology regularity representation | SR/VT/VF confusion |
| validity branch | error/correct separability | error AUROC, calibration |
| wavelet branch | time-frequency boundary evidence | VT/VF boundary detection |

## 5. 老师对 feature engineering 的理解

当你解释把 embedding、KNN、central distance、signal structure 放进模型时，老师理解为：

> Rather than just taking raw signals, you are using information about the data and feeding that into the model. This is basically feature engineering with your models.

这说明论文中可以把这些机制变量表述为：

- mechanism-informed features；
- representation-level evidence；
- reliability evidence variables；
- model-side auxiliary evidence。

但必须证明这些变量和模型结果之间有关系。

## 6. 老师认可 recovery/routing 的逻辑

你解释说不能疯狂增强模型本体，因为加入太多机制变量后模型可能学不好，甚至会让 VT/VF 边界被拉开但类别方向不清楚。因此你把这些机制放入 recover/routing 部分。

老师的反应是：

> On the surface it sounds great. It sounds really good.

这支持你把论文分成两层：

1. model-side improvement；
2. mechanism/routing/recover reliability layer。

但 routing 层也需要定量证明为什么这些机制变量能决定 recover policy。

## 7. 对论文最重要的改法

论文主线建议改成：

> 本文不是单纯追求最高 ECG 分类准确率，而是研究模型可靠性：先分析模型在 VT/VF/SR 边界、embedding neighborhood、prototype distance、waveform regularity 等机制上的失败原因，再定量验证这些机制变量如何影响模型 outcome，并进一步用因果-Pareto优化选择模型约束和 routing/recover 策略。

## 8. 下一步应该补的实验

应补三张定量内证表：

```text
1. intervention_to_mechanism_effects.csv
2. mechanism_to_outcome_association.csv
3. mediation_or_path_effect_summary.csv
```

对应问题：

1. 加某个训练约束后，机制变量是否变化？
2. 机制变量变化是否对应错误减少/校准改善？
3. 训练约束的效果是否部分通过这些机制变量传递？

## 9. 论文表述边界

不能写：

> 我做了很多机制分析，所以模型更好。

应该写：

> 我将机制分析转化为可量化的中介变量，并验证训练干预、机制变量和模型结果之间的统计关系。

