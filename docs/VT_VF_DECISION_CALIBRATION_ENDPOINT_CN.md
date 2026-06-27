# VT/VF 决策校准闭环结果

本文件是四步可靠性研究路线的当前终点：在不继续强行修改 backbone 的前提下，用表征层、局部邻域、prototype、softmax 和校准信号构建一个冻结 backbone 后的 VT/VF 决策校准原型。

研究问题：

> 如果单纯扩大 embedding 中心距离不能稳定改善 VT/VF 决策，那么能否把表征诊断证据转化为更可靠的下游决策策略？

重要边界：以下结果是研究原型证据，不是临床验证，也不是医疗器械性能声明。

## 方法

新增脚本：

```bash
python -m src.vtvf_decision_calibration --run-dir <run-dir>
```

该脚本只使用已有模型输出，不重新训练 backbone：

1. 从 frozen backbone 的 train/val/test embeddings 和 logits 中提取证据。
2. 用 train embedding 建立 prototype 和 KNN 邻域。
3. 在 validation set 上拟合轻量逻辑回归风险校准器。
4. 在 test set 上评估：
   - VT/VF boundary risk 的 AUROC/AUPR；
   - 固定 review budget 下能捕获多少 VT/VF 交叉错误；
   - 高风险 VT/VF 边界样本输出 `{VT,VF}` prediction set 后，自动单标签错误是否减少。

使用的主要证据包括：

- entropy / MSP：softmax 是否犹豫；
- temperature scaling：概率校准；
- softmax VT/VF ambiguity：VT 与 VF 概率是否接近；
- prototype VT/VF ambiguity：样本离 VT/VF 中心是否相近；
- KNN VT/VF mixing：局部邻域是否混入另一类 ventricular rhythm；
- logit/probability VT/VF margin：分类头是否强行二选一；
- nearest prototype agreement：classifier head 与 embedding geometry 是否一致。

## 单标签基线

| Model | Single-label accuracy | Error rate | VT/VF cross-errors | ECE before T | ECE after T |
|---|---:|---:|---:|---:|---:|
| Teacher | 0.924 | 0.076 | 193 | 0.044 | 0.016 |
| PRO | 0.907 | 0.093 | 281 | 0.074 | 0.039 |
| RISK-PRO++ | 0.902 | 0.098 | 289 | 0.078 | 0.025 |

PRO 和 RISK-PRO++ 在表征几何上更强，但单标签 VT/VF 交叉错误更多。因此，当前证据不支持继续简单强化 prototype/center-distance loss 作为主要路线。

## Learned VT/VF Boundary Risk

轻量校准器在测试集上识别 VT/VF 交叉错误的能力：

| Model | VT/VF boundary risk AUROC | VT/VF boundary risk AUPR | Any-error risk AUROC |
|---|---:|---:|---:|
| Teacher | 0.908 | 0.191 | 0.874 |
| PRO | 0.913 | 0.264 | 0.884 |
| RISK-PRO++ | 0.900 | 0.253 | 0.847 |

这说明虽然单标签分类没有变好，但已有表征证据足以预测“哪些样本容易发生 VT/VF 互错”。这就是本项目从分类任务上升为可靠性研究的关键价值。

## 固定复核比例下的错误捕获

使用 learned VT/VF boundary risk 对样本排序，在测试集上按固定比例送复核：

| Model | Review budget | VT/VF cross-error captured | All-error captured | Auto VT/VF cross-error rate after action |
|---|---:|---:|---:|---:|
| Teacher | 10% | 0.383 | 0.238 | 0.032 |
| Teacher | 20% | 1.000 | 0.740 | 0.000 |
| PRO | 10% | 0.388 | 0.284 | 0.046 |
| PRO | 20% | 1.000 | 0.822 | 0.000 |
| RISK-PRO++ | 10% | 0.218 | 0.166 | 0.060 |
| RISK-PRO++ | 20% | 1.000 | 0.798 | 0.000 |

结果显示：在 20% review budget 下，三组模型的 VT/VF 交叉错误都可以被完全捕获。这个结果应被表述为 review-routing/reliability evidence，而不是自动诊断性能。

## `{VT,VF}` Prediction Set 策略

另一种更适合写进 dissertation 的策略是：不强制所有 ventricular 边界样本二选一，而是在高风险边界样本上输出 `{VT,VF}` 集合。

| Model | Validation-selected budget | Test set action rate | Set contains true rate | Baseline VT/VF cross-error rate | Effective VT/VF cross-error rate |
|---|---:|---:|---:|---:|---:|
| Teacher | 10% | 0.124 | 0.989 | 0.046 | 0.022 |
| Teacher | 20% | 0.183 | 0.977 | 0.046 | 0.000 |
| PRO | 10% | 0.132 | 0.996 | 0.067 | 0.028 |
| PRO | 20% | 0.188 | 0.976 | 0.067 | 0.000 |
| RISK-PRO++ | 10% | 0.123 | 0.984 | 0.069 | 0.042 |
| RISK-PRO++ | 20% | 0.191 | 0.960 | 0.069 | 0.000 |

解释：

- 原模型必须输出 SR/VT/VF 中的一个类别；
- prediction set 策略允许模型在结构性边界样本上输出 `{VT,VF}`；
- 如果真实标签在集合中，则这个样本不再被看作“错误单标签自动决策”，而是被识别为 ventricular 边界不确定样本；
- 这显著降低了最危险的 VT/VF 强制互错。

## 核心研究结论

当前项目的最强结论不是“我们把 accuracy 提高了”，而是：

> 表征层诊断显示，VT/VF 的不可靠性主要来自局部表征重叠和边界样本混叠；单纯扩大类别中心距离不能保证决策可靠性。更有效的路线是冻结 backbone，将 embedding/prototype/KNN/softmax/temperature 信号转化为边界风险校准、prediction set 和 review routing，从而把高风险强制单标签错误转化为可解释的不确定集合或复核对象。

这比“输出一个 risk level”更有研究性，因为它解释了：

- 模型在哪里不可靠；
- 不同不确定性信号分别捕获什么；
- 为什么某些表征优化会失败；
- 如何把失败机制转化成决策层面的改进。

## 可以作为 Dissertation 的章节结构

1. **Problem formulation**：SR/VT/VF 分类中的可靠性问题，特别是 VT/VF 边界风险。
2. **Representation diagnosis**：层级 embedding、PCA/UMAP/LDA、center distance、KNN mixing。
3. **Decision-boundary diagnosis**：prototype agreement、classifier mismatch、class imbalance probe。
4. **Uncertainty mechanism decomposition**：entropy、KNN、prototype、regularity、conformal。
5. **Decision calibration**：frozen backbone + lightweight VT/VF boundary calibrator。
6. **Prediction-set / review-routing policy**：从强制单标签转向 `{VT,VF}` 或复核。
7. **Negative results and limitations**：PRO/RISK-PRO++ 改善几何但没有改善单标签可靠性。

## 当前局限

- 这里仍是 seed 42 的闭环结果，需要在 seed 43/44 或已有多 seed 结果上复核。
- `{VT,VF}` set policy 降低的是强制单标签错误，不等价于自动给出最终临床判断。
- 数据规模和外部验证仍然有限，不能声称临床泛化。
- 下一步需要把 prediction set 的可解释样本图、regularity 特征分布和错误案例可视化加入公开报告。

## 建议写给潜在导师的一句话

This project investigates reliable ECG rhythm classification not only by improving predictive accuracy, but by diagnosing how representation geometry, local neighbourhood mixing, prototype ambiguity, and softmax uncertainty interact to produce VT/VF boundary failures. A key finding is that stronger embedding separation does not necessarily improve decision reliability; therefore, the project develops a frozen-backbone decision calibration framework that converts representation-level uncertainty into VT/VF prediction sets and review-routing evidence.
