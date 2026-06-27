# Risk-Pro-readable seed42 机制分析

## 这次新模型改了什么

旧的 Risk-Pro++ 把风险加权 CE、entropy alignment、prototype separation、contrastive、consistency、regularity auxiliary loss 组合在一起。它确实让表征更稳定，但前一轮机制分析显示：稳定性可能没有对齐正确边界，甚至会产生更多稳定自信的 VT/VF 错判。

这次新增的 `risk_pro_readable` 不是简单加大 PRO 权重，而是把训练目标拆成三类更明确的约束：

1. **VT/VF readability constraint**：增加 `--vtvf-readability-weight`，在 embedding 上训练一个 VT-vs-VF 线性辅助头，让 VT/VF 边界必须在表征层可线性读出。
2. **Selective stability constraint**：增加 `--selective-stability-consistency-weight` 和 `--selective-embedding-consistency-weight`，只对低风险样本强制扰动一致，避免把高风险边界样本也过度稳定化。
3. **Anti-confident-risk constraint**：增加 `--anti-confident-risk-weight`，对高 risk target 样本的高 softmax confidence 加惩罚，目标是减少“高风险但过度自信”的错误。

因此新版模型的研究含义是：不再追求所有样本都更稳定，而是追求 **低风险区域稳定、VT/VF 边界可读、高风险区域不过度自信**。

## seed42 分类结果

| variant | accuracy | macro-F1 | ECE | VT/VF cross-errors | total errors |
|---|---:|---:|---:|---:|---:|
| teacher | 0.9239 | 0.7618 | 0.0445 | 193 | 319 |
| PRO | 0.9075 | 0.7139 | 0.0738 | 281 | 388 |
| Risk-Pro++ | 0.9022 | 0.7057 | 0.0781 | 289 | 410 |
| Risk-Pro-readable | 0.9065 | 0.7055 | 0.0183 | 283 | 392 |

新版没有超过 teacher，也没有在 raw VT/VF cross-error 上超过 PRO。它的主要改进不在 accuracy，而在可靠性表达：

- ECE 从 Risk-Pro++ 的 0.0781 降到 0.0183；
- total errors 从 Risk-Pro++ 的 410 降到 392；
- VT->VF 从 Risk-Pro++ 的 250 降到 221；
- 但 VF->VT 从 39 升到 62，所以 VT/VF 总互错只小幅改善。

这说明新版目标确实改变了模型行为，但还没有形成最终性能优势。

## 表征稳定性和边界可读性

| variant | layer | embedding shift | prediction flip | VT/VF probe AUROC | multiclass probe Macro-F1 | probe VT/VF errors |
|---|---|---:|---:|---:|---:|---:|
| teacher | fused | 5.787 | 0.105 | 0.788 | 0.703 | 275 |
| PRO | fused | 3.757 | 0.111 | 0.754 | 0.695 | 300 |
| Risk-Pro++ | fused | 2.909 | 0.064 | 0.781 | 0.692 | 307 |
| Risk-Pro-readable | fused | 3.012 | 0.056 | 0.793 | 0.697 | 302 |

这里能看到新版的作用：

- 相比 PRO，VT/VF probe AUROC 从 0.754 提升到 0.793，说明边界可读性确实被拉回来了。
- 相比 Risk-Pro++，prediction flip 从 0.064 降到 0.056，扰动后预测更稳定。
- 但 probe VT/VF errors 仍然高于 teacher，说明“边界可读”还没有完全转化成“边界判对”。

所以机制结论是：新版比 PRO 更像一个方向正确的模型，因为它不只是让几何稳定，而是让稳定性和 VT/VF 可读性部分对齐。

## 稳定自信错误

| variant | group | n | fraction | mean confidence | mean flip rate | mean embedding shift |
|---|---|---:|---:|---:|---:|---:|
| PRO | confident stable VT/VF cross-error | 24 | 0.0057 | 0.983 | 0.047 | 3.098 |
| Risk-Pro++ | confident stable VT/VF cross-error | 53 | 0.0126 | 0.970 | 0.050 | 3.308 |
| Risk-Pro-readable | confident stable VT/VF cross-error | 43 | 0.0103 | 0.931 | 0.036 | 3.203 |
| teacher | confident stable VT/VF cross-error | 2 | 0.0005 | 0.956 | 0.067 | 4.695 |

新版相对 Risk-Pro++ 有一个关键改善：稳定自信 VT/VF 错误从 53 降到 43，平均 confidence 从 0.970 降到 0.931。

这正是 anti-confident-risk loss 想要修的问题。它还没把错误数压到 teacher 水平，但已经证明新约束改变了错误性质：错误没有那么过度自信。

## 风险信号是否更能识别边界错误

| variant | softmax VT/VF ambiguity AUROC | KNN VT/VF mixing AUROC |
|---|---:|---:|
| teacher | 0.628 | 0.528 |
| PRO | 0.565 | 0.520 |
| Risk-Pro++ | 0.617 | 0.519 |
| Risk-Pro-readable | 0.690 | 0.610 |

这是新版最强的结果。它没有让所有分类指标都变好，但明显让 VT/VF 边界风险更可被检测：

- softmax boundary ambiguity AUROC 提升到 0.690；
- KNN VT/VF mixing AUROC 提升到 0.610。

这说明新模型更适合作为 **boundary-aware selective classifier**，而不是单纯追求最高 accuracy 的分类器。

## 当前结论

Risk-Pro-readable 是比旧 Risk-Pro++ 更合理的研究方向，但还不是最终模型。

它的优点：

- 明显改善 calibration；
- 降低 Risk-Pro++ 的总错误；
- 降低稳定自信 VT/VF 错误的置信度；
- 显著增强 VT/VF boundary risk 的可检测性；
- 让“稳定性 + 可读性 + 反自信错误”成为可训练、可分析的明确结构。

它的不足：

- raw VT/VF cross-errors 仍高于 teacher；
- VF->VT 错误增加，说明边界方向性还没平衡；
- fused embedding 的 probe VT/VF errors 仍高；
- 单 seed 结果只能作为 pilot，不能作为最终结论。

## 下一步建议

下一步不应该直接宣称新版更好，而应该做一个小型消融矩阵：

1. Risk-Pro++ 原版；
2. Risk-Pro++ + VT/VF readability；
3. Risk-Pro++ + selective stability；
4. Risk-Pro++ + anti-confident risk；
5. Risk-Pro-readable full。

评价指标不要只看 accuracy，而要看：

- VT/VF cross-errors；
- ECE；
- stable confident VT/VF cross-errors；
- softmax VT/VF ambiguity AUROC；
- KNN VT/VF mixing AUROC；
- prediction-set / review-routing capture。

如果 full 版本在 3-5 seeds 上稳定降低稳定自信 VT/VF 错误，并增强 boundary risk AUROC，即使 raw accuracy 不超过 teacher，也可以形成一个顶刊级别的论点：**表征约束的价值不是盲目提升分类准确率，而是把危险边界错误变得可识别、可拒判、可进入 review。**
