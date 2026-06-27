# CNN+TCN+Validity Bottleneck 10seed 综合对比

日期：2026-06-27

## 实验范围

本轮完成了 `CNN + TCN + Validity Bottleneck` 的 10seed 训练与综合对比。

完整 seeds：

- seed42, seed43, seed44, seed45, seed46, seed47, seed48, seed49, seed50, seed51

完整 run manifest：

- `results/cnn_tcn_validity_20260626/summary/cnn_tcn_validity_complete_manifest.csv`

综合对比输出：

- `results/cnn_tcn_validity_20260626/summary/combined_model_metric_summary.csv`
- `results/cnn_tcn_validity_20260626/summary/cnn_tcn_validity_paired_delta_summary.csv`
- `results/cnn_tcn_validity_20260626/summary/cnn_tcn_validity_gate_summary.csv`

本轮对比对象：

- CNN
- CNN-LSTM
- Teacher
- ProRisk
- CNN-TCN-Validity

注意：这是内部数据、duplicate-family split 下的研究原型结果，不是临床验证。

## 1. 模型均值对比

| 模型 | Accuracy | Macro-F1 | ECE | Total errors | VT/VF cross-errors | Embedding silhouette | Softmax VT/VF boundary AUROC | KNN VT/VF boundary AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.8649 | 0.5928 | 0.0670 | 589.5 | 232.4 | 0.2822 | 0.6824 | 0.6500 |
| CNN-LSTM | 0.8518 | 0.6145 | 0.0747 | 640.9 | 183.1 | 0.4102 | 0.6464 | 0.6648 |
| Teacher | 0.8997 | 0.6499 | 0.0554 | 435.4 | 189.7 | 0.4688 | 0.6747 | 0.6383 |
| ProRisk | 0.8951 | 0.6496 | 0.0535 | 459.8 | 198.2 | 0.5553 | 0.6685 | 0.6100 |
| CNN-TCN-Validity | 0.8600 | 0.6078 | 0.0872 | 607.8 | 219.3 | 0.4867 | 0.7013 | 0.6667 |

## 2. 相对 CNN 的 paired delta

CNN-TCN-Validity - CNN：

| 指标 | Mean delta | Bootstrap 95% CI | 方向 |
|---|---:|---:|---|
| Accuracy | -0.0048 | [-0.0218, 0.0100] | 无稳定提升 |
| Macro-F1 | +0.0150 | [-0.0087, 0.0392] | 有趋势，不稳定 |
| ECE | +0.0202 | [0.0049, 0.0385] | 明确变差 |
| Total errors | +18.3 | [-41.5, 86.2] | 不稳定 |
| VT/VF cross-errors | -13.1 | [-38.3, 15.7] | 有减少趋势，不稳定 |
| Embedding silhouette | +0.2045 | [0.1621, 0.2486] | 明确更规整 |
| Softmax VT/VF boundary AUROC | +0.0189 | [-0.0262, 0.0629] | 不稳定 |
| KNN VT/VF boundary AUROC | +0.0166 | [-0.0493, 0.0686] | 不稳定 |

解释：

相对传统 CNN，新模型确实让表征明显更规整，但没有稳定提升分类性能；ECE 还变差。这说明 TCN+validity 结构第一版并没有直接成为更好的分类器。

## 3. 相对 CNN-LSTM 的 paired delta

CNN-TCN-Validity - CNN-LSTM：

| 指标 | Mean delta | Bootstrap 95% CI | 方向 |
|---|---:|---:|---|
| Accuracy | +0.0082 | [-0.0194, 0.0349] | 不稳定 |
| Macro-F1 | -0.0067 | [-0.0546, 0.0404] | 不稳定 |
| ECE | +0.0125 | [-0.0157, 0.0406] | 不稳定，偏差 |
| Total errors | -33.1 | [-140.7, 80.9] | 不稳定 |
| VT/VF cross-errors | +36.2 | [-10.6, 88.3] | 相比 CNN-LSTM 偏差 |
| Embedding silhouette | +0.0764 | [-0.0040, 0.1377] | 更规整趋势 |
| VT/VF normalized separation | +0.0850 | [-0.0402, 0.2122] | 趋势 |
| Softmax VT/VF boundary AUROC | +0.0549 | [0.0039, 0.1022] | 明确更可检测 |
| KNN VT/VF boundary AUROC | +0.0019 | [-0.0756, 0.0655] | 无改善 |

解释：

这是最关键的结果：

1. 新模型没有超过 CNN-LSTM 的 VT/VF cross-error 控制。
2. 但是新模型显著提高了 softmax VT/VF boundary detectability。
3. 也就是说，validity bottleneck 第一版提高了“边界可检测性”，但没有把这种可检测性稳定转化成更少的边界错误。

这正好验证了我们之前的核心问题：有了显式 validity 结构以后，模型确实开始“知道哪里危险”，但 boundary adapter 还没有足够强地修正这些危险决策。

## 4. 相对 Teacher / ProRisk

CNN-TCN-Validity 相对 Teacher：

- Accuracy delta: -0.0397，CI [-0.0631, -0.0205]
- Macro-F1 delta: -0.0421，CI [-0.0839, -0.0057]
- ECE delta: +0.0318，CI [0.0117, 0.0550]
- Total errors delta: +172.4，CI [93.8, 263.4]
- VT/VF cross-errors delta: +29.6，CI [-17.6, 87.7]

CNN-TCN-Validity 相对 ProRisk：

- Accuracy delta: -0.0351，CI [-0.0626, -0.0073]
- Macro-F1 delta: -0.0418，CI [-0.0831, 0.0012]
- ECE delta: +0.0337，CI [0.0126, 0.0560]
- Total errors delta: +148.0，CI [28.7, 263.9]
- VT/VF cross-errors delta: +21.1，CI [-21.1, 68.2]
- KNN VT/VF boundary AUROC delta: +0.0567，CI [-0.0052, 0.1138]

解释：

新模型不能替代 Teacher 或 ProRisk 作为当前最优分类模型。它的价值主要在机制验证：它让边界风险更显性，但还没有把风险显性化转化成更高可靠性。

## 5. Validity gate 是否学到了东西？

这是本轮最重要的结构分析。

| Gate 指标 | Mean | Bootstrap 95% CI |
|---|---:|---:|
| gate mean | 0.2908 | [0.2772, 0.3041] |
| gate correct mean | 0.2483 | [0.2385, 0.2582] |
| gate error mean | 0.5559 | [0.5296, 0.5809] |
| gate VT/VF correct mean | 0.6413 | [0.5825, 0.6954] |
| gate VT/VF boundary error mean | 0.7566 | [0.7213, 0.7875] |
| gate any-error AUROC | 0.9095 | [0.8781, 0.9365] |
| boundary-score any-error AUROC | 0.9147 | [0.8824, 0.9401] |
| gate VT/VF-boundary AUROC | 0.6555 | [0.5846, 0.7212] |
| boundary-score VT/VF-boundary AUROC | 0.6537 | [0.5834, 0.7198] |

确认事实：

1. Gate 对任意错误非常敏感：any-error AUROC 约 0.91。
2. Gate 在错误样本上的均值 0.556，明显高于正确样本 0.248。
3. Gate 在 VT/VF boundary error 上也更高：0.757 vs VT/VF correct 0.641。
4. 但 gate 对 VT/VF boundary error 的 AUROC 只有约 0.656，说明它能感知边界风险，但不够尖锐。

这说明结构不是没学到东西。它确实学到了 validity/risk signal。但是第一版 adapter 没有把这个 signal 成功转化为分类边界修正。

## 6. 综合结论

本轮结果不是一个简单的“成功”或“失败”，而是一个很有价值的机制定位：

**已成功的部分：**

- CNN+TCN+Validity 明显提高了表征规整性。
- 相比 CNN-LSTM，它提高了 softmax VT/VF boundary detectability。
- Gate/boundary score 能很好识别任意错误。
- Gate 对 VT/VF boundary error 有中等检测力。

**没有成功的部分：**

- 没有稳定降低 VT/VF cross-errors。
- 没有超过 CNN-LSTM 的边界错误控制。
- 没有超过 Teacher/ProRisk 的总体分类性能。
- ECE 比 CNN、CNN-LSTM、Teacher、ProRisk 都更差。

最准确的结论是：

> CNN+TCN+Validity Bottleneck 第一版证明了“显式 validity 结构可以学习错误/边界风险”，但还没有证明“validity-conditioned adapter 能稳定修正 VT/VF 决策边界”。

这比 ProRisk 结果更进一步，因为 ProRisk 的问题是“几何规整但错误也被规整”；而这次新模型的问题是“风险被识别出来了，但修正机制还不够强”。

## 7. 下一步应该怎么改

第一版没有使用 teacher risk targets，只用了标签级 boundary/gate target、consistency 和 VT/VF readability。当前 gate 主要学到的是“错误风险”和“ventricular/boundary tendency”，但 boundary adapter 没有被明确训练成“修正 VT/VF 互错”。

下一版建议：

1. 用 validity-domain map 生成 risk targets，训练 gate 对齐真实 high-risk / boundary-risk，而不是只用 ventricular 标签。
2. 给 boundary adapter 加专门的 VT/VF correction objective，只在 VT/VF 子空间内训练 specialist head。
3. 把 gate 从 `main + gate * adapter` 改成更保守的 mixture-of-experts：

```text
final_logits = (1 - gate) * main_logits + gate * boundary_specialist_logits
```

4. 引入 prototype memory：特别是 stable-correct prototypes 和 high-confidence-error prototypes。
5. 如果仍然不能降低 VT/VF cross-errors，再进入第二个大模型方向：`CNN + Wavelet + TCN + Boundary Adapter`。

## 论文表达建议

这轮可以写成机制闭环中的一个关键实验：

> We first converted representation diagnostics into a decision-time validity bottleneck. The bottleneck learned a strong error-validity signal, but its first adapter implementation did not yet translate boundary awareness into reduced VT/VF cross-errors. This indicates that reliability requires not only validity detection, but also boundary-specific corrective capacity.

中文：

> 我们把表征层解释第一次真正放进了预测路径。模型确实学会了识别不可靠区域，但第一版边界 adapter 还没有足够能力修正 VT/VF 边界错误。因此下一步的核心不是再证明风险可检测，而是增强边界专门修正机制。
