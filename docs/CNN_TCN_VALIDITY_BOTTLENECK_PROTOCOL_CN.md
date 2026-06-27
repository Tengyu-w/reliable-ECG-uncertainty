# CNN + TCN + Validity Bottleneck 协议

日期：2026-06-26

## 目的

这个模型用于验证一个核心假设：

> 不是时序平滑本身带来可靠性，而是时序表征必须经过 validity-domain 约束，才能转化为 VT/VF 边界可靠性。

它不是普通的 `CNN + X` 堆模块，也不是 ProRisk 式的单纯 loss 正则。它把前面表征层分析得到的机制问题变成了预测路径中的显式结构。

## 结构

模型名称：`cnn_tcn_validity`

实现位置：

- `src/models.py`: `CNNTCNValidityBottleneck`
- `src/train.py`: 训练入口支持 `--model cnn_tcn_validity`
- `src/run_cnn_tcn_validity_experiment.py`: 推荐实验入口

结构：

```text
ECG waveform
  ↓
CNN morphology branch
  ↓
dilated TCN rhythm branch
  ↓
embedding z
  ↓
validity bottleneck b
  ├─ boundary_score
  └─ validity_gate
  ↓
final logits = main_logits(z) + validity_gate(b) * boundary_adapter_logits(z, b)
```

## 和之前 ProRisk / 边界 loss 的区别

之前：

```text
embedding → classifier
      ↑
 boundary/prototype/risk losses
```

也就是说，边界信息主要在训练时影响 embedding。预测时模型仍然走同一条分类路径。

现在：

```text
embedding → validity bottleneck → gate/adaptor → final logits
```

边界/有效域信息不仅参与训练，也参与预测时的 logits 生成。

核心区别：

| 维度 | ProRisk / loss 正则 | CNN+TCN+Validity Bottleneck |
|---|---|---|
| 边界信息使用时间 | 训练时 | 训练时 + 预测时 |
| 是否改变 logits 生成 | 间接 | 直接 |
| 是否有专门边界路径 | 无 | 有 boundary adapter |
| 表征规整风险 | 错误也可能被规整 | 可监控 gate 是否启用错误区域 |
| 论文定位 | regularization ablation | mechanism-guided architecture |

## 推荐初始实验

轻量单 seed：

```bash
python -m src.run_cnn_tcn_validity_experiment --mat RHYTHMS.mat --seeds 42 --epochs 8 --out results/cnn_tcn_validity_20260626
```

10seed：

```bash
python -m src.run_cnn_tcn_validity_experiment --mat RHYTHMS.mat --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 8 --out results/cnn_tcn_validity_20260626
```

如果后续生成了与同 split 对齐的 teacher risk target，可以加：

```bash
--risk-targets path/to/risk_targets.npz
```

此时会额外启用：

- risk-boundary alignment
- risk-gate alignment
- risk-entropy alignment
- boundary CE upweighting
- anti-confident high-risk penalty
- selective low-risk consistency

## 需要重点评估的指标

这个模型不应该只用 accuracy 判断。核心评估应该包括：

- VT/VF cross-errors 是否下降；
- stable confident VT/VF errors 是否下降；
- validity gate 是否在 VT/VF boundary 或 latent high-risk strata 中更高；
- embedding silhouette 是否提升时，边界 AUROC 是否同步提升；
- model disagreement 是否减少 both-agree-wrong；
- coverage-risk curve 是否优于 CNN-LSTM。

## 预期解释

如果成功，它支持：

> 时序建模本身只能带来平滑表征；只有当表征有效域被显式建模并参与决策路径时，平滑性才可能转化为边界可靠性。

如果失败，它也有价值：

> 显式 gate/adaptor 仍无法稳定改善 VT/VF 边界，说明当前 validity signal 或 boundary target 不够充分，需要引入 wavelet/time-frequency branch 或更强的 prototype memory。
