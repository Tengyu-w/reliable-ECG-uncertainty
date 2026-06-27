# CNN+TCN+Validity v2 协议

日期：2026-06-27

## 为什么做 v2

v1 的 10seed 结果显示：

- validity gate 能识别错误，any-error AUROC 约 0.91；
- gate 在 VT/VF boundary error 上更高；
- 但 VT/VF cross-errors 没有稳定下降。

因此 v1 的问题不是“不知道哪里危险”，而是：

> 知道危险以后，boundary adapter 没有足够能力修正 VT/VF 决策。

v2 的目标就是把 adapter 从“附加修正项”改成真正的 VT/VF specialist。

## 结构变化

v1：

```text
final logits = main logits + gate * adapter logits
```

v2：

```text
SR logit = main SR logit
VT/VF logits = (1 - gate) * main VT/VF logits + gate * VT/VF specialist logits
```

也就是说，v2 不让 specialist 乱动 SR，而是只专门处理 VT/VF 子空间。

## 新增训练目标

v2 新增：

```text
--vtvf-specialist-weight
```

它只在真实 VT/VF 样本上训练内部 `vtvf_specialist_logits`：

```text
VT -> 0
VF -> 1
```

这使 specialist 明确学习 VT/VF 边界，而不是被整体 SR/VT/VF 三分类目标稀释。

## 入口

模型：

```bash
--model cnn_tcn_validity_v2
```

推荐 runner：

```bash
python -m src.run_cnn_tcn_validity_v2_experiment --mat RHYTHMS.mat --seeds 42 --epochs 8 --out results/cnn_tcn_validity_v2_20260627
```

Smoke：

```bash
python -m src.run_cnn_tcn_validity_v2_experiment --mat RHYTHMS.mat --seeds 42 --epochs 1 --max-windows-per-record 1 --batch-size 32 --out results/cnn_tcn_validity_v2_smoke_20260627
```

## 判断标准

v2 是否成功，不应只看 accuracy。核心指标是：

- VT/VF cross-errors 是否低于 v1；
- 是否接近或超过 CNN-LSTM；
- gate 对 VT/VF boundary error 的 AUROC 是否提高；
- ECE 是否不再明显恶化；
- stable confident VT/VF errors 是否下降。

## 论文意义

如果 v2 成功，说明：

> validity bottleneck 不仅能识别风险，还能通过 VT/VF specialist 把风险信号转化为边界修正。

如果 v2 失败，说明：

> raw waveform + TCN 的 validity signal 仍不足以修正 VT/VF 边界，需要进入 wavelet/time-frequency 分支。
