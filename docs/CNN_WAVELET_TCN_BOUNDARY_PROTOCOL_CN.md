# CNN + Wavelet + TCN + Boundary Adapter 实验说明

## 研究动机

前一版 `CNN + TCN + Validity Bottleneck` 证明了模型可以学到较强的 validity/risk 信号，但这个信号没有稳定转化成更少的 VT/VF 边界错误。这说明瓶颈不只是“是否知道哪里危险”，还包括“边界附近是否有足够可分的波形信息”。

本版本把 ECG 的波形本质显式放进模型：在 CNN 局部形态分支之外，加入固定多尺度 wavelet-like 滤波器，让模型能同时看到不同尺度的尖峰、斜率和振荡响应。

## 模型结构

```text
ECG window
  -> CNN morphology branch
  -> fixed multi-scale wavelet filter bank -> wavelet encoder
  -> feature fusion
  -> dilated TCN rhythm branch
  -> embedding
  -> validity bottleneck
  -> main SR/VT/VF classifier
  -> gated VT/VF specialist boundary adapter
```

## 和只加 loss 的区别

只加 loss 是训练阶段的约束，推理时仍然是普通分类路径。

这个模型把解释分析得到的结构假设放进推理路径：预测必须经过 wavelet/time-frequency 表征、validity bottleneck 和 VT/VF specialist adapter。也就是说，模型不只是被惩罚“应该更可靠”，而是在结构上被要求利用边界有效性信号修正 VT/VF 决策。

## 需要观察的指标

- 总体 accuracy、macro-F1、ECE。
- VT/VF cross-errors：`VT -> VF` 和 `VF -> VT`。
- 总错误数：避免模型只减少 VT/VF 错误却制造更多 SR 错误。
- gate/boundary score 对任意错误和 VT/VF 边界错误的 AUROC。
- embedding silhouette、KNN boundary risk、softmax boundary risk。

## 解释标准

如果 wavelet 版本优于 `CNN + TCN + Validity v2`，说明单纯 validity gate 不够，边界修正需要额外的时频信息。

如果它接近或超过 `CNN + LSTM`，可以形成一个更强的机制论结论：可靠性不是来自时序平滑本身，而是来自“局部形态 + 时频尺度 + validity-domain gating”的组合。

如果它仍然不能超过 `CNN + LSTM`，则说明当前数据内 VT/VF 边界可能需要更强的窗口级上下文、分层决策或 record-cluster-level 约束，而不是继续堆局部前端。
