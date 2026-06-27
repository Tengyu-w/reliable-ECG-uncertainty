# CNN + Wavelet + TCN + Boundary Adapter Seed42 结果

## 实验设置

- 模型：`cnn_wavelet_tcn_boundary`
- 数据：`RHYTHMS.mat`
- split：`duplicate_family`
- seed：42
- epoch：8
- 训练约束：
  - auxiliary ventricular boundary loss
  - validity gate target/sparsity
  - mild perturbation stability consistency
  - embedding consistency
  - VT/VF readability auxiliary head
  - VT/VF specialist loss

完整运行目录：

`results/cnn_wavelet_tcn_boundary_20260627/20260627_105654_cnn_wavelet_tcn_boundary_wavelet_boundary_seed42`

## 核心结果

| model | seed | accuracy | macro-F1 | ECE | VT/VF cross-errors | total errors |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 42 | 0.8591 | 0.6409 | 0.0761 | 349 | 591 |
| CNN-LSTM | 42 | 0.9049 | 0.7255 | 0.0451 | 242 | 399 |
| CNN+TCN+Validity v2 | 42 | 0.8972 | 0.6912 | 0.0466 | 286 | 431 |
| CNN+Wavelet+TCN+Boundary | 42 | 0.8701 | 0.6158 | 0.0788 | 415 | 545 |

## 混淆矩阵结论

Wavelet 版本不是简单地“更差”，而是出现了方向性偏置：

- `VT -> VF`：412
- `VF -> VT`：3
- `VT -> SR`：68
- `VF -> SR`：6

也就是说，模型几乎很少把 VF 判成 VT，但大量把 VT 推成 VF。它提升了 VF 敏感性，却牺牲了 VT 可读性和 VT/VF 边界方向。

## Gate 与 Boundary Score

| signal | any-error AUROC | VT/VF-boundary AUROC | correct mean | error mean |
|---|---:|---:|---:|---:|
| validity gate | 0.9084 | 0.9450 | 0.1525 | 0.4739 |
| boundary score | 0.9163 | 0.9563 | 0.1051 | 0.8251 |

这说明模型不是“不知道哪里危险”。相反，它非常清楚哪些样本是边界/高风险样本。真正失败点在于：当前 boundary adapter 看到风险后，修正方向偏向 VF，导致 VT 被大量规整到 VF。

## 机制解释

这次实验支持一个更细的论文结论：

> 加入 wavelet/time-frequency 前端以后，模型的 validity/boundary detection 可以更强，但“识别风险”仍然不等于“可靠地修正边界”。如果 adapter 没有方向约束，边界修正会变成单向风险保守，把 VT 大量推向 VF。

这和我们之前的主结论一致，但更进一步：

- `CNN-LSTM`：时间平滑更好，VT/VF cross-errors 少，但不是完整可靠性。
- `CNN+TCN+Validity v2`：validity gate 有效，整体接近 CNN-LSTM，但没有超过。
- `CNN+Wavelet+TCN+Boundary`：边界风险识别很强，但修正方向失衡。

## 当前判断

不建议直接对当前 wavelet 版本做 10seed，因为 seed42 已经暴露出明显机制缺陷。下一步更有价值的是做方向约束版：

1. 限制 adapter 只能在 VT/VF 子空间内做有符号修正。
2. 增加 VT-preservation loss，避免 VT 大量坍缩到 VF。
3. 把 gate 拆成两个方向：`VT-risk-to-VF` 和 `VF-risk-to-VT`，而不是一个统一风险门。
4. 单独评估 VT recall、VF recall 和 VT/VF directional error balance。

这个负结果可以写进论文机制链条：它证明“更多波形尺度信息 + 更强风险识别”仍不自动产生可靠边界，必须有方向性边界控制。
