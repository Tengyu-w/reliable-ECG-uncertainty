# CNN vs CNN-LSTM：10-seed 成对统计不确定性分析

本文件记录 CNN vs CNN-LSTM 从 6 seeds 扩展到 10 paired seeds 后的统计结果。重点不是追求漂亮的 `p < 0.05`，而是报告效应大小、方向一致性、bootstrap 区间、Bayesian interval、paired tests，以及 duplicate-family cluster bootstrap。

重要边界：这是 ECG 可靠性研究原型证据，不是临床验证或医疗器械性能声明。

## 实验设置

- Split：`duplicate_family`
- 模型：CNN vs CNN-LSTM
- Paired seeds：42、43、44、45、46、47、48、49、50、51
- 每个 seed 下 CNN 和 CNN-LSTM 使用相同数据 split 和相同训练轮数
- 新增 seeds：48、49、50、51
- 训练轮数：12 epochs

新增统计脚本：

```bash
python -m src.paired_statistical_uncertainty \
  --comparison <seed>=<comparison_dir> \
  --pair <seed> CNN CNN-LSTM <cnn_run_dir> <cnn_lstm_run_dir> \
  --out results/cnn_lstm_baseline_20260626/paired_statistical_uncertainty_10seed
```

主要输出：

- `results/cnn_lstm_baseline_20260626/multiseed_summary_10seed/`
- `results/cnn_lstm_baseline_20260626/paired_statistical_uncertainty_10seed/paired_seed_uncertainty_summary.csv`
- `results/cnn_lstm_baseline_20260626/paired_statistical_uncertainty_10seed/duplicate_family_cluster_bootstrap_summary.csv`

## 10-seed 汇总

下表均为 `CNN-LSTM minus CNN`。Accuracy、Macro-F1、Mean margin 越高通常越好；ECE、VT/VF cross-errors、Total errors 越低越好。

| Metric | Mean delta | Median delta | Direction |
|---|---:|---:|---|
| Accuracy | -0.0131 | -0.0123 | 4/10 seeds 增加，6/10 seeds 降低 |
| Macro-F1 | +0.0216 | +0.0377 | 6/10 seeds 增加 |
| ECE | +0.0077 | +0.0113 | 6/10 seeds 变差 |
| Mean margin | -0.0105 | +0.0058 | 方向混合 |
| VT/VF cross-errors | -49.3 | -37.5 | 6/10 seeds 减少 |
| Total errors | +51.4 | +67.0 | 6/10 seeds 增加 |

和 6-seed 结果相比，新增 48-51 后，CNN-LSTM 对 VT/VF cross-errors 的证据变强：seed 48、49、50、51 都减少了 VT/VF cross-errors，尤其 seed 49 减少 261 个。但是 total errors 和 accuracy 仍然不稳定。

## Seed-level bootstrap CI

Seed-level bootstrap 是以 10 个 paired seeds 为重采样单位，估计 mean delta 的 95% CI。

| Metric | Mean delta | 95% seed bootstrap CI |
|---|---:|---:|
| Accuracy | -0.0131 | [-0.0367, +0.0123] |
| Macro-F1 | +0.0216 | [-0.0251, +0.0641] |
| ECE | +0.0077 | [-0.0148, +0.0303] |
| VT/VF cross-errors | -49.3 | [-104.1, -6.2] |
| Total errors | +51.4 | [-53.2, +149.7] |

解释：

- Accuracy、Macro-F1、ECE、Total errors 的 seed bootstrap CI 都跨 0，不能说稳定改善。
- VT/VF cross-errors 的 seed bootstrap CI 不跨 0，提示 CNN-LSTM 在 10 seeds 下对 VT/VF 互错有更稳定的减少趋势。
- 但这仍然是 10 seeds 的内部 split 证据，不是外部验证。

## Bayesian credible interval

Bayesian interval 使用 paired seed differences 的弱假设正态均值模型。它回答：给定这 10 个 seeds，真实 mean effect 的后验区间和方向概率是多少。

| Metric | 95% Bayesian credible interval | P(delta preferred direction) |
|---|---:|---:|
| Accuracy | [-0.0430, +0.0165] | 0.169 for improvement |
| Macro-F1 | [-0.0325, +0.0754] | 0.805 for improvement |
| ECE | [-0.0198, +0.0353] | 0.270 for improvement |
| VT/VF cross-errors | [-111.0, +12.7] | 0.948 for reduction |
| Total errors | [-75.1, +175.5] | 0.189 for reduction |

解释：

- VT/VF cross-errors 的 posterior probability of reduction 为 0.948，方向证据较强。
- 但 credible interval 仍轻微跨 0，因此更稳妥的表述是“stronger evidence of reduction”，而不是“definitive reduction”。
- Total errors 的 preferred direction probability 只有 0.189，说明总体错误没有改善证据。

## Paired significance tests

由于只有 10 paired seeds，p-value 只能作为描述性辅助，不应作为主要结论。

| Metric | paired t-test p | Wilcoxon p | sign-flip permutation p |
|---|---:|---:|---:|
| Accuracy | 0.344 | 0.322 | 0.328 |
| Macro-F1 | 0.389 | 0.322 | 0.381 |
| ECE | 0.542 | 0.625 | 0.529 |
| VT/VF cross-errors | 0.105 | 0.078 | 0.070 |
| Total errors | 0.379 | 0.432 | 0.371 |

解释：

- 没有指标达到传统 `p < 0.05`。
- VT/VF cross-errors 的 Wilcoxon 和 sign-flip permutation p 值接近 0.05-0.10，和 bootstrap/Bayesian 结果一致：方向证据增强，但仍应谨慎。
- 论文中可以说：paired tests were descriptive and underpowered at n=10。

## Duplicate-family cluster bootstrap

Cluster bootstrap 是在每个 seed 的 test set 内，以 duplicate-family group 为单位重采样，而不是以单个 ECG window 为单位。这避免把高度相似 windows 当成独立样本。

主要看 `vtvf_cross_error_rate_within_vtvf_delta`，即在真实 VT/VF 样本中 VT/VF 互错率的变化。

| Seed | Mean delta | 95% duplicate-family cluster CI |
|---:|---:|---:|
| 42 | -0.117 | [-0.237, +0.015] |
| 43 | -0.114 | [-0.267, -0.016] |
| 44 | +0.052 | [-0.056, +0.176] |
| 45 | +0.027 | [-0.016, +0.107] |
| 46 | +0.020 | [-0.070, +0.098] |
| 47 | +0.033 | [-0.033, +0.110] |
| 48 | -0.129 | [-0.323, +0.084] |
| 49 | -0.272 | [-0.604, +0.058] |
| 50 | -0.219 | [-0.431, -0.052] |
| 51 | -0.085 | [-0.151, -0.031] |

Pooled descriptive cluster bootstrap:

| Metric | Mean delta | 95% descriptive CI |
|---|---:|---:|
| accuracy_delta | -0.012 | [-0.142, +0.106] |
| macro_f1_delta | +0.021 | [-0.191, +0.204] |
| error_rate_delta | +0.012 | [-0.106, +0.142] |
| vtvf_cross_error_rate_within_vtvf_delta | -0.080 | [-0.415, +0.106] |

解释：

- 在 individual seed 内，seed 43、50、51 的 VT/VF 互错率 cluster CI 明确为负，说明这些 split 下 CNN-LSTM 对 VT/VF 边界有较稳定改善。
- 其他 seeds 的 CI 跨 0，说明不同 duplicate families 组成会明显影响结论。
- Pooled descriptive CI 仍跨 0，说明不能把窗口级结果过度解释为泛化结论。

## 研究结论

10-seed 后，结论比 6-seed 更细：

> CNN-LSTM provides stronger evidence for reducing VT/VF cross-errors than in the earlier 6-seed analysis, but this improvement is not accompanied by stable gains in accuracy, calibration, or total error reduction. The result supports a narrower claim: recurrent temporal modelling may reduce some high-risk VT/VF boundary confusions, but it is not a reliable general solution for ECG classification reliability.

中文解释：

> CNN-LSTM 在 10 seeds 下对 VT/VF 互错的减少证据更强，但它没有稳定提高整体 accuracy、ECE 或 total errors。因此它不能被表述为“全面更可靠的模型”。更准确的结论是：CNN-LSTM 可能缓解部分 VT/VF 边界互错，但不能替代后续的 frozen-backbone boundary calibration、prediction set 和 review routing。

## 对最终论文主线的意义

这组结果不推翻 frozen-backbone 方向，反而让主线更清楚：

1. CNN-LSTM 对 VT/VF 互错有一定帮助，但不是全面可靠性提升。
2. Accuracy 和 total errors 仍不稳定，说明单标签 backbone comparison 不足以定义安全可靠性。
3. 因此最终贡献仍应是：用表征诊断和边界风险校准，把高风险 VT/VF forced-choice errors 转化为 prediction set 或 expert-review routing。

建议论文表述：

> Across ten paired duplicate-family seeds, CNN-LSTM reduced VT/VF cross-errors on average, with seed-level bootstrap evidence favouring a reduction. However, this boundary-specific improvement did not translate into stable gains in accuracy, calibration, or total error reduction. These results motivate treating recurrent modelling as a partial backbone-level mitigation, while the main reliability contribution remains representation-guided boundary calibration and set-valued/review-routed decision policies.
