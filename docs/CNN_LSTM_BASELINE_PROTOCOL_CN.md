# CNN vs CNN-LSTM 组合模型基线实验

本实验线用于回答一个朴素但重要的问题：

> 在同一 ECG 数据、同一 split、同一训练设置下，CNN-LSTM 这种“卷积 + 序列建模”的组合模型，是否比普通 CNN 更好？

这里的目的不是替代后续可靠性研究，而是作为 dissertation / GitHub 中的传统模型基线对比。

## 模型含义

### 普通 CNN

`src.models.ECGCNN`

作用：

- 用 1D convolution 提取 ECG 局部波形形态；
- 适合捕捉 QRS 形态、局部峰值、短时模式；
- 结构简单，是传统 baseline。

### CNN-LSTM

`src.models.CNNLSTMClassifier`

作用：

- 先用 CNN 提取局部 waveform features；
- 再把 CNN 输出当作 sequence 输入 LSTM；
- LSTM 捕捉较长时间上的节律变化和前后依赖；
- 理论上更适合 ECG 这种时序信号。

模型路径：

```text
ECG window -> CNN local encoder -> LSTM temporal encoder -> embedding -> classifier -> SR/VT/VF
```

## 比较指标

不要只看 accuracy。推荐同时报告：

| Metric | 含义 |
|---|---|
| Accuracy | 总体分类正确率 |
| Macro-F1 | 对类别不平衡更敏感，避免 SR 多导致虚高 |
| ECE | 置信度校准误差 |
| Mean margin | top-1 概率减 top-2 概率，越大表示越果断 |
| Low-margin rate | 模型犹豫样本比例 |
| VT/VF cross-errors | VT 判成 VF 或 VF 判成 VT 的高风险错误 |
| Per-class recall | SR、VT、VF 每类召回率 |

这里的 margin 对应你说的 “manning” 我先按概率间隔理解。如果你原本指的是另一个指标，后续可以替换，但当前这个 margin 很适合衡量模型是否在 VT/VF 边界犹豫。

## 已有工具

新增比较脚本：

```bash
python -m src.compare_classification_runs \
  --run CNN <cnn-run-dir> \
  --run CNN-LSTM <cnn-lstm-run-dir> \
  --baseline CNN \
  --out results/cnn_lstm_baseline_comparison
```

输出：

- `classification_run_comparison.csv`
- `classification_run_deltas.csv`
- `classification_run_comparison.json`

## 正式实验命令

为了公平比较，CNN 和 CNN-LSTM 必须使用相同 seed、相同 split、相同 epoch。

如果使用原始 record-level split：

```powershell
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 42 --split-grouping record
python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 42 --split-grouping record
```

如果使用当前更严格的 duplicate-family split：

```powershell
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 42 --split-grouping duplicate_family
python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 42 --split-grouping duplicate_family
```

建议最终跑 3 个 seeds：

```powershell
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 42 --split-grouping duplicate_family
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 43 --split-grouping duplicate_family
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 44 --split-grouping duplicate_family

python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 42 --split-grouping duplicate_family
python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 43 --split-grouping duplicate_family
python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 44 --split-grouping duplicate_family
```

然后逐 seed 比较：

```powershell
python -m src.compare_classification_runs --run CNN <cnn-seed42-run> --run CNN-LSTM <cnn-lstm-seed42-run> --baseline CNN --out results/cnn_lstm_comparison_seed42
python -m src.compare_classification_runs --run CNN <cnn-seed43-run> --run CNN-LSTM <cnn-lstm-seed43-run> --baseline CNN --out results/cnn_lstm_comparison_seed43
python -m src.compare_classification_runs --run CNN <cnn-seed44-run> --run CNN-LSTM <cnn-lstm-seed44-run> --baseline CNN --out results/cnn_lstm_comparison_seed44
```

## Seed 42 正式结果：duplicate-family split

已完成一组正式 seed 42 对比。两组模型使用相同数据、相同 seed、相同 duplicate-family split、相同 12 epochs。

运行命令：

```powershell
python -m src.train --mat RHYTHMS.mat --model cnn --epochs 12 --seed 42 --split-grouping duplicate_family --out results\cnn_lstm_baseline_20260626
python -m src.train --mat RHYTHMS.mat --model cnn_lstm --epochs 12 --seed 42 --split-grouping duplicate_family --out results\cnn_lstm_baseline_20260626
python -m src.compare_classification_runs --run CNN results\cnn_lstm_baseline_20260626\20260626_003641_cnn --run CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_004030_cnn_lstm --baseline CNN --out results\cnn_lstm_baseline_20260626\seed42_comparison
```

总体结果：

| Model | Accuracy | Macro-F1 | ECE | Total errors | VT/VF cross-errors | Mean margin |
|---|---:|---:|---:|---:|---:|
| CNN | 0.859 | 0.641 | 0.076 | 591 | 349 | 0.875 |
| CNN-LSTM | 0.905 | 0.726 | 0.045 | 399 | 242 | 0.883 |

CNN-LSTM 相对 CNN 的变化：

| Metric | CNN-LSTM minus CNN |
|---|---:|
| Accuracy | +0.046 |
| Macro-F1 | +0.085 |
| ECE | -0.031 |
| Total errors | -192 |
| VT/VF cross-errors | -107 |
| Mean margin | +0.008 |

分类别表现：

| Model | SR recall | VT recall | VF recall | SR F1 | VT F1 | VF F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.948 | 0.407 | 0.836 | 0.963 | 0.491 | 0.469 |
| CNN-LSTM | 0.978 | 0.573 | 0.758 | 0.977 | 0.678 | 0.522 |

解释：

- CNN-LSTM 明显优于普通 CNN：accuracy、macro-F1、ECE、total errors 和 VT/VF cross-errors 都改善。
- CNN-LSTM 对 VT 的召回提升很明显，从 0.407 到 0.573。
- VF recall 从 0.836 降到 0.758，但 VF F1 仍从 0.469 提升到 0.522，说明 precision/recall 的整体平衡更好。
- VT/VF 交叉错误减少 107 个，说明组合模型不仅提高总体准确率，也改善了高风险类别对的互错。
- 目前这仍然只是 seed 42，不能作为最终统计结论；建议继续跑 seed 43 和 44。

## 六个 seed 的正式汇总：duplicate-family split

已完成 seed 42、43、44、45、46、47 的成对实验。每一对都使用相同数据、相同 duplicate-family split 规则、相同 epoch 数，只改变模型结构：

- baseline：普通 CNN；
- comparator：CNN-LSTM；
- 训练轮数：12 epochs；
- seeds：42、43、44、45、46、47。

复现实验命令示例：

```powershell
python -m src.aggregate_classification_comparisons `
  --comparison 42=results\cnn_lstm_baseline_20260626\seed42_comparison `
  --comparison 43=results\cnn_lstm_baseline_20260626\seed43_comparison `
  --comparison 44=results\cnn_lstm_baseline_20260626\seed44_comparison `
  --comparison 45=results\cnn_lstm_baseline_20260626\seed45_comparison `
  --comparison 46=results\cnn_lstm_baseline_20260626\seed46_comparison `
  --comparison 47=results\cnn_lstm_baseline_20260626\seed47_comparison `
  --out results\cnn_lstm_baseline_20260626\multiseed_summary_6seed
```

### 每个 seed 的变化量

下表表示 `CNN-LSTM minus CNN`。Accuracy、Macro-F1、Mean margin 越高通常越好；ECE、Total errors、VT/VF cross-errors 越低越好。

| Seed | Accuracy Δ | Macro-F1 Δ | ECE Δ | Mean margin Δ | VT/VF cross-errors Δ | Total errors Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.046 | +0.085 | -0.031 | +0.008 | -107 | -192 |
| 43 | -0.063 | +0.027 | +0.057 | +0.069 | -45 | +240 |
| 44 | -0.031 | -0.050 | -0.028 | -0.106 | +21 | +139 |
| 45 | -0.047 | -0.131 | +0.047 | +0.008 | +17 | +148 |
| 46 | -0.012 | -0.014 | +0.024 | +0.025 | +18 | +55 |
| 47 | +0.055 | +0.056 | -0.025 | +0.021 | +17 | -255 |

### 聚合变化

| Metric | Mean Δ | Median Δ | Direction |
|---|---:|---:|---|
| Accuracy | -0.009 | -0.021 | 2/6 seeds 改善，平均略差 |
| Macro-F1 | -0.004 | +0.007 | 3/6 seeds 改善，整体接近持平但不稳定 |
| ECE | +0.007 | -0.001 | 3/6 seeds 改善，平均略差 |
| Mean margin | +0.004 | +0.015 | 5/6 seeds 增大，但不等于更可靠 |
| VT/VF cross-errors | -13.2 | +17.0 | 2/6 seeds 减少，4/6 seeds 增加 |
| Total errors | +22.5 | +97.0 | 2/6 seeds 减少，4/6 seeds 增加 |

### 研究解释

六个 seed 后，结论比三 seed 更清楚：这个结果不支持“CNN-LSTM 全面优于 CNN”的简单结论。更准确的结论是：

> CNN-LSTM 有时可以提高总体分类表现，例如 seed 42 和 seed 47；但它没有稳定提高 accuracy、Macro-F1 或 ECE，也没有稳定降低 VT/VF 之间的高风险互错。特别是 VT/VF cross-errors 在 4/6 seeds 中反而增加，说明组合时序结构并不能可靠解决 VT/VF 决策边界问题。

这正好强化本项目的研究主线：只看 accuracy 会掩盖模型真实风险。一个模型可能提高总体正确率，却让 VT/VF 之间的互错更多；也可能让 probability margin 变大，但校准误差和错误类型并没有同步改善。因此后续更有价值的方向不是继续盲目堆模型，而是把模型拆成四个层面分析：

1. 表征层是否真的把 SR/VT/VF 分开；
2. VT/VF 错误是 embedding 重叠造成，还是 classifier head 边界放错；
3. entropy、KNN mixing、prototype distance、margin、conformal set 分别捕获哪一种不确定性；
4. 在冻结 backbone 的基础上，训练 decision head 或 calibration head，让解释信号转化为更可靠的决策。

### 负结果和局限性

- CNN-LSTM 没有稳定赢过 CNN，这是负结果，但不是失败结果；它说明结构复杂度不能自动带来可靠性提升。
- seed45 和 seed46 中 CNN-LSTM 同时恶化 accuracy、Macro-F1、ECE、VT/VF cross-errors 和 total errors，说明组合结构在某些 split 下会带来更差的优化或边界放置。
- seed47 中 CNN-LSTM 明显减少 total errors，但 VT/VF cross-errors 仍从 125 增加到 142，说明总体分类提升和高风险边界可靠性不是同一件事。
- 六个 seeds 比三 seeds 更稳，但仍然不能声称统计显著；这适合作为 dissertation/GitHub 的研究证据，而不是最终临床证据。
- 当前数据来自单一项目数据源，缺少外部验证集，不能表述为医学诊断能力或临床部署结论。

博士申请材料里建议这样写：

> Beyond aggregate classification accuracy, I evaluated whether a CNN-LSTM architecture improves the reliability of SR/VT/VF ECG classification under a duplicate-family split. Across six paired seeds, CNN-LSTM did not consistently improve accuracy, Macro-F1, calibration, or VT/VF boundary reliability. Although it improved total errors in some seeds, VT/VF cross-errors increased in four of six seeds. This mixed and partially negative result motivated a deeper uncertainty-oriented analysis of representation geometry, decision-boundary placement, and post-hoc calibration, rather than simply increasing model complexity.

## 现有旧结果参考

早期旧实验中还比较过 CNN 和 BiGRU。它不是 CNN-LSTM 的正式结论，但可以作为“单纯序列模型不一定更好”的参考。

| Model | Accuracy | Macro-F1 | ECE | VT/VF cross-errors | Mean margin |
|---|---:|---:|---:|---:|---:|
| CNN | 0.924 | 0.716 | 0.016 | 181 | 0.874 |
| BiGRU | 0.819 | 0.534 | 0.095 | 344 | 0.820 |

这个旧对比说明：单纯换成序列模型不一定更好。BiGRU 比 CNN 差很多，但这不能直接代表 CNN-LSTM，因为 CNN-LSTM 先用 CNN 提取局部形态，再用 LSTM 建模时序，结构更合理。

## 预期写法

如果 CNN-LSTM 赢：

> CNN-LSTM improves over the plain CNN baseline, suggesting that temporal dependencies beyond local convolutional morphology contribute to SR/VT/VF rhythm discrimination.

如果 CNN-LSTM 没赢：

> CNN-LSTM does not outperform the plain CNN baseline under the same data split, suggesting that the available 5-second ECG windows may be sufficiently captured by local convolutional morphology, or that recurrent temporal modelling introduces additional optimization difficulty under limited data.

如果 accuracy 赢但 VT/VF cross-errors 更差：

> CNN-LSTM improves overall accuracy but worsens VT/VF boundary reliability, reinforcing the need to evaluate high-risk class-pair errors rather than relying on aggregate accuracy alone.
