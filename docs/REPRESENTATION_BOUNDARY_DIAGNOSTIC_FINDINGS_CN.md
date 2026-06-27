# 表征层与决策边界诊断：6-seed CNN vs CNN-LSTM

本文件回答一个核心研究问题：

> CNN-LSTM 没有稳定改善 VT/VF 可靠性，到底是因为 embedding 表征层没有把 VT/VF 分开，还是因为 classifier head 的决策边界放错了？

这里暂不进入校准机制分析，只做两个诊断：

1. 表征层诊断：embedding 是否把 SR/VT/VF 分开，尤其 VT/VF 是否混叠。
2. 决策边界诊断：错误来自 representation overlap，还是 classifier boundary mismatch。

## 实验设置

- 数据划分：duplicate-family split。
- 对比模型：CNN vs CNN-LSTM。
- seeds：42、43、44、45、46、47。
- 每个 seed 成对比较：CNN 和 CNN-LSTM 使用相同 split。
- 诊断脚本：
  - `src/layerwise_representation_diagnosis.py`
  - `src/decision_boundary_diagnosis.py`
  - `src/aggregate_representation_boundary_diagnostics.py`

聚合命令：

```powershell
python -m src.aggregate_representation_boundary_diagnostics `
  --run 42 CNN results\cnn_lstm_baseline_20260626\20260626_003641_cnn `
  --run 42 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_004030_cnn_lstm `
  --run 43 CNN results\cnn_lstm_baseline_20260626\20260626_005036_cnn `
  --run 43 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_005407_cnn_lstm `
  --run 44 CNN results\cnn_lstm_baseline_20260626\20260626_010216_cnn `
  --run 44 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_084006_cnn_lstm `
  --run 45 CNN results\cnn_lstm_baseline_20260626\20260626_090424_cnn `
  --run 45 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_090759_cnn_lstm `
  --run 46 CNN results\cnn_lstm_baseline_20260626\20260626_091631_cnn `
  --run 46 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_091952_cnn_lstm `
  --run 47 CNN results\cnn_lstm_baseline_20260626\20260626_092709_cnn `
  --run 47 CNN-LSTM results\cnn_lstm_baseline_20260626\20260626_093032_cnn_lstm `
  --out results\cnn_lstm_baseline_20260626\representation_boundary_diagnostics_6seed
```

## 指标含义

| 指标 | 含义 |
|---|---|
| `final_vt_vf_norm_dist_test` | test embedding 中 VT 与 VF 中心距离，按类内离散程度归一化；越高代表 VT/VF 越分开 |
| `final_vtvf_mixing_ventricular` | VT/VF 样本的 KNN 邻域中，对侧室性类别比例；越高代表局部混叠越严重 |
| `representation_overlap` | 预测错，并且最近 embedding 原型也指向错类；说明表征层本身支持了错误 |
| `classifier_boundary_mismatch` | 预测错，但最近 embedding 原型仍指向真类；说明 classifier head 边界可能放错 |
| `frozen_plain_head` | 冻结 embedding 后重新训练线性 head，用于诊断 head 是否有改进空间 |

## 表征层证据

6-seed 均值：

| Model | Accuracy | Macro-F1 | VT/VF cross-errors | VT/VF center distance | VT/VF KNN mixing |
|---|---:|---:|---:|---:|---:|
| CNN | 0.844 | 0.610 | 223.3 | 0.926 | 0.323 |
| CNN-LSTM | 0.835 | 0.606 | 210.2 | 0.904 | 0.332 |

CNN-LSTM 相对 CNN 的平均变化：

| Metric | Mean Δ | Median Δ | Interpretation |
|---|---:|---:|---|
| Accuracy | -0.009 | -0.021 | 没有稳定提升整体分类 |
| Macro-F1 | -0.004 | +0.007 | 基本持平，但 seed 间波动明显 |
| VT/VF cross-errors | -13.2 | +17.0 | 平均略少，但 4/6 seeds 反而增加 |
| VT/VF center distance | -0.022 | -0.031 | 没有把 VT/VF 稳定拉开 |
| VT/VF KNN mixing | +0.010 | -0.004 | 局部邻域混叠没有改善，平均略差 |

结论：

> CNN-LSTM 并没有稳定提高 VT/VF embedding 可分性。它没有让 VT/VF 中心距离稳定变大，也没有让 VT/VF 局部 KNN 混叠稳定下降。因此，CNN-LSTM 不是一个可靠的表征层修复方案。

## 决策边界证据

6-seed 均值：

| Model | Representation overlap | Classifier boundary mismatch |
|---|---:|---:|
| CNN | 0.135 | 0.0169 |
| CNN-LSTM | 0.152 | 0.0096 |

解释：

- `representation_overlap` 远高于 `classifier_boundary_mismatch`。
- 这说明多数错误不是“embedding 已经对了，但 classifier head 判错”。
- 更多错误是“embedding 最近原型本身已经偏向错类”，也就是表征空间发生了类别混叠。
- CNN-LSTM 的 representation-overlap 比例还从 0.135 上升到 0.152，说明组合时序结构没有稳定减少表征混叠。

按 seed 看，CNN-LSTM 的 representation-overlap 相对 CNN：

| Seed | Δ representation overlap | Δ classifier boundary mismatch |
|---:|---:|---:|
| 42 | -0.006 | -0.039 |
| 43 | +0.091 | -0.025 |
| 44 | +0.027 | +0.001 |
| 45 | +0.048 | -0.000 |
| 46 | +0.003 | +0.011 |
| 47 | -0.060 | +0.008 |

结论：

> 在多数 seeds 中，CNN-LSTM 没有减少 representation overlap，甚至经常增加。它有时减少 classifier-boundary mismatch，但这不足以转化为稳定的 VT/VF 边界可靠性提升。

## Frozen Head 反推验证

冻结 embedding 后重新训练线性 head，用来检查“如果只换 head，能不能修复问题”。

6-seed 均值：

| Model | Original Macro-F1 | Frozen plain Macro-F1 | Original VT/VF cross-errors | Frozen plain VT/VF cross-errors |
|---|---:|---:|---:|---:|
| CNN | 0.610 | 0.653 | 223.3 | 210.0 |
| CNN-LSTM | 0.606 | 0.636 | 210.2 | 201.8 |

这个结果说明：

- 重新训练 head 可以带来一定改善，说明 classifier head 的确有一部分责任。
- 但改善幅度有限，且没有消除 VT/VF cross-errors。
- 因为 representation-overlap 仍然是主要错误机制，所以仅靠换 head 不足以根治问题。

更准确的判断是：

> VT/VF 可靠性问题主要来自表征层混叠，classifier head 边界错放是次要但真实存在的因素。

## 可证明或可验证的结论

### 结论 1：CNN-LSTM 不是稳定的表征改进

证据：

- 6-seed mean accuracy：CNN-LSTM 比 CNN 低 0.009。
- 6-seed mean Macro-F1：CNN-LSTM 比 CNN 低 0.004。
- VT/VF center distance：CNN-LSTM 比 CNN 低 0.022。
- VT/VF KNN mixing：CNN-LSTM 比 CNN 高 0.010。

因此不能说 CNN-LSTM 学到了更可靠的 VT/VF 表征。

### 结论 2：VT/VF 错误主要不是简单边界错放

证据：

- CNN representation-overlap：0.135。
- CNN classifier-boundary mismatch：0.0169。
- CNN-LSTM representation-overlap：0.152。
- CNN-LSTM classifier-boundary mismatch：0.0096。

representation-overlap 显著高于 boundary mismatch，说明很多错误在 embedding 空间已经发生类别混叠。

### 结论 3：重新训练 head 有帮助，但不是根本解决方案

证据：

- CNN frozen head Macro-F1 从 0.610 提高到 0.653，VT/VF cross-errors 从 223.3 降到 210.0。
- CNN-LSTM frozen head Macro-F1 从 0.606 提高到 0.636，VT/VF cross-errors 从 210.2 降到 201.8。

这说明 classifier head 有优化空间，但由于 representation-overlap 仍占主导，单纯改 head 不能充分解决 VT/VF 混淆。

## 对后续研究的意义

这组结果支持下一阶段研究方向：

1. 不应继续单纯堆 CNN-LSTM 或更复杂 backbone。
2. 应把重点放在 VT/VF 表征混叠的解释：哪些样本在 embedding 中落到对侧原型附近，哪些样本 KNN 邻域混杂。
3. 决策边界分析仍有价值，因为 frozen head 可以带来小幅改善，但它应作为验证工具，而不是目前的主线解决方案。
4. 在暂不做校准机制的前提下，下一步最有价值的是输出 VT/VF boundary case atlas：列出高 overlap、高 KNN mixing、高 confidence wrong 的样本类型。

## 限制

- 当前是 6 seeds，仍不能声称统计显著。
- 数据来自单一项目数据源，不能作为临床泛化结论。
- 诊断基于模型 embedding 和原型距离，是机制证据，不等同于因果证明。
- 结论应表述为 research prototype evidence，不应表述为医学诊断或临床部署能力。
