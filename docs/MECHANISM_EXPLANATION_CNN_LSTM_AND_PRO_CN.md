# CNN-LSTM 稳定性与 PRO 失败机制解释

本文件回答两个机制问题：

1. CNN-LSTM 为什么表现出更好的扰动稳定性，但稳定性没有完整转化为可靠性？
2. PRO / prototype-style 表征增强为什么改善某些几何或稳定性，却没有稳定改善分类，甚至会恶化 VT/VF 错误？

重要边界：以下内容是 ECG 可靠性研究原型证据，不是临床验证，也不是医疗器械性能声明。

## 三个统一解释轴

为了避免 CNN-LSTM 和 PRO 各讲各的，本项目现在用同一套机制轴解释它们：

1. **Layer-wise perturbation stability**
   同一个 ECG window 加轻微扰动后，每一层表征移动多少、预测是否翻转。

2. **Stability-readability trade-off**
   一层表征更稳定，并不代表它更能区分 VT/VF。这里同时看：
   - embedding shift / prediction flip；
   - VT-vs-VF linear probe AUROC；
   - SR/VT/VF multiclass probe Macro-F1。

3. **Confident stable errors**
   统计那些“模型错了，但很自信，而且扰动后也不改”的样本。这个比普通错误更危险，因为它说明模型不是不确定地错，而是稳定地错。

新增脚本：

```bash
python -m src.representation_mechanism_analysis --mat RHYTHMS.mat --run-dir <run-dir> --model <model>
python -m src.aggregate_representation_mechanisms --run <family> <variant> <seed> <model> <run-dir> --out results/representation_mechanism_explanations_20260626
```

主要输出：

`results/representation_mechanism_explanations_20260626/`

## 1. CNN-LSTM：更平滑，但不是全面更可靠

### 1.1 Layer-wise perturbation stability

10-seed 机制聚合结果显示，CNN-LSTM 的最终层扰动稳定性更好。

| Model | Layer | Embedding shift | Prediction flip rate | VT-vs-VF probe AUROC | Multiclass probe Macro-F1 |
|---|---|---:|---:|---:|---:|
| CNN | conv3 | 3.374 | 0.163 | 0.686 | 0.631 |
| CNN | final embedding | 4.550 | 0.163 | 0.691 | 0.622 |
| CNN | classifier logits | 4.304 | 0.163 | 0.697 | 0.624 |
| CNN-LSTM | cnn_conv3 | 1.985 | 0.133 | 0.702 | 0.631 |
| CNN-LSTM | lstm_last_state | 2.101 | 0.133 | 0.708 | 0.631 |
| CNN-LSTM | final embedding | 2.680 | 0.133 | 0.708 | 0.632 |
| CNN-LSTM | classifier logits | 1.747 | 0.133 | 0.738 | 0.625 |

解释：

- CNN-LSTM 的 embedding shift 明显更低，prediction flip rate 也更低。
- 这支持“LSTM 后端让表征更平滑/更抗扰动”的说法。
- 但这种平滑不是简单等于整体可靠性，因为 10-seed 统计里 accuracy、ECE、total errors 仍然没有稳定改善。

### 1.2 Stability-readability trade-off

CNN-LSTM 相对 CNN 的平均变化：

| Layer pair | Shift delta | Flip delta | VT/VF AUROC delta | Multiclass Macro-F1 delta |
|---|---:|---:|---:|---:|
| final embedding | -1.870 | -0.030 | +0.017 | +0.011 |
| classifier logits | -2.557 | -0.030 | +0.042 | +0.001 |

解释：

- 10 seeds 后，CNN-LSTM 不仅更稳定，也对 VT/VF probe 有小幅正向变化。
- 但这个变化很小，且没有转化成稳定的 accuracy、ECE、total error 改善。
- 因此正确结论不是“LSTM 没用”，而是：

> CNN-LSTM provides partial boundary-level mitigation and perturbation smoothing, but this does not constitute a general reliability improvement.

中文：

> CNN-LSTM 对边界错误有一定缓解，也让表征更平滑；但它不是全面可靠性修复方案。

### 1.3 Confident stable errors

10-seed 均值：

| Model | Group | Mean n | Fraction | Mean confidence | Mean flip rate | Mean final shift |
|---|---|---:|---:|---:|---:|---:|
| CNN | any error | 589.5 | 0.135 | 0.748 | 0.326 | 3.414 |
| CNN | confident stable error | 104.3 | 0.0255 | 0.910 | 0.016 | 3.279 |
| CNN | confident stable VT/VF cross-error | 61.0 | 0.0146 | 0.886 | 0.011 | 3.040 |
| CNN-LSTM | any error | 640.9 | 0.148 | 0.732 | 0.363 | 3.091 |
| CNN-LSTM | confident stable error | 57.3 | 0.0147 | 0.932 | 0.030 | 2.414 |
| CNN-LSTM | confident stable VT/VF cross-error | 5.4 | 0.00135 | 0.886 | 0.033 | 2.013 |

解释：

- CNN-LSTM 的总错误均值反而更多，但 confident stable VT/VF cross-errors 明显更少。
- 这和 10-seed 统计一致：CNN-LSTM 对 VT/VF cross-errors 有更强减少证据，但不是全面减少错误。
- 这说明 LSTM 的优势更集中在“部分边界错误缓解”和“扰动平滑”，而不是整体分类可靠性。

### 1.4 CNN-LSTM 机制结论

更准确的表述是：

> CNN-LSTM smooths the representation and reduces some stable high-risk VT/VF boundary errors, but the smoothing does not reliably improve overall classification, calibration, or total error burden. Its benefit is boundary-specific and partial.

中文：

> CNN-LSTM 让表征更平滑，并减少一部分稳定自信的 VT/VF 边界错误；但这种平滑没有稳定提升整体分类、校准或总错误。因此它是部分边界缓解，不是通用可靠性方案。

## 2. PRO：几何/稳定性改善，但边界可读性和错误类型恶化

这里的 PRO 指 duplicate-family final 中的 prototype-style intervention，以及 core intervention seed42 中的 prototype separation / contrastive / RISK-PRO++ 线。

### 2.1 PRO 3-seed：表征更平滑，但 fused readability 下降

3-seed duplicate-family baseline vs PRO：

| Variant | Layer | Embedding shift | Prediction flip | VT/VF AUROC | Multiclass Macro-F1 | Probe VT/VF cross-errors |
|---|---|---:|---:|---:|---:|---:|
| baseline | waveform embedding | 4.826 | 0.101 | 0.780 | 0.745 | 158.0 |
| baseline | fused embedding | 5.742 | 0.101 | 0.795 | 0.733 | 160.7 |
| baseline | classifier logits | 2.433 | 0.101 | 0.776 | 0.724 | 162.7 |
| PRO | waveform embedding | 7.960 | 0.104 | 0.781 | 0.722 | 156.7 |
| PRO | fused embedding | 3.629 | 0.104 | 0.781 | 0.708 | 167.7 |
| PRO | classifier logits | 2.353 | 0.104 | 0.788 | 0.706 | 167.7 |

PRO 相对 baseline 的变化：

| Layer | Shift delta | Flip delta | VT/VF AUROC delta | Multiclass Macro-F1 delta |
|---|---:|---:|---:|---:|
| waveform embedding | +3.134 | +0.003 | +0.001 | -0.023 |
| fused embedding | -2.113 | +0.003 | -0.014 | -0.025 |
| classifier logits | -0.080 | +0.003 | +0.012 | -0.019 |

解释：

- PRO 让 fused embedding 的扰动 shift 大幅下降，说明融合层更平滑。
- 但 waveform embedding 的 shift 反而变大，说明前端表征更敏感。
- 更重要的是，fused embedding 的 VT/VF readability 和 multiclass readability 都下降。
- 所以 PRO 的问题不是“完全没改变表征”，而是它改变了表征，但改变方向不等于更可靠的决策表征。

### 2.2 PRO 的 confident stable errors

3-seed duplicate-family：

| Variant | Group | Mean n | Fraction | Mean confidence | Mean flip rate | Mean final shift |
|---|---|---:|---:|---:|---:|---:|
| baseline | any error | 226.7 | 0.0549 | 0.832 | 0.293 | 5.917 |
| baseline | confident stable error | 64.0 | 0.0159 | 0.956 | 0.019 | 5.132 |
| baseline | confident stable VT/VF cross-error | 35.0 | 0.0090 | 0.943 | 0.029 | 4.349 |
| PRO | any error | 359.3 | 0.0852 | 0.889 | 0.344 | 4.277 |
| PRO | confident stable error | 52.7 | 0.0131 | 0.973 | 0.026 | 3.781 |
| PRO | confident stable VT/VF cross-error | 40.0 | 0.0101 | 0.966 | 0.025 | 3.644 |

解释：

- PRO 的 total errors 更多，any-error confidence 更高。
- PRO 的 confident stable VT/VF cross-errors 从 35.0 增到 40.0，而且 mean confidence 从 0.943 升到 0.966。
- 这很关键：PRO 不是只让模型“不确定地错”，而是会产生更多高置信、稳定的 VT/VF 边界错误。

### 2.3 Core seed42：为什么 prototype/contrastive/RISK-PRO++ 没有变成好模型

core seed42 fused embedding：

| Variant | Shift | Flip | VT/VF AUROC | Multiclass Macro-F1 | Probe VT/VF cross-errors |
|---|---:|---:|---:|---:|---:|
| teacher | 5.787 | 0.105 | 0.788 | 0.703 | 275 |
| PRO | 3.757 | 0.111 | 0.754 | 0.695 | 300 |
| contrastive | 2.718 | 0.057 | 0.731 | 0.698 | 303 |
| RISK-PRO++ | 2.909 | 0.064 | 0.781 | 0.692 | 307 |

stable/confident VT/VF errors：

| Variant | Confident stable VT/VF cross-errors | Fraction | Mean confidence | Mean flip rate |
|---|---:|---:|---:|---:|
| teacher | 2 | 0.0005 | 0.956 | 0.067 |
| PRO | 24 | 0.0057 | 0.983 | 0.047 |
| contrastive | 158 | 0.0377 | 0.990 | 0.033 |
| RISK-PRO++ | 53 | 0.0126 | 0.970 | 0.050 |

解释：

- Contrastive 和 RISK-PRO++ 都明显降低 shift / flip，也就是更稳定。
- 但它们的 probe VT/VF cross-errors 更高，confident stable VT/VF errors 也更多。
- 这说明表征强化目标可能把模型推向“更平滑、更自信、更稳定”，但不一定推向“更正确”。
- 特别是 contrastive：稳定性最好，但 confident stable VT/VF cross-errors 最多。这是最强的反例。

## 3. 为什么 PRO 不好：机制解释

PRO 类方法的假设是：

> 如果把类别中心拉开，或让 prototype 几何更好，分类应该更可靠。

但我们的结果显示这个假设不充分。

### 机制 1：中心距离不是局部边界可靠性

PRO 可能改善全局几何或降低 fused embedding shift，但 VT/VF 错误发生在局部边界区域。中心更远不代表边界附近样本更清楚。

### 机制 2：平滑可能压掉边界细节

PRO / contrastive / RISK-PRO++ 可以让表征更稳定，但稳定性可能来自压缩或规则化，而不是保留 VT/VF 细节。

### 机制 3：错误变得更自信

PRO 的 any-error confidence 从 baseline 的 0.832 升到 0.889；confident stable VT/VF cross-error confidence 从 0.943 升到 0.966。

这说明 PRO 有时不是让错误变得可识别，而是让错误更稳定、更自信。

### 机制 4：训练目标和主分类目标竞争

RISK-PRO++、contrastive 等目标同时约束几何、边界、risk、regularity，可能和 supervised CE 目标竞争。结果是几何或稳定性变好，但分类边界被放到更差的位置。

## 4. CNN-LSTM 和 PRO 的共同教训

| 方法 | 好处 | 问题 | 机制结论 |
|---|---|---|---|
| CNN-LSTM | 更平滑，扰动更稳定，10 seeds 下 VT/VF cross-errors 证据更好 | accuracy/ECE/total errors 不稳定 | 时序聚合是部分边界缓解，不是通用可靠性修复 |
| PRO | fused embedding 更稳定，几何被改变 | readability 下降，错误更自信，VT/VF stable errors 增加 | 几何/稳定性目标不等于边界可靠性目标 |
| Contrastive/RISK-PRO++ | 稳定性更强 | 可能产生大量 confident stable VT/VF errors | 过强表征规则化会把错决策固定下来 |

## 5. 对下一步模型设计的启示

这些结果说明，下一代模型不能只追求：

- 更稳定；
- 更大中心距离；
- 更强 prototype separation；
- 更低 embedding shift。

更合理的结构目标是：

> 保留 VT/VF boundary-readable information，同时避免把错误决策变成稳定自信错误。

因此 architecture upgrade 应该考虑：

1. **boundary-preserving skip connection**
   保留中层 waveform/CNN boundary features，不让后端 LSTM 或 prototype objective 压掉它。

2. **lightweight VT/VF auxiliary supervision**
   在关键中层加弱 VT/VF probe-style supervision，但权重要小，避免 RISK-PRO++ 式过正则。

3. **hierarchical head**
   先区分 SR vs ventricular，再区分 VT vs VF，避免三分类 head 把高风险边界当普通分类处理。

4. **anti-confident-error regularisation**
   对高置信错误或局部混叠样本加约束，目标不是单纯拉开中心，而是避免稳定自信错误。

## 6. 最终论文表述

建议论文里这样写：

> The additional mechanisms explain why stability alone is insufficient. CNN-LSTM reduces representation shift and some stable VT/VF boundary errors, but does not reliably improve global reliability metrics. PRO-style representation interventions reduce fused embedding instability, yet often reduce boundary readability and increase confident stable VT/VF errors. These results show that representation smoothing or centre separation is not equivalent to decision reliability; reliable ECG rhythm classification requires preserving boundary-readable features and explicitly controlling stable high-confidence boundary failures.

中文：

> 这些机制分析说明，稳定性本身不等于可靠性。CNN-LSTM 降低了表征扰动和部分稳定 VT/VF 边界错误，但没有稳定改善全局可靠性指标。PRO 类表征干预虽然降低了 fused embedding 不稳定性，却会降低边界可读性，并增加高置信稳定 VT/VF 错误。这说明表征平滑或中心分离不等于决策可靠；可靠 ECG 分类需要保留边界可读特征，并显式控制稳定高置信边界错误。
