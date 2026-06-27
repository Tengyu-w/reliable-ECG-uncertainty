# 方法小节草稿：Mechanism-aware Evidence-Routed Selective Classification

## 方法概述

我们提出一种机制感知的证据路由框架，用于 ECG SR/VT/VF 分类中的可靠性决策。与传统选择性分类方法不同，该方法不将不确定性压缩为单一 scalar risk，而是将分类失败分解为多个错误机制，并为每个机制建立对应的 evidence-specific reliability head 和 routing action。

该框架的目标不是替代主分类器，而是在主分类器给出单标签预测后，判断该预测是否应该：

- 保持单标签输出；
- 转换为 `{VT,VF}` prediction set；
- 进入机制特定 review route。

## 错误机制定义

给定主模型预测 `y_pred` 和真实标签 `y_true`，在 validation 阶段定义以下机制监督信号：

### 1. VT/VF Boundary Confusion

目标：

```text
is_vtvf_cross_error =
  (y_true = VT and y_pred = VF) or
  (y_true = VF and y_pred = VT)
```

该机制对应 VT 与 VF 的边界混淆。路由动作是输出 `{VT,VF}` 或进入边界复核。

### 2. SR-Ventricular Direction Error

目标：

```text
is_sr_ventricular_error =
  (y_true = SR and y_pred in {VT,VF}) or
  (y_true in {VT,VF} and y_pred = SR)
```

该机制对应正常节律与室性节律之间的方向性错误。路由动作是 `SR-ventricular review`。

### 3. Representation Conflict Error

目标：

```text
is_representation_conflict_error =
  is_error and (
    nearest_prototype != y_pred
    or KNN_prediction != y_pred
  )
```

该机制描述分类器输出与 embedding/prototype/KNN 表征证据之间的冲突。路由动作是 `representation review`。

### 4. Atypical Signal Error

目标：

```text
is_atypical_signal_error =
  is_error and (
    latent_cluster_distance is high
    or KNN_distance is high
    or prototype_distance is high
  )
```

该机制描述局部表征空间或节律特征上的 atypical / outlying 样本。路由动作是 `atypical review`。

### 5. Hidden Confident Failure

目标：

```text
is_hidden_confident_error =
  is_error and
  max_prob is high and
  entropy is low and
  KNN_entropy is low and
  second_model agrees
```

该机制对应高置信、低不确定性但仍然错误的隐藏失败模式。当前 10-seed 中 validation 正例不足，因此该机制仅作为分析项保留，不启用路由。

## 证据组

每个机制头使用不同证据组：

| evidence group | examples |
|---|---|
| softmax evidence | probability, entropy, MSP, margin, VT/VF ambiguity |
| representation evidence | prototype distance, nearest prototype, KNN entropy, KNN mixing |
| regularity evidence | spectral entropy, dominant frequency, autocorrelation, line length |
| latent cluster evidence | cluster distance, validation cluster error rate |
| model disagreement | second model prediction, second entropy, disagreement flags |
| historical diagnostics | risk target components, prior calibration scores |

在 seed42 中，可训练证据共 102 个。不同机制头使用不同子集：

| mechanism | n features |
|---|---:|
| VT/VF boundary | 59 |
| SR-ventricular | 61 |
| representation conflict | 59 |
| atypical signal | 54 |
| hidden confident | 66 |

此外，还有 test-only 历史诊断列用于 post-routing audit，但不进入训练，以避免测试集泄漏。

## 机制风险头

每个机制头使用 validation split 训练一个 binary risk head：

```text
r_m(x) = P(mechanism m is active | evidence_m(x))
```

其中 `m` 表示某个错误机制。每个机制头只接收与该机制相关的证据组，而不是全部证据。

10-seed 中，机制头表现为：

| mechanism | AUROC mean | AUPR mean |
|---|---:|---:|
| representation conflict | 0.9899 | 0.6857 |
| atypical signal | 0.9492 | 0.5606 |
| VT/VF boundary | 0.9539 | 0.3872 |
| SR-ventricular | 0.9125 | 0.5092 |
| hidden confident | 0.5000 | 0.0079 |

## Validation-optimized Profile Selection

v4 不使用固定机制权重，而是在 validation split 上从一组可解释 profile 中选择最优组合，例如：

- equal
- VT/VF boundary heavy
- representation heavy
- atypical heavy
- boundary + atypical
- representation + atypical
- single-mechanism profiles

对于每个预算 `b`，方法选择 validation utility 最高的 profile，并在 test split 上选择 top candidates，使 action rate 接近目标 budget。

## 路由动作

根据机制路由结果，输出动作包括：

| route | action |
|---|---|
| VT/VF boundary | `{VT,VF}` |
| SR-ventricular | review |
| representation conflict | review |
| atypical signal | review |
| hidden confident | reserved / not enabled |
| none | original single label |

## 与传统路由区别

传统 selective classification 通常是：

```text
uncertainty_score(x) > threshold -> reject
```

我们的方法是：

```text
identify likely error mechanism
-> select mechanism-specific evidence head
-> choose mechanism-specific routing action
```

因此它不仅判断“是否可靠”，也判断“不可靠的原因是什么”。

## 实验结果摘要

在 10-seed 内部验证中，v4 相比 learned total-risk layered routing：

| budget | delta VT/VF cross-error capture | delta all-error capture |
|---:|---:|---:|
| 0.05 | +0.0437 | +0.0448 |
| 0.10 | +0.0915 | +0.0168 |
| 0.20 | +0.0465 | +0.0223 |
| 0.30 | -0.0002 | +0.0223 |

这说明机制感知路由在低/中 action budget 下优于 scalar total-risk routing。

## 重要限制

更强 baseline 对比显示，`softmax_vtvf_ambiguity` 是非常强的 VT/VF-specific baseline：

- 10% budget 下，softmax VT/VF ambiguity 捕获 84.1% VT/VF cross-error。
- v4 捕获 59.7%。

因此当前结论应表述为：

- v4 证明机制感知框架优于 learned total-risk routing；
- 但对于纯 VT/VF boundary capture，仍需进一步强化 boundary expert；
- 下一版应将 strong VT/VF ambiguity baseline 纳入机制 profile 的核心目标。

## 贡献表述

英文：

> We propose a mechanism-aware evidence-routed selective classification framework for ECG reliability. Instead of relying on a scalar uncertainty score, the framework decomposes classification failures into distinct mechanisms and routes each mechanism through evidence-specific risk heads and tailored decision actions.

中文：

> 我们提出一种机制感知的证据路由选择性分类框架。该框架不依赖单一不确定性分数，而是将 ECG 分类失败分解为不同错误机制，并为每种机制建立特定证据头和对应路由动作。

## 下一步

v5 应重点解决：

1. 将 `softmax_vtvf_ambiguity` 明确纳入 VT/VF boundary expert。
2. validation utility 分成两个目标：VT/VF boundary capture 和 all-error capture。
3. 比较 v5 是否同时优于 total-risk routing 和 softmax VT/VF ambiguity baseline。
4. 为所有 seed 重新生成 conformal validation/test 输出，做完整 conformal baseline。
