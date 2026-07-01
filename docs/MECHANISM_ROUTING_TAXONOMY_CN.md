# ECG 机制路由层级与公平比较规则

本文件只保留最终规则，用于避免将不同层级的对象混在一起比较。

## 1. 四个层级

### A. 问题发现层

这一层回答模型为什么不可靠，变量多数是观测证据：

- embedding / prototype / KNN：prototype distance、nearest prototype、KNN density、KNN entropy、VT/VF mixing、latent cluster distance；
- softmax / logit：confidence、entropy、margin、VT/VF ambiguity；
- ECG waveform：regularity、spectral entropy、line length、dominant frequency、autocorrelation；
- wavelet / time-frequency：Mexican-hat、slope、oscillation、多尺度能量比例、fine-to-coarse energy；
- validity / domain：validity gate、boundary score、gate x boundary；
- failure types：VT/VF cross-error、SR vs ventricular confusion、representation conflict、atypical signal、hidden confident error。

这些变量用于诊断机制，不直接等于完整路由策略。

### B. Evidence Head / Mechanism Evidence 层

这一层把发现的机制转成可评分信号，例如：

- `vtvf_boundary`;
- `wavelet_boundary`;
- `validity_boundary`;
- `sr_ventricular`;
- `representation_conflict`;
- `atypical_signal`;
- `hidden_confident`.

Evidence head 只回答“这个样本为什么值得怀疑”，不是完整 router。

### C. Complete Routing Policy 层

只有完整 routing policy 才能和 V5D 或其他完整 router 公平比较。完整 policy 必须定义：

- 使用哪些 evidence heads；
- 每个 evidence head 的权重；
- 每个机制允许处理哪些 candidate samples；
- review / recover budget 如何分配；
- 被选中样本进入什么 recover/review action；
- 未被选中样本是否继续 single-label 自动输出。

例如 `complete_router_profile_wavelet_boundary_heavy` 表示 wavelet-boundary-heavy evidence profile 已经被 candidate mask、budget selection 和 recover action mapping 包装成完整路由策略。

### D. Recovery Action 层

Recovery action 是完整 router 内部的动作分支，例如：

- `vtvf_boundary_set`;
- `sr_ventricular_review`;
- `representation_review`;
- `atypical_review`;
- `hidden_failure_review`;
- `single_label`.

Recovery action 可用于 route composition 和 error migration 分析，但不能单独作为主性能比较对象。

## 2. 公平比较规则

| 比较对象 | 是否公平 | 指标 |
| --- | --- | --- |
| evidence head vs evidence head | 是 | AUROC、AUPR、target alignment |
| model vs model | 是 | accuracy、macro-F1、ECE、VT/VF errors、total errors |
| complete router vs complete router | 是 | VT/VF capture、all-error capture、residual risk、budget、OOD-style stability |
| recovery action vs recovery action | 条件性可以 | route composition、error migration |
| evidence head vs complete router | 否 | 层级不同 |
| classifier vs recover/router policy | 否 | 层级不同 |

## 3. 本文允许的主比较

### Model-only

用于比较分类模型：

```text
CNN vs CNN-LSTM vs PRO vs ProRisk vs constrained models
```

### Evidence-head-only

用于比较机制信号：

```text
entropy vs KNN vs prototype ambiguity vs wavelet risk vs validity boundary
```

### Router-only

用于比较完整路由机制：

```text
V5D vs optimized mechanism router vs complete wavelet-heavy router
```

### Fixed-router downstream

用于比较同一路由器下不同模型输入：

```text
V5D(model=A) vs V5D(model=B)
```

## 4. 可干预变量、机制变量与 Outcome

### 可干预变量

- `do(training_constraint = c)`;
- `do(policy_profile = p)`;
- `do(weight_m = w)`;
- `do(include_evidence_family = e)`;
- `do(review_budget = b)`;
- `do(candidate_mask_m = mask)`;
- `do(residual_reserve = r)`.

### 机制变量

- embedding separation；
- KNN local purity / VT/VF mixing；
- prototype VT/VF ambiguity；
- softmax VT/VF ambiguity；
- validity gate / boundary score；
- wavelet VT/VF boundary risk；
- regularity waveform features；
- mechanism-head risk scores；
- explanation alignment scores。

### Outcome

模型层 outcome：

- accuracy；
- macro-F1；
- ECE；
- VT/VF cross-errors；
- total errors；
- error migration penalty。

路由层 outcome：

- VT/VF cross-error capture；
- all-error capture；
- automatic unresolved VT/VF error rate；
- automatic unresolved all-error rate；
- review / recover budget；
- route composition stability。

## 5. 写作口径

推荐写法：

> Evidence heads are evaluated as mechanism signals. Complete routing policies are evaluated as decision policies. Model classifiers are evaluated separately from routing/recovery policies.

中文：

> 证据头作为机制信号评估，完整路由策略作为决策策略评估，分类模型与路由/恢复策略分层比较。
