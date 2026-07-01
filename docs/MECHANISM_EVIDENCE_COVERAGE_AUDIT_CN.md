# 机制变量证据覆盖审计

日期：2026-06-30

## 1. 结论

不是所有早期分析过的变量都已经完成同等级的因果式验证。当前更准确的结论是：

> 核心机制家族已经被纳入定量证据链；其中模型层的 embedding/KNN/prototype、softmax/confidence、validity/boundary 变量完成了 paired do-intervention proxy；机制库层的 wavelet、regularity、mechanism head、explanation alignment 完成了信号强度和固定预算 outcome 证据化。但早期所有原始探索变量并不是每一个都完成了独立、同 seed、同等级的因果验证。

因此论文中应区分：

1. **已完成因果式 paired 验证的机制变量**；
2. **已完成机制库定量证据化的 evidence family**；
3. **只作为诊断、辅助解释或负结果保留的变量**；
4. **不能写成已验证主机制的早期探索变量**。

## 2. 模型层：已完成 paired do-intervention proxy

对应结果目录：

```text
results/causal_mechanism_quantification_20260630/
```

模型层覆盖了 32 个机制变量，分为三组：

| 机制家族 | 已覆盖变量数 | 代表变量 | 证据类型 |
| --- | ---: | --- | --- |
| representation / KNN / prototype | 17 | `silhouette_full`, `local_purity_k_mean`, `knn_vtvf_mix_ventricular_mean`, `prototype_vtvf_ambiguity_ventricular_mean` | same-seed intervention delta + mechanism-outcome association |
| softmax / confidence / boundary ambiguity | 7 | `entropy_mean`, `prob_margin_mean`, `softmax_vtvf_ambiguity_ventricular_mean`, `low_margin_any_error_auroc` | same-seed intervention delta + mechanism-outcome association |
| validity / boundary | 8 | `validity_gate_mean`, `boundary_score_mean`, `gate_x_boundary_any_error_auroc`, `gate_x_boundary_vtvf_cross_auroc` | same-seed intervention delta + mechanism-outcome association |

模型层 outcome 覆盖：

```text
accuracy
macro_f1
ece
vtvf_cross_errors
total_errors
error_migration_penalty
```

最强的模型层证据是 `boundary075_prototype`：它在 3 个 paired seeds 上 6 个 outcome 全部朝好方向变化，同时伴随 embedding separation、KNN purity、prototype ambiguity、softmax ambiguity 和 validity signal 的机制变化。

## 3. 机制库层：已完成信号强度和 routing outcome 证据化

对应结果目录：

```text
results/mechanism_library_evidence_chain_20260630/
```

机制库覆盖情况：

| 机制家族 | 已覆盖机制数 | 证据行数 | 证据类型 |
| --- | ---: | ---: | --- |
| wavelet time-frequency | 2 | 4 | 10-seed AUROC/AUPR + fixed-budget routing outcome |
| regularity waveform | 16 | 98 | feature-family ablation + 3-seed routing/review outcome |
| mechanism-specific error head | 5 | 10 | 10-seed mechanism head AUROC/AUPR |
| explanation reliability | 6 | 288 | explanation-target alignment evidence |

机制库层 policy outcome 覆盖：

```text
all_error_addressed
vtvf_cross_error_addressed
automatic_unresolved_error_rate
automatic_unresolved_vtvf_cross_error_rate
```

这说明 wavelet、regularity、mechanism heads 和 explanation alignment 不是只被口头分析，而是已经被整理成机制库证据。但这类证据主要证明“机制信号是否有用”和“固定预算路由是否有效”，不等同于模型层训练干预的 paired causal proof。

## 4. 已验证程度最高的核心机制

论文主机制建议优先写：

| 机制 | 当前证据等级 | 写作建议 |
| --- | --- | --- |
| embedding separation / silhouette | 强 | 可作为模型层 mediator |
| KNN local purity / KNN mixing | 强 | 可作为局部表征可靠性机制 |
| prototype VT/VF ambiguity | 强 | 可作为 VT/VF 边界机制 |
| softmax VT/VF ambiguity / entropy / margin | 中强 | 可作为分类置信边界机制 |
| validity gate x boundary score | 中强 | 可作为 reliability signal，但不要写成 gate 天然正确 |
| wavelet VT/VF boundary risk | 强 | 可作为路由层 ECG 时频证据 |
| representation_conflict / vtvf_boundary heads | 强 | 可作为机制风险头证据 |
| boundary / representation explanation | 强 | 可作为 explanation alignment 证据 |

## 5. 需要谨慎写的机制

| 机制 | 原因 | 建议 |
| --- | --- | --- |
| hidden_confident mechanism head | 机制头 AUROC 接近 0.5 | 保留为负结果，不写成有效主机制 |
| regularity explanation | 对 atypical signal 有辅助意义，但解释对齐弱于 boundary/representation | 写成辅助机制 |
| regularity feature injection | 有波形意义，但模型层效果不是所有目标最优 | 写成 auxiliary waveform evidence |
| early gate-target variants | 早期表现不稳定，部分结果弱于 CNN | 不能写成 gate 本身已被证明正确 |
| 单个 wavelet / validity head | 只是 evidence head，不是完整 router | 不能直接和 V5D 完整路由比较 |

## 6. 论文中最稳妥的表述

建议写：

> We did not assume that every diagnostic variable was causally valid. Instead, we organized the previously identified ECG representation, waveform, uncertainty, and routing variables into a tiered evidence framework. Core model-layer mechanisms, including embedding geometry, KNN local purity, prototype ambiguity, softmax ambiguity, and validity-boundary signals, were evaluated using paired same-seed intervention deltas and mechanism-outcome associations. Route-level evidence families, including wavelet time-frequency risk, regularity descriptors, mechanism-specific risk heads, and explanation alignment, were evaluated through signal-quality metrics and fixed-budget routing outcomes. This distinction prevents diagnostic observations from being overstated as causal mechanisms.

中文：

> 本研究并未假设所有诊断变量都天然具有因果意义，而是将前期发现的表征、波形、不确定性和路由变量组织成分层证据框架。模型层核心机制通过 same-seed paired intervention delta 和 mechanism-outcome association 进行因果式验证；路由层机制证据通过 AUROC/AUPR、解释对齐和固定预算 routing outcome 进行证据化。该分层设计避免了将所有诊断发现直接夸大为因果机制。

