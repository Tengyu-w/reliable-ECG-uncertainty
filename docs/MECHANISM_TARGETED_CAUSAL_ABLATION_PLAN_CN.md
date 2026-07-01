# 机制靶向因果消融实验计划

日期：2026-06-30

## 1. 目标

上一轮 `boundary075_prototype` 已经证明组合干预有效：

```text
do(boundary075 + prototype)
  -> embedding/KNN/prototype/softmax/validity 机制变化
  -> model outcome 改善
```

但它仍然是组合干预。下一步要回答更细的问题：

> 每一个核心机制到底是否真的影响模型？如果影响，它通过哪条机制变量链影响 outcome？

因此下一步不是再随便训练一个“更复杂模型”，而是做机制靶向因果消融：

```text
机制假设
  -> 专门设计一个可干预训练变量
  -> 检查目标机制变量是否变化
  -> 检查最终 outcome 是否变化
  -> 判断该机制是否应保留为主机制、辅助机制或负结果
```

## 2. 总体实验原则

| 原则 | 说明 |
| --- | --- |
| 同 backbone | 默认使用 `reliability_gated_fusion`，避免架构差异混淆机制解释 |
| same-seed paired | 每个候选与同 seed baseline 比较，优先 seeds 42/43/44 |
| 小步干预 | 每次只主攻一个机制，避免重新变成大组合 |
| 机制变量优先 | 不只看 accuracy，也看 target mechanism 是否真的改变 |
| 多目标 guard | 任何候选都必须检查 ECE、VT/VF cross-error、total error、migration |
| 先 smoke 再 full | 先 1 seed / 少 epoch smoke，确认脚本和变量输出，再跑 3 seeds |

## 3. 第一轮优先机制

第一轮只选 5 个核心机制，原因是它们和 `boundary075_prototype` 的成功链条最接近。

| 编号 | 机制 | 为什么优先 |
| --- | --- | --- |
| M1 | prototype VT/VF ambiguity | 直接来自 prototype 分析，也是当前组合干预的核心 |
| M2 | KNN local purity / KNN mixing | 老师特别关心“邻域纯净度是否真的影响模型” |
| M3 | softmax VT/VF ambiguity / margin | 对应 boundary CE 和分类概率层边界 |
| M4 | validity gate x boundary alignment | 对应 gate 是否真的可靠，而不是 attention-like overfit |
| M5 | ECG waveform regularity | 对应波形结构、节律、频率和复杂度分析 |

## 4. M1：Prototype VT/VF ambiguity

### 机制假设

VT/VF 错误部分来自 embedding 原型层面的边界歧义。若 VT/VF 样本同时接近两个 prototype，分类头更容易发生 VT/VF cross-error。

### 干预设计

| 候选 | 训练干预 | 目的 |
| --- | --- | --- |
| `proto_center_only` | `prototype_center_weight=0.02` | 只压缩同类 embedding |
| `proto_margin_only` | `prototype_margin_weight=0.05`, `prototype_vtvf_margin=1.0` | 只拉开 VT/VF 原型 |
| `proto_center_margin` | center + margin | 复现 prototype guard 的核心 |
| `boundary075_proto_margin` | boundary 0.75 + margin only | 测 boundary 与 VT/VF margin 的交互 |

### 目标机制变量

```text
prototype_vtvf_ambiguity_ventricular_mean
prototype_vtvf_ambiguity_auroc
vt_vf_norm_dist
proto_margin
nearest_proto_is_pred
```

### 目标 outcome

```text
VT/VF cross-errors
macro-F1
total errors
error migration penalty
```

### 判断标准

若 prototype ambiguity 下降，并且 VT/VF cross-error 与 migration 不恶化，则 prototype 是主机制。若 ambiguity 下降但 total error 或 migration 恶化，则 prototype 是有用但需要 guard 的机制。

## 5. M2：KNN local purity / KNN mixing

### 机制假设

模型错误不仅来自全局中心距离，也来自局部邻域混杂。若一个样本附近同时存在 VT 和 VF，或者邻域标签熵高，模型更容易发生边界错误和迁移错误。

### 干预设计

| 候选 | 训练干预 | 目的 |
| --- | --- | --- |
| `contrastive_vtvf_light` | `contrastive_weight=0.02`, VT/VF negatives 加权 | 直接提高局部邻域可分性 |
| `contrastive_vtvf_medium` | `contrastive_weight=0.05` | 检查剂量效应 |
| `embedding_consistency_light` | `embedding_consistency_weight=0.01` | 稳定 embedding 邻域 |
| `prototype_plus_contrastive` | prototype center/margin + contrastive light | 检查 prototype 与 KNN purity 是否互补 |

### 目标机制变量

```text
local_purity_k_mean
knn_label_entropy_mean
knn_vtvf_mix_ventricular_mean
knn_vtvf_mix_auroc
neighbor_jaccard_mean
```

### 目标 outcome

```text
accuracy
macro-F1
ECE
VT/VF cross-errors
error migration penalty
```

### 判断标准

若 local purity 提高、KNN entropy 降低，并且 total errors 与 migration 下降，则 KNN purity 可写成核心机制。若 purity 提高但 outcome 不变，说明 KNN purity 只是诊断变量，不应写成主因。

## 6. M3：Softmax boundary ambiguity / margin

### 机制假设

VT/VF 错误部分来自分类概率层面的边界不清楚。若 VT/VF 概率接近或 margin 低，模型容易发生交叉错误。

### 干预设计

| 候选 | 训练干预 | 目的 |
| --- | --- | --- |
| `boundary025` | `boundary_ce_weight=0.25` | 低剂量边界加权 |
| `boundary050` | `boundary_ce_weight=0.50` | 中低剂量 |
| `boundary075` | `boundary_ce_weight=0.75` | 当前成功组合中的剂量 |
| `boundary100` | `boundary_ce_weight=1.00` | 高剂量，检查副作用 |
| `risk_entropy_light` | `risk_entropy_weight=0.05` | 校准 entropy 与 risk target |
| `anti_confident_light` | `anti_confident_risk_weight=0.02` | 抑制高风险高置信 |

### 目标机制变量

```text
softmax_vtvf_ambiguity_ventricular_mean
softmax_vtvf_ambiguity_auroc
prob_margin_mean
entropy_mean
entropy_any_error_auroc
low_margin_any_error_auroc
```

### 目标 outcome

```text
ECE
VT/VF cross-errors
total errors
macro-F1
```

### 判断标准

若 softmax ambiguity 下降、margin 增加、ECE 下降，并且 VT/VF cross-error 不恶化，则 boundary/softmax 机制成立。若 boundary100 改善某个指标但制造 migration，则支持使用 0.75 作为 Pareto 剂量。

## 7. M4：Validity gate x boundary alignment

### 机制假设

gate 本身不是天然正确。只有当 gate 与 boundary / risk 对齐，且能识别真实错误时，才是可靠机制。

### 干预设计

| 候选 | 训练干预 | 目的 |
| --- | --- | --- |
| `gate_target_only` | `gate_target_weight>0` | 检查单独 gate target 是否有用 |
| `risk_gate_light` | `risk_gate_weight=0.05` | 让 gate 对齐 risk target |
| `risk_boundary_light` | `risk_boundary_weight=0.05` | 让 boundary head 对齐 risk target |
| `gate_boundary_joint` | risk gate + risk boundary | 检查 gate-boundary 联合机制 |
| `boundary075_gate_joint` | boundary 0.75 + gate-boundary joint | 检查当前 boundary 成功是否能被 gate 进一步解释 |

### 目标机制变量

```text
validity_gate_mean
boundary_score_mean
gate_x_boundary_mean
validity_gate_any_error_auroc
boundary_score_any_error_auroc
gate_x_boundary_any_error_auroc
gate_x_boundary_vtvf_cross_auroc
```

### 目标 outcome

```text
ECE
any-error AUROC
VT/VF cross-errors
total errors
review capture under fixed budget
```

### 判断标准

若 gate_x_boundary AUROC 提高且 outcome 改善，gate 可作为 reliability mechanism。若 gate 指标提高但分类或 calibration 恶化，则 gate 只能作为辅助路由信号，不能作为主模型机制。

## 8. M5：ECG waveform regularity

### 机制假设

部分错误与 ECG 节律、频谱结构、周期性和复杂度有关。regularity 变量应解释模型在哪些波形结构下更不可靠。

### 干预设计

| 候选 | 训练干预 | 目的 |
| --- | --- | --- |
| `regularity_aux_light` | `regularity_aux_weight=0.01` | 轻量保留波形规则性 |
| `regularity_aux_medium` | `regularity_aux_weight=0.02` | 当前组合附近剂量 |
| `regularity_frequency_only` | 只预测 frequency group | 检查频率机制 |
| `regularity_periodicity_only` | 只预测 periodicity group | 检查周期性机制 |
| `regularity_complexity_only` | 只预测 complexity group | 检查复杂度机制 |

### 目标机制变量

```text
spectral_entropy
dominant_frequency
dominant_frequency_concentration
spectral_centroid
spectral_bandwidth
autocorr_peak
autocorr_peak_lag_s
zero_crossing_rate
line_length
regularity_atypicality_explanation
```

### 目标 outcome

```text
macro-F1
SR/VT/VF confusion
atypical_signal_error
VT/VF cross-errors
OOD/corruption robustness
```

### 判断标准

若 regularity aux 改善 waveform-aligned errors 或 OOD robustness，但不改善主分类指标，则写成辅助 ECG waveform mechanism。若它导致 macro-F1 或 migration 恶化，则不能作为主模型约束。

## 9. 第一轮实验矩阵建议

为了控制训练成本，第一轮不要把上面全部跑完。建议先跑 10 个候选：

| 组 | 候选 | 目的 |
| --- | --- | --- |
| baseline | `baseline` | same-seed 对照 |
| M1 | `proto_center_only` | 检查同类 compactness |
| M1 | `proto_margin_only` | 检查 VT/VF margin |
| M1 | `proto_center_margin` | 检查 prototype 组合 |
| M2 | `contrastive_vtvf_light` | 检查 KNN purity |
| M2 | `prototype_plus_contrastive` | 检查 prototype + KNN 联合作用 |
| M3 | `boundary050` | 检查 boundary 低剂量 |
| M3 | `boundary075` | 检查当前成功剂量 |
| M4 | `gate_boundary_joint` | 检查 gate 是否可作为可靠机制 |
| M5 | `regularity_aux_medium` | 检查波形 regularity 是否辅助 |
| combo | `boundary075_prototype` | 当前成功组合复现 |

第一轮建议：

```text
seeds = 42, 43, 44
epochs = 30
backbone = reliability_gated_fusion
split_grouping = record 或 duplicate_family，保持与前一轮一致
```

对应脚本已经建立：

```text
python -m src.run_mechanism_targeted_causal_ablation
```

已验证 dry-run 命令：

```text
python -m src.run_mechanism_targeted_causal_ablation \
  --dry-run \
  --seeds 42 \
  --epochs 1 \
  --out results/mechanism_targeted_causal_ablation_dryrun_20260630
```

已验证 smoke 训练命令：

```text
python -m src.run_mechanism_targeted_causal_ablation \
  --seeds 42 \
  --epochs 1 \
  --max-windows-per-record 2 \
  --candidates baseline proto_margin_only boundary075 \
  --out results/mechanism_targeted_causal_ablation_smoke_20260630
```

Smoke 结果只用于验证流程，不用于论文结论。该 smoke 已确认：

| 检查项 | 状态 |
| --- | --- |
| baseline 训练 | 通过 |
| risk target 生成 | 通过 |
| risk-target candidate 训练 | 通过 |
| prototype candidate 训练 | 通过 |
| paired effect summary | 通过 |
| by-mechanism summary | 通过 |

正式 3-seed 运行命令建议为：

```text
python -m src.run_mechanism_targeted_causal_ablation \
  --seeds 42 43 44 \
  --epochs 30 \
  --out results/mechanism_targeted_causal_ablation_full_20260630
```

该脚本会将每个候选标注为 `target_mechanism`，例如：

| candidate | target mechanism |
| --- | --- |
| `proto_center_only` | prototype compactness |
| `proto_margin_only` | prototype VT/VF ambiguity |
| `contrastive_vtvf_light` | KNN local purity |
| `boundary050` / `boundary075` | softmax boundary ambiguity |
| `gate_boundary_joint` | validity gate-boundary alignment |
| `regularity_aux_medium` | waveform regularity |
| `boundary075_prototype` | known joint candidate |

## 10. 输出表必须包含什么

每个候选都必须输出四张表：

| 表 | 内容 |
| --- | --- |
| run-level metrics | 每个 seed 的 accuracy、macro-F1、ECE、错误矩阵 |
| mechanism deltas | 每个机制变量相对 same-seed baseline 的变化 |
| outcome deltas | 每个 outcome 相对 same-seed baseline 的变化 |
| mechanism-outcome association | 机制变化与 outcome 变化的 Spearman/Pearson 关联 |

当前脚本第一阶段会自动输出：

```text
mechanism_targeted_ablation_manifest_*.csv
mechanism_targeted_ablation_config.json
mechanism_targeted_ablation_report.json
mechanism_targeted_ablation_run_level.csv
mechanism_targeted_ablation_paired_effects.csv
mechanism_targeted_ablation_summary.csv
mechanism_targeted_ablation_by_mechanism.csv
```

训练完成后，再用机制量化脚本读取这些 run dirs，生成机制变量 delta 与 mechanism-outcome association。若需要，下一步可以把 `src/run_causal_mechanism_quantification.py` 泛化为可读取 `mechanism_targeted_ablation_manifest_*.csv`。

最终不是只选分数最高的模型，而是判断：

```text
这个机制是否真的被干预改变？
这个机制变化是否和 outcome 改善一致？
这个机制是否有副作用？
这个机制应写成主机制、辅助机制，还是负结果？
```

## 11. 预期论文结果形态

论文中最终应形成以下表格：

| Mechanism | Intervention | Target mechanism changed? | Outcome improved? | Verdict |
| --- | --- | --- | --- | --- |
| Prototype ambiguity | prototype margin | yes/no | yes/no | core / auxiliary / reject |
| KNN purity | contrastive / neighborhood proxy | yes/no | yes/no | core / auxiliary / reject |
| Softmax ambiguity | boundary CE / entropy alignment | yes/no | yes/no | core / auxiliary / reject |
| Gate alignment | risk gate / boundary head | yes/no | yes/no | core / auxiliary / reject |
| Regularity | regularity aux | yes/no | yes/no | core / auxiliary / reject |

这就是老师要的“为什么这些分析出来的部分能够影响模型”的直接证据。
