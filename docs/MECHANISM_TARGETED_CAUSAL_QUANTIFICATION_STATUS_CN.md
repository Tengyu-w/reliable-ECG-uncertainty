# 机制定向因果式量化当前状态

更新时间：2026-06-30

## 1. 这一版在做什么

这一版的目标不是再提出一个新的名字，也不是把模型分类结果和 V5D 路由结果混在一起比较。它专门回答导师提出的问题：

```text
为什么前期分析出来的表征层、波形、boundary、prototype、KNN、gate 等机制，
有理由被认为会影响模型结果？
```

因此当前实验采用：

```text
do(training intervention)
  -> mechanism variable change
  -> model outcome change
```

也就是同一个 seed 下，用 baseline 作为对照，只改变一个或一组训练约束，观察机制变量和 outcome 是否同步改变。

## 2. 可干预变量、机制变量、outcome

### 可干预变量

当前真正被干预的是训练约束或训练权重，例如：

| candidate | intervention | intended target |
| --- | --- | --- |
| `proto_center_only` | prototype center loss | within-class compactness |
| `proto_margin_only` | prototype VT/VF margin loss | VT/VF prototype ambiguity |
| `contrastive_vtvf_light` | boundary-aware contrastive loss | KNN local purity / VT-VF mixing |
| `boundary050` / `boundary075` | boundary CE weight | softmax VT/VF boundary ambiguity |
| `gate_boundary_joint` | risk gate and boundary-head alignment | validity gate-boundary alignment |
| `regularity_aux_medium` | regularity auxiliary loss | ECG waveform regularity |
| `boundary075_prototype` | boundary + prototype joint constraint | known joint mechanism chain |

这些是可以写成 `do(...)` 的变量。

### 机制变量

这些变量不是直接手动设置的，而是模型训练后被测量出来的中介/机制证据：

| family | variables |
| --- | --- |
| embedding geometry | `silhouette_full`, `davies_bouldin_full`, `sr_vt_norm_dist`, `sr_vf_norm_dist`, `vt_vf_norm_dist` |
| KNN neighborhood | `local_purity_k_mean`, `knn_label_entropy_mean`, `knn_vtvf_mix_ventricular_mean`, `knn_distance_mean` |
| prototype ambiguity | `prototype_vtvf_ambiguity_ventricular_mean`, `prototype_vtvf_ambiguity_auroc` |
| softmax ambiguity | `entropy_mean`, `prob_margin_mean`, `softmax_vtvf_ambiguity_ventricular_mean`, `low_margin_any_error_auroc` |
| validity/boundary | `validity_gate_mean`, `boundary_score_mean`, `gate_x_boundary_any_error_auroc`, `gate_x_boundary_vtvf_cross_auroc` |

这些变量回答的是“训练干预到底改变了模型内部哪一部分”。

### Outcome

模型层 outcome 只和模型层模型比较，不和 V5D 路由 policy 混比：

```text
accuracy
macro_f1
ece
vtvf_cross_errors
total_errors
error_migration_penalty
```

## 3. 当前已经完成的代码

新增/更新了两个入口：

```text
python -m src.run_mechanism_targeted_causal_ablation
python -m src.run_causal_mechanism_quantification
```

第一个脚本负责训练 targeted candidates，并输出：

```text
mechanism_targeted_ablation_manifest_*.csv
mechanism_targeted_ablation_run_level.csv
mechanism_targeted_ablation_paired_effects.csv
mechanism_targeted_ablation_summary.csv
mechanism_targeted_ablation_by_mechanism.csv
```

第二个脚本负责读取 manifest 和 run directories，计算 32 个机制变量，并输出：

```text
run_level_mechanism_outcome_table.csv
paired_candidate_seed_mechanism_outcome_deltas.csv
intervention_to_mechanism_effects.csv
intervention_to_outcome_effects.csv
mechanism_to_outcome_association.csv
mediation_or_path_effect_summary.csv
causal_mechanism_variable_dictionary.csv
```

第三个脚本负责把量化结果整理成 thesis-facing verdict table：

```text
python -m src.summarize_mechanism_targeted_causal_quantification \
  --quant-dir results/mechanism_targeted_causal_quantification_full_20260630
```

它会输出：

```text
mechanism_targeted_verdict_table.csv
top_mechanism_outcome_associations.csv
mechanism_targeted_causal_quantification_summary.md
```

这个脚本采用保守判断：`0` 变化只算 no effect，不算 improvement。

另外已经加入自动收尾脚本：

```text
python -m src.finalize_mechanism_targeted_causal_pipeline \
  --run-dir results/mechanism_targeted_causal_ablation_full_20260630 \
  --quant-dir results/mechanism_targeted_causal_quantification_full_20260630 \
  --wait \
  --poll-seconds 300 \
  --max-wait-minutes 720
```

它会每 5 分钟检查一次 manifest。如果 full run 未完成，只写入当前状态；如果达到 `33/33` completed rows，则自动执行：

```text
python -m src.run_causal_mechanism_quantification ...
python -m src.summarize_mechanism_targeted_causal_quantification ...
```

当前后台 watcher 已经启动，日志在：

```text
results/mechanism_targeted_causal_quantification_full_20260630/finalizer_stdout.log
results/mechanism_targeted_causal_quantification_full_20260630/finalizer_stderr.log
```

## 4. Smoke run 结果

smoke run 已完成，路径为：

```text
results/mechanism_targeted_causal_ablation_smoke_20260630/
results/mechanism_targeted_causal_quantification_smoke_20260630/
```

smoke 只用了 1 个 seed 和 1 个 epoch，因此只能证明流程可运行，不能作为论文结论。

确认结果：

| item | status |
| --- | --- |
| baseline training | passed |
| risk target generation | passed |
| prototype candidate training | passed |
| boundary candidate training | passed |
| mechanism-variable extraction | passed |
| paired intervention delta table | passed |
| empty association handling for too-few seeds | passed |

smoke 量化输出：

```text
n_run_level_rows = 3
n_paired_delta_rows = 2
n_seeds = 1
n_candidates = 3
n_mechanism_variables = 32
n_association_rows = 0
```

`n_association_rows = 0` 是合理的，因为 1 个 seed 不足以计算稳定的 mechanism-outcome association。

smoke verdict summary 也已经通过格式验证：

| candidate | target mechanism | smoke verdict meaning |
| --- | --- | --- |
| `boundary075` | softmax boundary ambiguity | outcome 有变化，但目标 softmax 机制未朝预期方向改变；smoke 不作结论 |
| `proto_margin_only` | prototype VT/VF ambiguity | 1-epoch smoke 基本无变化；正确标记为 no effect |

## 5. 正式 full run 当前状态

正式实验已经启动：

```text
python -m src.run_mechanism_targeted_causal_ablation \
  --seeds 42 43 44 \
  --epochs 30 \
  --out results/mechanism_targeted_causal_ablation_full_20260630
```

该 full run 的设计是：

```text
11 candidates x 3 seeds = 33 training runs
```

当前还在运行，不能提前下结论。手动收尾命令为：

```text
python -m src.run_causal_mechanism_quantification \
  --search-dir results/mechanism_targeted_causal_ablation_full_20260630 \
  --out results/mechanism_targeted_causal_quantification_full_20260630 \
  --k 15
```

这样会生成真正可用于论文的 same-seed paired mechanism evidence。

如果后台 watcher 正常运行，则无需手动执行；它会在 full run 完成后自动执行上述收尾步骤。

## 6. 最终要判断什么

full run 完成后，不只看哪个 candidate 分数最高，而是逐个机制判断：

| question | meaning |
| --- | --- |
| Target mechanism changed? | 干预是否真的改变了它声称要改变的机制变量 |
| Outcome improved? | accuracy、macro-F1、ECE、VT/VF cross-error 等是否朝好方向变 |
| Seed consistency? | 3 个 seed 是否方向一致，而不是偶然 |
| Side effect? | 是否改善一个指标但恶化校准、迁移错误或 VT/VF 边界 |
| Verdict | core mechanism / auxiliary mechanism / negative result |

这才是这一版因果推断加多目标优化在模型层的正确使用位置。
