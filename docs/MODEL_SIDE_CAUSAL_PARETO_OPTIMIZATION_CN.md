# 模型层因果-Pareto优化：比较层级修正版

对应脚本：

```text
python -m src.model_side_causal_pareto_inventory
```

输出目录：

```text
results/model_side_causal_pareto_inventory_20260629/
```

## 1. 这次修正解决什么问题

本项目里有三类对象，不能混在一起比较：

1. **分类模型**：例如 CNN、GatedFusion、PRO、CNN-TCN-Validity、CNN-Wavelet-TCN。
2. **证据头/证据信号**：例如 validity gate、wavelet VT/VF boundary risk、softmax ambiguity、KNN mixing density。
3. **路由/recover机制**：例如 V5D stage1-stage2，或者后续的因果-Pareto routing policy。

因此，不能写成“某个分类模型 vs V5D”。这两个不是同一层级的对象。正确写法是：

| 层级 | 正确比较对象 | Outcome |
| --- | --- | --- |
| model-only | `model(A)` vs `model(B)` | accuracy, macro-F1, ECE, VT/VF cross-errors, total errors, error migration |
| evidence-head-only | `evidence_head(A)` vs `evidence_head(B)` | any-error AUROC, VT/VF boundary AUROC, risk calibration, ECG-shift stability |
| fixed-router downstream | `V5D(model=A)` vs `V5D(model=B)` | capture, residual risk, review budget, recover action distribution |

也就是说，V5D 只能作为**固定下游路由器**来测试不同模型输出是否更适配它，不能作为分类模型的直接对照组。

## 2. 新增输出文件

脚本现在输出四类表：

```text
model_side_intervention_inventory.csv
model_only_intervention_inventory.csv
evidence_head_inventory.csv
fixed_router_downstream_design.csv
comparison_layer_dictionary.csv
model_side_next_model_only_experiment_plan.csv
```

其中 `model_side_intervention_inventory.csv` 只是资料索引，保留全部已分析证据；正式比较应使用后面三个分层表。

## 3. 可干预变量与不可干预变量

模型层可干预变量包括：

| Variable | Meaning |
| --- | --- |
| `do(model_family)` | 改变 ECG 分类模型结构，例如 CNN、TCN、GatedFusion |
| `do(add_prototype_objective)` | 改变 embedding 几何，例如 prototype center loss、VT/VF margin |
| `do(add_risk_objective)` | 加入风险目标，例如 risk entropy alignment |
| `do(add_validity_branch)` | 加入 validity/domain branch |
| `do(add_wavelet_boundary_branch)` | 加入 wavelet time-frequency boundary branch |
| `do(add_regularity_auxiliary)` | 加入 rhythm/morphology regularity auxiliary |

不可干预变量包括原始 ECG morphology/rhythm、真实标签、record-level split、duplicate-family结构、训练/测试划分。它们可以作为 context 或 design constraint，但不能被写成模型优化时直接操纵的 treatment。

## 4. Model-only结论

model-only 只回答一个问题：

> 改变模型结构或训练目标以后，分类器本身有没有更好？

当前证据显示：

- `GatedFusion-12` 是目前公开表中最强的 aggregate classifier：accuracy 94.91%，macro-F1 77.50%，适合作为下一轮模型层 Pareto 实验的默认 backbone。
- `PRO` 在 3 个 paired duplicate-family seeds 中有正向分类信号：accuracy +3.71 pp，macro-F1 +4.42 pp，VT/VF cross-errors -15.33，total errors -160.33；但同时有明显 error migration，例如 SR->VT 约 +128.3，VT->VF 约 +36.7。因此它只能做 guarded ablation，不能直接扩大为主模型。
- `risk_pro_readable` 的 10 seed paired 结果不稳定：accuracy delta -0.0046，macro-F1 delta -0.0003，VT/VF cross-errors delta +8.5，total errors delta +24.4。因此不应作为下一轮主模型方向。
- `CNN-TCN-Validity` 相对 CNN 的分类效果是混合的：accuracy -0.0049，macro-F1 +0.0150，ECE +0.0202，VT/VF cross-errors -13.1，total errors +18.3。它不能被简单宣称为更好的分类器。
- `CNN-Wavelet-TCN boundary` 的 seed42 分类器本身较弱：accuracy 87.01%，macro-F1 61.58%，ECE 7.88%，VT/VF cross-errors 415，total errors 545。因此 wavelet 不适合作为 standalone classifier replacement。

model-only 的初步结论是：下一轮不应该盲目扩大 PRO/Risk-Pro，也不应该把 wavelet classifier 当作主模型替代。更合理的模型层路线是以 `GatedFusion` 为 backbone，做受控的 validity、wavelet、regularity auxiliary ablation，并用分类指标、校准指标和 error migration 共同筛选。

## 5. Evidence-head结论

evidence-head-only 只回答一个问题：

> 某个证据信号是否更能识别错误、边界或不可靠样本？

当前证据显示：

- validity gate 的 any-error AUROC 为 0.909，说明它对“是否会错”有较强信号。
- validity gate 的 VT/VF boundary AUROC 为 0.656，说明它对 VT/VF 边界有一定信号，但还不够强。
- wavelet VT/VF boundary risk 的 AUROC 约 0.962，说明 wavelet 对 VT/VF 边界风险非常有价值。
- KNN mixing density、embedding distance、prototype distance 应保留为 evidence variables，但必须在 held-out split 上重算，避免用同一批数据形成原因、又用同一批数据验证原因。

这部分不能直接推出“某个模型更好”。它只能说明某些证据头值得进入 fixed-router downstream 或下一轮辅助训练。

## 6. Fixed-router downstream设计

fixed-router downstream 只回答一个问题：

> 在 V5D 路由机制固定不变时，不同模型输出是否能让同一个路由器 recover 得更好？

正确对比是：

| Experiment | Fair comparison |
| --- | --- |
| validity auxiliary | `V5D(GatedFusion + validity_aux)` vs `V5D(GatedFusion baseline)` |
| wavelet auxiliary | `V5D(GatedFusion + wavelet_aux)` vs `V5D(GatedFusion baseline)` |
| regularity auxiliary | `V5D(GatedFusion + regularity_aux)` vs `V5D(GatedFusion baseline)` |
| guarded prototype | `V5D(GatedFusion + guarded_PRO)` vs `V5D(GatedFusion baseline)` |

固定项必须包括同一个 V5D stage1-stage2 policy、同一个 review budget、同一个 split、同一套 recover rules。变化项只能是模型输出或模型侧证据。这样得到的结论才是“某个模型侧干预更适配固定路由器”，而不是把模型和路由混为一谈。

## 7. 下一步实验

建议下一步先做四个 model-only smoke/ablation，而不是直接跑完整下游路由：

| Experiment | Intervention | Model-only outcomes | Guardrail |
| --- | --- | --- | --- |
| A | `GatedFusion + validity_aux` | accuracy, macro-F1, ECE, VT/VF cross-errors, total errors | 不接受 accuracy-only 改善 |
| B | `GatedFusion + wavelet_aux` | accuracy, macro-F1, ECE, VT->VF/VF->VT migration | 禁止 VT/VF 迁移变差 |
| C | `GatedFusion + regularity_aux` | SR-ventricular errors, total errors, VT/VF non-regression | 保持 VT/VF 分类不退化 |
| D | `GatedFusion + guarded_PRO` | macro-F1, VT/VF cross-errors, SR->VT/VF migration, ECE | 必须有 error-migration penalty |

只有 model-only 层过关以后，才进入 fixed-router downstream，比较 `V5D(model=A)` 与 `V5D(model=B)`。

## 8. 当前结论

这一步没有启动新训练，也不能声称新模型已经提升。它完成的是方法学升级：

> 把模型层、证据头层、路由/recover层拆成三个公平比较单元；保留已有 ECG 波形、embedding、KNN density、prototype distance、validity、wavelet、regularity 等证据；并把下一轮模型改善写成可干预变量、不可干预变量和 outcome 清晰分离的因果-Pareto实验设计。

