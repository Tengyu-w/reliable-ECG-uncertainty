# Validity/Boundary Signal 作为机制路由证据的专门审计

## 一句话结论

这次专门实验支持一个很关键的判断：`CNN+TCN+Validity` 第一版作为端到端分类器没有稳定变好，但它内部学到的 `validity_gate` / `boundary_score` 可以非常有效地识别 VT/VF 交叉错误。因此，这类结构性信号更适合作为恢复路由里的机制证据，而不是直接作为 logits 修正器。

## 这次审计回答的问题

前面的疑问是：

> 表征层、validity gate、boundary score 这些信号在训练约束里不一定能让 VT/VF 分得更清楚，那把它们放进 recovery router 会不会只是把不靠谱的东西换个地方用？

所以这次没有再看新模型自己的 accuracy，而是做了一个更直接的路由审计：

1. 读取 `CNN+TCN+Validity`、`CNN+TCN+Validity v2`、`Wavelet+TCN+Boundary` 输出的 `validity_gate_scores_test.csv`。
2. 与 v4 mechanism router 的 teacher evidence test set 对齐。
3. 用 teacher/router 的真实错误标签作为目标：
   - `is_error`
   - `is_vtvf_cross_error`
4. 把以下信号当成 review ranking score：
   - `validity_gate`
   - `boundary_score`
   - `gate_x_boundary`
   - `gate_minus_confidence`
   - `low_validity_model_confidence`
5. 在 5%、10%、20%、30% review budget 下评估能捕获多少错误。

## 数据对齐情况

| source | seeds | y_true aligned | y_pred same as teacher | 解释 |
|---|---:|---|---|---|
| CNN+TCN+Validity v1 | 10 | yes | no | 样本完全对齐，但预测来自另一个模型，符合预期 |
| CNN+TCN+Validity v2 | 1 | yes | no | seed42 探索性结果 |
| Wavelet+TCN+Boundary | 1 | yes | no | seed42 探索性结果 |

这说明审计比较的是同一批 test samples 上的风险排序能力，而不是混用了不同测试集。

## 10seed 主要结果：CNN+TCN+Validity v1

| score | budget | all-error capture | VT/VF cross-error capture | 解释 |
|---|---:|---:|---:|---|
| `validity_gate` | 10% | 48.9% | 83.1% | 最强的 10% budget 信号 |
| `gate_x_boundary` | 10% | 48.7% | 82.7% | 与 gate 接近 |
| `boundary_score` | 10% | 48.1% | 81.5% | 边界信号本身也很强 |
| `boundary_score` | 20% | 77.8% | 98.3% | 基本捕获绝大多数 VT/VF 交叉错误 |
| `gate_x_boundary` | 20% | 78.1% | 98.3% | 与 boundary score 并列 |
| `validity_gate` | 20% | 78.2% | 98.1% | 仍然很强 |

这个结果非常重要：第一版 validity bottleneck 的分类 adapter 没有稳定修正 VT/VF 决策边界，但 gate/boundary 作为错误检测信号是有用的。也就是说，失败的不是“validity 信号完全没学到”，而是“把这个信号直接写进分类修正路径的方法还不够好”。

## 与已有 v4 router / strong baseline 的关系

此前 v4 optimized mechanism router 的 10seed 结果：

| method | budget | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|
| v4 optimized mechanism router | 10% | 57.7% | 59.7% |
| v4 optimized mechanism router | 20% | 82.6% | 87.9% |

此前 strong baseline 中，`softmax_vtvf_ambiguity` 很强：

| method | budget | all-error capture | VT/VF cross-error capture |
|---|---:|---:|---:|
| softmax VT/VF ambiguity | 10% | 52.3% | 84.1% |
| softmax VT/VF ambiguity | 20% | 81.0% | 99.2% |

因此现在的定位应当很清楚：

- v4 mechanism router 比 total risk 更有机制解释性，但它还没有充分吸收最强 VT/VF boundary expert。
- `CNN+TCN+Validity` 的 gate/boundary 信号达到了接近 softmax VT/VF ambiguity 的边界错误捕获能力。
- 下一版 router 不应该只用原来的 v4 证据，而应该把 `softmax_vtvf_ambiguity` 和 `validity_gate/boundary_score` 都作为 VT/VF boundary expert。

## seed42 探索性结果

`CNN+TCN+Validity v2` 和 `Wavelet+TCN+Boundary` 目前只有 seed42，所以只能作为探索性证据：

| source | score | budget | all-error capture | VT/VF cross-error capture |
|---|---|---:|---:|---:|
| CNN+TCN+Validity v2 | `gate_x_boundary` | 10% | 45.8% | 75.1% |
| CNN+TCN+Validity v2 | `boundary_score` | 20% | 73.4% | 100.0% |
| Wavelet+TCN+Boundary | `gate_minus_confidence` | 10% | 41.1% | 67.9% |
| Wavelet+TCN+Boundary | `boundary_score` | 20% | 75.5% | 100.0% |

这提示 v2/wavelet 也可能有路由价值，但证据强度不如 v1 的 10seed 结果。后续如果要写进论文主结果，应补 10seed。

## 对你前面疑问的回答

这次实验把“结构性信号是否可靠”拆成了两个问题：

1. 它能不能直接让分类器更准？
2. 它能不能帮助识别哪些样本不该自动给单标签？

前面 CNN+TCN+Validity 的结果说明，答案 1 暂时不是很稳。
这次 audit 说明，答案 2 是肯定的，尤其对 VT/VF cross-error 很强。

所以我们不是把一个不靠谱的分类依据搬进 router，而是在把它重新定义成 failure evidence。它不负责判断“这是 VT 还是 VF”，而是负责判断“这个样本很可能处于模型有效域边界，应该进入 VT/VF 专家分支、prediction set 或人工复核”。

## 对分层决策系统的更新建议

下一版机制路由可以把 `vtvf_boundary` 分支升级为 boundary expert ensemble：

| evidence group | 进入哪个机制分支 | 作用 |
|---|---|---|
| softmax VT/VF ambiguity | VT/VF boundary | 捕获分类器自己的 VT/VF 犹豫 |
| validity gate | VT/VF boundary / hidden-risk | 捕获模型内部有效域风险 |
| boundary score | VT/VF boundary | 捕获结构模型认为的边界风险 |
| gate x boundary | VT/VF boundary | 强化同时高 validity-risk 和 boundary-risk 的样本 |
| KNN mixing / prototype ambiguity | representation conflict | 判断局部表征邻域是否混杂 |
| regularity / signal morphology | atypical signal | 判断波形结构是否异常或不典型 |
| model disagreement | second-opinion conflict | 判断是否需要第二专家或 review |

## 论文可用表述

中文：

> 虽然 CNN+TCN+Validity Bottleneck 没有稳定提升端到端分类性能，但其内部 validity/boundary 信号在固定 review budget 下能够高效捕获 VT/VF 交叉错误。10seed 审计显示，`validity_gate` 在 10% review budget 下捕获 83.1% 的 VT/VF cross-errors，而 `boundary_score` 在 20% budget 下捕获 98.3%。这说明结构性有效域信号更适合作为机制路由证据，而不一定适合作为直接的标签修正约束。

English:

> Although the CNN+TCN validity bottleneck did not consistently improve end-to-end classification, its internal validity and boundary signals provided strong decision-time reliability evidence. In a 10-seed audit, the validity gate captured 83.1% of VT/VF cross-errors at a 10% review budget, while the boundary score captured 98.3% at a 20% budget. This supports using validity-domain signals as mechanism-specific routing evidence rather than direct label-correction constraints.

## 输出文件

- `src/validity_boundary_signal_audit.py`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/validity_boundary_signal_audit/all_seed_validity_boundary_signal_policy.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/validity_boundary_signal_audit/validity_boundary_signal_mean_std.csv`
- `results/evidence_informed_mechanism_routing_10seed_v4_20260627/validity_boundary_signal_audit/validity_boundary_alignment_manifest.csv`

## 下一步

最合理的下一步不是再单独讨论这些信号有没有用，而是做 v5 router：

1. 把 `validity_gate`、`boundary_score`、`softmax_vtvf_ambiguity` 合并成一个 VT/VF boundary expert。
2. 在 validation set 上学习 expert 权重，而不是手动拍权重。
3. 对比：
   - softmax-only
   - validity-only
   - v4 optimized mechanism
   - v5 boundary-expert router
4. 用 10seed paired comparison 验证 v5 是否真的提升：
   - VT/VF cross-error capture
   - all-error capture
   - automatic unresolved error rate
   - automatic unresolved VT/VF cross-error rate
