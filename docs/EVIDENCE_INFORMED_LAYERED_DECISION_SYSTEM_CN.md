# Evidence-informed 分层决策系统 v1

## 研究问题

这一步不是再证明某个表征层“看起来分得开”，而是把前面做过的输出层、表征层、局部邻域、regularity、latent cluster、模型分歧等分析证据，接入最终的恢复/复核路由。目标是回答：

- 哪些样本可以继续给单标签输出？
- 哪些 VT/VF 边界样本不应该强行二选一，而应输出 `{VT,VF}` 集合？
- 哪些样本应该进入人工复核或后续恢复模块？
- 路由之后，自动区里是否真的减少了 VT/VF cross-error，而不是只让 accuracy 看起来提高？

该系统仍是研究原型，不能解释为临床验证或医疗器械级决策。

## 已接入的证据

脚本：`src/evidence_informed_recovery_routing.py`

1. 输出层证据
   - softmax probability
   - max probability / MSP uncertainty
   - entropy / temperature entropy
   - top-1/top-2 margin
   - VT/VF probability margin
   - VT/VF softmax ambiguity

2. 表征层证据
   - embedding 到 SR/VT/VF prototype centroid 的距离
   - nearest prototype 是否同意模型预测
   - prototype margin
   - VT/VF prototype ambiguity
   - KNN label entropy
   - KNN VT/VF mixing
   - KNN prediction 是否同意模型预测

3. 数据/节律 regularity 证据
   - spectral entropy
   - dominant frequency
   - dominant frequency concentration
   - spectral centroid / bandwidth
   - autocorr peak / autocorr lag
   - zero crossing rate
   - line length

4. latent cluster 证据
   - train embedding 上 KMeans 聚类
   - validation cluster error rate
   - validation cluster VT/VF cross-error rate
   - validation cluster VT/VF truth density
   - cluster distance

5. 第二模型分歧证据
   - second-opinion probabilities
   - second-opinion entropy
   - 主模型与第二模型是否 disagreement
   - disagreement 是否发生在 VT/VF 范围内
   - second-opinion VT/VF ambiguity

6. 历史逐样本诊断证据
   - `risk_target_components_{val,test}.csv`
   - `vtvf_decision_calibration_scores_{val,test}.csv`
   - 这些表有 val/test 对应版本，因此可以进入风险模型训练。

7. test-only 历史分析审计证据
   - `ambiguity_scores.csv`
   - `stability_scores.csv`
   - `reliability_map_scores.csv`
   - `local_rhythm_instability_scores.csv`
   - `regularity_features.csv`
   - `uncertainty_scores.csv`
   - `decision_boundary_diagnosis.csv`
   - `runtime_supervisor_policy.csv`
   - `ambiguity_routing_policy.csv`
   - `embedding_neighborhood_k15.csv`
   - `embedding_lda2_coordinates.csv`
   - `embedding_pca3_coordinates.csv`
   - `conformal_sets.csv`

这些 test-only 表目前只进入 `evidence_scores_test.csv`、`layered_routing_assignments_test.csv` 和 `post_routing_audit.csv`，不直接进入风险模型训练。原因是它们没有对应 validation 表；如果直接拿 test-only 诊断训练路由，会有测试集泄漏风险。

## 分层策略

v1 使用 validation split 学两个风险头：

- `any_error_risk`：该样本是否可能总体分类错误。
- `vtvf_boundary_risk`：该样本是否可能是 VT/VF cross-error。

测试时按预算生成三类动作：

- `review`：总体错误风险最高，交给复核/恢复模块。
- `vtvf_set`：不进入 review，但 VT/VF boundary risk 高，输出 `{VT,VF}`，不强行 VT/VF 二选一。
- `single_label`：风险较低，保留原始单标签输出。

这一步刻意不把“表征层距离大”直接当成正确性标准。表征层、KNN、prototype 等只作为风险证据进入路由模型；最后还要用 routing 后的真实错误残留重新审计。

## 机制分型路由 v3

v3 不再只使用一个总风险头，而是把错误先分成多个机制，再给每个机制分配对应证据组和处理动作：

| 错误机制 | 目标 | 主要证据 | 路由动作 |
|---|---|---|---|
| VT/VF boundary error | VT 被判 VF 或 VF 被判 VT | VT/VF softmax margin、prototype VT/VF ambiguity、KNN VT/VF mixing、regularity、historical boundary diagnostics | 输出 `{VT,VF}` 或进入边界复核 |
| SR-ventricular direction error | SR 与 VT/VF 互相误判 | SR/ventricular probability、regularity、prototype/KNN、latent cluster、risk target | SR-ventricular review |
| representation conflict error | classifier 与 prototype/KNN 表征证据冲突时的错误 | nearest prototype、KNN prediction、prototype distance、prior calibration diagnostics | representation review |
| atypical signal error | atypical 或局部数据结构异常相关错误 | regularity、cluster distance、KNN/prototype distance、atypicality/risk target | atypical review |
| hidden confident error | 高置信、低熵、邻域稳定但仍然错误 | confidence、entropy、KNN stability、模型一致性 | hidden failure review |

v3b 进一步加入两个约束：

- 每个机制不再各自拿完整 budget，而是共享总预算，避免 action rate 失控。
- validation 中没有足够正例的机制只保留分析，不参与路由。seed42 中 `hidden_confident` 的 validation 正例为 0，因此不启用路由。

## 当前 seed42 验证

运行命令：

```powershell
python -m src.evidence_informed_recovery_routing `
  --run-dir results\core_interventions_risk_pro_plus\20260625_202904_reliability_gated_fusion_core_regularity_injection_seed42 `
  --second-opinion-run-dir results\risk_pro_readable_20260626\20260626_152753_reliability_gated_fusion_core_risk_pro_readable_seed42 `
  --out results\evidence_informed_recovery_routing_seed42_teacher_readable `
  --budgets 0.05 0.10 0.20 0.30
```

输出目录：

`results/evidence_informed_recovery_routing_seed42_teacher_readable/`

扩展版输出目录：

`results/evidence_informed_recovery_routing_seed42_teacher_readable_v2/`

机制分型路由输出目录：

`results/evidence_informed_recovery_routing_seed42_teacher_readable_v3b/`

核心结果：

| budget | review rate | VT/VF set rate | single-label rate | addressed all errors | addressed VT/VF cross-errors | single-label VT/VF cross-error after routing |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0489 | 0.0088 | 0.9423 | 0.1442 | 0.1658 | 0.0407 |
| 0.10 | 0.0925 | 0.0031 | 0.9044 | 0.2759 | 0.3420 | 0.0335 |
| 0.20 | 0.2000 | 0.0000 | 0.8000 | 0.7085 | 0.8653 | 0.0077 |
| 0.30 | 0.3214 | 0.0000 | 0.6786 | 0.9843 | 1.0000 | 0.0000 |

baseline：

- accuracy: 0.9239
- total error rate: 0.0761
- VT/VF cross-errors: 193
- VT/VF cross-error rate: 0.0460

风险模型初步表现：

- `any_error_risk` AUROC: 0.8777
- `any_error_risk` AUPR: 0.2447
- `vtvf_boundary_risk` AUROC: 0.9037
- `vtvf_boundary_risk` AUPR: 0.1806

扩展版 v2 的特征规模：

- softmax only: 18
- representation only: 19
- regularity only: 9
- latent cluster only: 5
- model disagreement only: 11
- historical diagnostics: 40
- all evidence: 104

扩展版 v2 的风险模型：

- `any_error_risk` AUROC: 0.8802
- `any_error_risk` AUPR: 0.2840
- `vtvf_boundary_risk` AUROC: 0.9037
- `vtvf_boundary_risk` AUPR: 0.1798

v3b 机制风险头：

| mechanism | validation positives | test positives | enabled | AUROC | AUPR |
|---|---:|---:|---|---:|---:|
| VT/VF boundary | 302 | 193 | yes | 0.9006 | 0.1778 |
| SR-ventricular | 142 | 126 | yes | 0.8967 | 0.2669 |
| representation conflict | 174 | 64 | yes | 0.9893 | 0.5517 |
| atypical signal | 314 | 182 | yes | 0.8830 | 0.2224 |
| hidden confident | 0 | 8 | no | 0.5000 | 0.0019 |

v3b 机制路由效果：

| budget | mechanism action rate | `{VT,VF}` set rate | single-label rate | all-error addressed | VT/VF cross-error addressed | single-label VT/VF cross-error after routing |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0794 | 0.0606 | 0.9206 | 0.1755 | 0.1554 | 0.0422 |
| 0.10 | 0.1056 | 0.0830 | 0.8944 | 0.2476 | 0.2280 | 0.0397 |
| 0.20 | 0.1390 | 0.1094 | 0.8610 | 0.3793 | 0.3938 | 0.0324 |
| 0.30 | 0.1784 | 0.1369 | 0.8217 | 0.5831 | 0.6269 | 0.0209 |

这里的 v3b 故意不是追求短期最高捕获率，而是证明“错误机制 -> 证据组 -> 路由动作”的结构是可运行、可审计的。后续可以用验证集优化各机制预算权重，让稳定性和捕获率一起提高。

## 证据消融的初步信号

在 VT/VF cross-error 捕获上，10% budget 时：

- regularity only 捕获 57.5% VT/VF cross-error。
- all evidence 捕获 39.4%。
- softmax only 捕获 38.9%。
- latent cluster only 捕获 34.2%。
- model disagreement only 捕获 31.6%。
- representation only 捕获 26.9%。

这说明当前 seed42 里，节律/数据特征对 VT/VF 边界风险非常有信息量；表征层证据不是没用，但单独使用并不是最强。这也支持我们之前的担心：不能把 embedding 分离直接等同于 VT/VF 判对。

扩展版 v2 在 10% budget 时：

- regularity only 捕获 57.5% VT/VF cross-error。
- softmax only 捕获 38.9%。
- all evidence 捕获 37.3%。
- historical diagnostics 捕获 36.8%。
- latent cluster only 捕获 34.2%。
- model disagreement only 捕获 31.6%。
- representation only 捕获 26.9%。

这个结果说明，简单把所有特征拼起来并不一定优于最强的单一证据组。下一版更应该做分层门控或特征选择，而不是继续粗暴加列。

## 产物文件

- `evidence_scores_val.csv`：validation split 的所有证据、风险分数和标签。
- `evidence_scores_test.csv`：test split 的所有证据、风险分数和标签。
- `evidence_ablation_summary.csv`：softmax、representation、regularity、latent cluster、model disagreement、all evidence 的消融。
- `layered_policy_summary.csv`：不同预算下的分层路由结果。
- `layered_routing_assignments_test.csv`：每个 test 样本的最终路由动作。
- `post_routing_audit.csv`：路由后不同动作区域的错误率、VT/VF cross-error、entropy、KNN mixing、模型分歧等审计指标。
- `layered_decision_system_report.json`：机器可读汇总。
- `mechanism_risk_head_summary.csv`：每个错误机制风险头的目标、特征数量、正例数、AUROC/AUPR、是否启用路由。
- `mechanism_layered_policy_summary.csv`：机制分型路由在不同预算下的总体表现。
- `mechanism_routing_assignments_test.csv`：每个 test 样本被分配到哪个错误机制路由。
- `mechanism_route_audit.csv`：每个机制路由区域内部的真实错误组成。

## 下一步

v1 已经把所有分析证据接入路由，但目前只跑了 seed42。下一步应该做：

1. 10-seed paired routing validation：确认路由收益不是 seed42 偶然现象。
2. 把 `review` 和 `vtvf_set` 拆成两个不同预算，而不是共用同一个 budget。
3. 将当前 logistic risk head 升级为更清晰的层级模型，例如先做 OOD/atypicality gate，再做 VT/VF boundary gate，最后做 recovery route。
4. 对 routed single-label 区域重新做 representation/KNN/regularity audit，证明“留下来的自动判断”确实更干净。
