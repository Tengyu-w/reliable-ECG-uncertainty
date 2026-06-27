# Representation Evidence Usefulness Audit

## 为什么做这个实验

我们此前发现：直接通过 loss/约束优化表征层，不一定让 VT/VF 分类更清晰。因此需要验证一个关键问题：

> 表征层证据如果不能稳定改善主分类器，把它放进机制路由里是否真的有用？还是只是把一个不可靠信号换了个地方继续使用？

本实验专门回答这个问题。

## 实验设计

基于 v4 10seed 结果，读取每个 seed 的：

- `evidence_scores_val.csv`
- `evidence_scores_test.csv`

重新训练机制风险头，比较三种版本：

1. `full`
   - 使用完整机制证据。

2. `no_representation`
   - 去掉 representation 相关特征。
   - 去掉的包括 prototype、nearest prototype、KNN、representation prior-calibration 等。

3. `representation_only`
   - 只使用 representation 相关特征。

然后在同样的 optimized mechanism routing v4 策略下比较：

- 机制头 AUROC/AUPR
- all-error addressed
- VT/VF cross-error addressed
- unresolved VT/VF rate

脚本：

`src/representation_usefulness_audit.py`

输出目录：

`results/evidence_informed_mechanism_routing_10seed_v4_20260627/representation_usefulness_audit/`

## 机制头结果

| variant | mechanism | AUROC mean | AUPR mean |
|---|---|---:|---:|
| full | representation conflict | 0.9899 | 0.6857 |
| no representation | representation conflict | 0.9860 | 0.6442 |
| representation only | representation conflict | 0.9897 | 0.6589 |
| full | VT/VF boundary | 0.9541 | 0.3876 |
| no representation | VT/VF boundary | 0.9552 | 0.3880 |
| representation only | VT/VF boundary | 0.9586 | 0.4188 |
| full | atypical signal | 0.9492 | 0.5606 |
| no representation | atypical signal | 0.9425 | 0.5277 |
| representation only | atypical signal | 0.9401 | 0.5006 |
| full | SR-ventricular | 0.9126 | 0.5089 |
| no representation | SR-ventricular | 0.9090 | 0.5097 |
| representation only | SR-ventricular | 0.9148 | 0.5101 |

解释：

- representation-only 并不弱，说明表征证据确实能独立预测某些错误机制。
- representation conflict 机制头对表征证据最敏感，但即使去掉显式 representation 特征，historical diagnostics 中仍保留了一些相关信号。
- VT/VF boundary 机制中，representation-only 的 AUROC/AUPR 甚至略高，说明 embedding/KNN/prototype 对 VT/VF 边界风险不是无效的。

## 路由结果

### 10% budget

| variant | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|
| full | 0.5977 | 0.6709 | 0.0188 |
| no representation | 0.5431 | 0.5652 | 0.0217 |
| representation only | 0.5711 | 0.5766 | 0.0208 |

full 相比 no-representation：

- all-error addressed: +0.0546
- VT/VF cross-error addressed: +0.1057
- unresolved VT/VF rate: -0.0029

full 相比 representation-only：

- all-error addressed: +0.0266
- VT/VF cross-error addressed: +0.0942
- unresolved VT/VF rate: -0.0020

### 20% budget

| variant | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|
| full | 0.8268 | 0.8796 | 0.0082 |
| no representation | 0.8081 | 0.8816 | 0.0079 |
| representation only | 0.8376 | 0.8963 | 0.0069 |

20% budget 下，representation-only 的 VT/VF capture 更高，但 full 的 overall 也很接近。

### 30% budget

| variant | all-error addressed | VT/VF cross-error addressed | unresolved VT/VF rate |
|---|---:|---:|---:|
| full | 0.9387 | 0.9735 | 0.0021 |
| no representation | 0.9497 | 0.9955 | 0.0004 |
| representation only | 0.9305 | 0.9975 | 0.0001 |

30% budget 接近饱和，不是最适合区分机制贡献的区间。

## 关键结论

### 1. 表征证据不是无效信号

representation-only 能独立训练出强机制头：

- representation conflict AUROC: 0.9897
- VT/VF boundary AUROC: 0.9586
- atypical signal AUROC: 0.9401

这说明表征证据虽然不一定能直接改善分类器，但能作为 failure detector。

### 2. 表征证据不是万能信号

full 并不在所有 budget 都优于 no-representation 或 representation-only。尤其 20%-30% budget 下，representation-only 或 no-representation 有时更强。

这支持我们之前的谨慎判断：

> 表征层不能被当作正确性裁判，只能作为可验证的错误机制证据。

### 3. 10% budget 是表征证据最有增益的区间

10% budget 下，full 明显优于 no-representation：

- VT/VF capture 提升约 +0.1057
- unresolved VT/VF rate 下降约 -0.0029

这说明在低/中复核预算下，表征证据能提供独立价值。

### 4. 表征证据的正确用法是“冲突检测”，不是“改标签”

我们没有用表征层直接把 VT 改成 VF 或把 VF 改成 VT。它用于判断：

- classifier 是否和 prototype/KNN 冲突；
- 样本是否处于表征边界；
- 样本是否 atypical。

因此它的角色是：

> failure detector, not final classifier.

## 对论文/申请的意义

这个实验能回应一个很强的审稿人问题：

> 如果 representation optimization 没能稳定改善分类，为什么 representation evidence 能用于 routing？

回答是：

> Because representation evidence is not used as a direct classifier. It is used as a mechanism-specific failure signal, and its usefulness is empirically validated through feature-removal and representation-only routing ablations across 10 seeds.

中文：

> 表征证据不是作为直接分类器使用，而是作为机制特定的失败信号使用。我们通过 10seed 的去表征和仅表征消融证明，它在低/中预算路由中提供独立价值。

## 限制

- full 并非所有预算下都最好。
- representation-only 在部分预算下很强，说明 v5 应考虑让 validation profile 更自由地选择 representation-heavy 策略。
- historical diagnostics 中仍可能包含部分 representation-derived 信号，因此 no-representation 不是完全无表征，只是去掉显式 representation 特征。

## 下一步

v5 应该：

1. 显式加入 `representation-heavy` 和 `boundary-representation` profile。
2. 把 objective 分成 VT/VF capture 与 all-error capture。
3. 报告每个机制 route 的 feature-removal sensitivity。
4. 将表征证据描述为 failure signal，而不是 classification evidence。
