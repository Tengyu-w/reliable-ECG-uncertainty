# 机制感知分层路由的文献定位矩阵

## 我们的方法定位

暂定方法名：

**Mechanism-aware Evidence-Routed Selective Classification**

核心思想：

不是把 uncertainty 压缩成一个总风险分数，而是把 ECG 分类失败拆成不同错误机制，再给每个机制分配对应证据组和路由动作。

当前机制包括：

- VT/VF boundary confusion
- SR-ventricular direction error
- representation conflict
- atypical signal
- hidden confident failure

## 相关方向与差异

| 方向 | 代表工作 | 主要思想 | 与我们的关系 | 我们的差异 |
|---|---|---|---|---|
| Selective classification / reject option | Geifman & El-Yaniv, 2017, [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) | 用置信度或选择函数决定接收/拒绝预测 | 是我们的基础问题设定 | 它通常是单一 reject score；我们分错误机制和证据组 |
| Integrated reject model | Geifman & El-Yaniv, 2019, [SelectiveNet](https://arxiv.org/pdf/1901.09192) | 把 selective head 集成到网络中 | 可作为未来训练端 baseline | 我们目前是后验 evidence routing，不是单一 selective head |
| Conformal prediction | Angelopoulos & Bates, [Conformal Prediction tutorial](https://arxiv.org/abs/2107.07511) | 产生覆盖有保证的 prediction set | 与 `{VT,VF}` set 有关 | conformal 强调覆盖保证；我们强调错误机制对应的 set/review 动作 |
| 医疗 AI uncertainty/abstention | Kompa et al., [Nature Digital Medicine](https://www.nature.com/articles/s41746-020-00367-3) | 医疗 ML 中表达不确定性和 abstention | 支持安全/复核动机 | 该方向多是 general uncertainty；我们做 ECG 机制化路由 |
| ECG rejection / uncertainty | Zhang et al., [Deep Bayesian Neural Network for Cardiac Arrhythmia Classification with Rejection](https://arxiv.org/abs/2203.00512) | ECG 分类中用 Bayesian uncertainty 拒判 | 最接近 ECG 场景 | 它仍是 uncertainty threshold/rejection；我们区分 VT/VF boundary、representation conflict、atypical 等机制 |
| ECG deep classification | 例如 CNN/BiLSTM ECG 分类综述与模型 | 提升分类 accuracy | 是任务背景 | 我们不是只追求 accuracy，而是处理不可靠输出 |
| Mixture-of-experts routing | MoE / expert routing 文献 | 不同输入进入不同专家 | 思想相近 | 我们的 expert 是错误机制专家，不是普通类别专家或容量专家 |

## 可主张的创新点

可以主张：

1. **从单一 uncertainty score 转向错误机制分解。**
2. **把不同证据组映射到不同错误机制头。**
3. **把不同错误机制映射到不同动作：`{VT,VF}`、SR-ventricular review、representation review、atypical review。**
4. **用 validation 选择机制组合，而不是手写固定规则。**
5. **在 10-seed 内部验证中，v4 在低/中预算段优于 learned total-risk routing。**

不应夸大：

- 不能说 selective routing 是我们首创。
- 不能说 conformal prediction 或 reject option 是我们首创。
- 不能说已经达到临床可用。
- 不能说已经外部验证。

## 最稳妥的论文表述

英文：

> Existing selective classification and abstention methods commonly rely on a scalar uncertainty or risk score. We instead propose a mechanism-aware evidence-routed framework that decomposes ECG classification failures into distinct error mechanisms and assigns each mechanism to evidence-specific reliability heads and tailored routing actions.

中文：

> 现有选择性分类和拒判方法通常依赖单一不确定性或风险分数。我们提出一种机制感知的证据路由框架，将 ECG 分类失败分解为不同错误机制，并为每种机制分配特定证据组和对应路由动作。

## 目前定位

这更像一个**新框架/新组合创新**，而不是某个单独模块的全新发明。它的价值在于把：

- selective classification
- prediction set
- representation diagnostics
- rhythm regularity analysis
- local neighborhood instability
- model disagreement

组合成一个面向 VT/VF reliability 的机制化路由系统。
