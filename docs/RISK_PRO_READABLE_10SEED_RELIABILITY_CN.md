# Risk-Pro-readable 10-seed 可靠性验证

## 目的

本轮验证不是继续改模型，而是先回答一个问题：

> seed42 上 Risk-Pro-readable 看起来改善了 calibration 和边界风险可检测性，这个现象是否能在 10 seed 上稳定复现？

验证使用 duplicate-family split，seeds 42-51。每个 seed 包含一个 regularity-injection teacher 和一个 Risk-Pro-readable 模型。Risk-Pro-readable 的核心约束包括：

- VT/VF readability loss；
- selective low-risk stability；
- anti-confident high-risk penalty；
- risk-weighted CE；
- risk entropy alignment；
- mild prototype / contrastive / regularity auxiliary losses。

结果目录：

`results/risk_pro_readable_10seed_20260626/summary/`

## 10-seed 平均结果

| metric | teacher mean | Risk-Pro-readable mean | readable - teacher |
|---|---:|---:|---:|
| Accuracy | 0.8997 | 0.8951 | -0.0046 |
| Macro-F1 | 0.6499 | 0.6496 | -0.0003 |
| ECE | 0.0554 | 0.0535 | -0.0019 |
| VT/VF cross-errors | 189.7 | 198.2 | +8.5 |
| Total errors | 435.4 | 459.8 | +24.4 |
| Softmax VT/VF ambiguity AUROC | 0.6747 | 0.6685 | -0.0062 |
| KNN VT/VF mixing AUROC | 0.6383 | 0.6100 | -0.0283 |
| Stable confident VT/VF cross-errors | 40.5 | 57.0 | +16.5 |
| Stable confident VT/VF confidence | 0.8943 | 0.9053 | +0.0110 |

## Paired uncertainty

| metric | mean delta | median delta | bootstrap 95% CI | direction |
|---|---:|---:|---:|---|
| Accuracy | -0.0046 | -0.0094 | [-0.0273, 0.0184] | no stable gain |
| Macro-F1 | -0.0003 | -0.0018 | [-0.0284, 0.0297] | no stable gain |
| ECE | -0.0019 | -0.0004 | [-0.0185, 0.0148] | no stable gain |
| VT/VF cross-errors | +8.5 | -0.5 | [-13.2, 32.5] | no stable reduction |
| Total errors | +24.4 | +31.5 | [-83.6, 130.5] | no stable reduction |
| Softmax VT/VF ambiguity AUROC | -0.0062 | +0.0102 | [-0.0451, 0.0305] | no stable gain |
| KNN VT/VF mixing AUROC | -0.0283 | -0.0441 | [-0.0749, 0.0209] | likely worse / unstable |
| Stable confident VT/VF cross-errors | +16.5 | -3.5 | [-10.1, 52.1] | no stable reduction |
| Stable confident VT/VF confidence | +0.0110 | +0.0147 | [-0.0056, 0.0272] | no stable reduction |

## Per-seed behavior

Risk-Pro-readable is highly seed-dependent.

Improved seeds:

- seed45: accuracy +0.0131, macro-F1 +0.0457, VT/VF cross-errors -39, total errors -41.
- seed47: accuracy +0.0581, macro-F1 +0.0851, VT/VF cross-errors -3, total errors -270.
- seed49: accuracy +0.0355, macro-F1 +0.0545, VT/VF cross-errors -38, total errors -159.

Worse seeds:

- seed44: accuracy -0.0470, macro-F1 -0.0553, total errors +209.
- seed48: accuracy -0.0381, VT/VF cross-errors +34, total errors +235.
- seed50: accuracy -0.0625, macro-F1 -0.0486, total errors +282.
- seed51: VT/VF cross-errors +35, total errors +45.

The model therefore has useful behavior in some seeds, but it is not yet a stable improvement.

## Interpretation

The seed42 pilot was not fully representative.

Risk-Pro-readable introduced a principled training objective: boundary readability, selective stability, and anti-confident risk. However, in 10-seed validation these constraints do not reliably improve classification, calibration, VT/VF cross-errors, or stable confident VT/VF errors.

The important negative result is:

> Making risk and readability explicit is conceptually useful, but the current loss mixture is not yet well-conditioned enough to be the final model.

This is still valuable. It tells us that the next model should not simply add more auxiliary losses. The failure mode is probably loss interference:

- risk-weighted CE may overemphasize difficult boundary samples and destabilize SR/VT/VF balance;
- anti-confident penalty may lower confidence but not necessarily move the boundary correctly;
- readability loss can improve probe separability without improving the deployed classifier decision;
- prototype/contrastive losses may still stabilize the wrong regions.

## Current Decision

Risk-Pro-readable should **not** replace CNN-LSTM or the teacher as the main final model yet.

Current best positioning:

- CNN-LSTM remains the stronger evidence-backed boundary-mitigation baseline because it has 10-seed evidence for reducing VT/VF cross-errors relative to CNN.
- Risk-Pro-readable is a useful mechanistic pilot, but not a validated improvement.
- The next model should be simpler and more targeted, probably not another large additive loss mixture.

## Recommended Next Step

Use this negative result to design a cleaner second-generation model:

1. Keep the frozen backbone / teacher representation.
2. Do not retrain the full classifier with many competing losses.
3. Add a separate VT/VF reliability head or approval head on top of frozen embeddings.
4. Train the approval head to identify boundary-risk / review-needed samples.
5. Evaluate it with review capture, prediction-set error reduction, and stable confident VT/VF error suppression.

This would turn the current result into a stronger paper story:

> Full-model risk regularization is unstable across seeds, but frozen-backbone boundary approval is a more reliable way to convert representation diagnostics into safer selective prediction.
