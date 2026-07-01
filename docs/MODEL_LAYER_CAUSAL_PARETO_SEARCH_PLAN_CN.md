# 模型层因果-Pareto约束重组搜索

对应脚本：

```text
python -m src.run_model_layer_causal_pareto_search
```

## 1. 这一步做的是什么

这一步才是对旧约束候选做真正的模型层优化，而不是只汇总旧结果。

上一轮全模型段 benchmark 发现：

- `prototype_center=0.02 + prototype_margin=0.05`：macro-F1 和 VT/VF cross-errors 最强；
- `boundary_ce=1.0 + risk_targets`：accuracy、total errors、error migration penalty 最强；
- `full_supervisor`：均衡，但不是全局最优；
- `RiskProReadable complex objective`：ECE 强，但错误和迁移风险较大。

因此新的优化逻辑是：

> 以旧强候选为起点，重组 prototype、boundary/risk、light regularity、light stability、light calibration，形成新的 model-layer Pareto search。

## 2. 可干预变量

本轮搜索使用以下可干预训练变量：

| Variable | Meaning |
| --- | --- |
| `do(prototype_center_weight)` | 控制同类 embedding compactness |
| `do(prototype_margin_weight)` | 控制 VT/VF prototype separation |
| `do(boundary_ce_weight)` | 对高 risk/boundary 样本提高 CE 权重 |
| `do(risk_targets)` | 使用 baseline teacher 生成的 risk target |
| `do(regularity_aux_weight)` | 加入 ECG rhythm/morphology regularity auxiliary |
| `do(stability_consistency_weight)` | 加入扰动预测一致性 |
| `do(embedding_consistency_weight)` | 加入扰动 embedding 一致性 |
| `do(risk_entropy_weight)` | 轻量校准/风险熵对齐 |
| `do(anti_confident_risk_weight)` | 限制高风险样本过度自信 |

不可干预变量仍然是 ECG 原始波形、真实标签、split/grouping、seed 设计和数据来源。

## 3. 候选矩阵

脚本目前定义了 8 个候选：

| Candidate | Role | Intervention |
| --- | --- | --- |
| `baseline` | control | 无额外约束 |
| `prototype_guard` | old strong candidate | `prototype_center=0.02`, `prototype_margin=0.05` |
| `boundary_risk` | old strong candidate | `boundary_ce=1.0 + risk_targets` |
| `boundary075_prototype` | new recombination | `boundary_ce=0.75 + prototype` |
| `boundary100_prototype` | new recombination | `boundary_ce=1.0 + prototype` |
| `boundary075_prototype_reg` | new recombination | `boundary_ce=0.75 + prototype + regularity_aux=0.02` |
| `boundary075_prototype_stability` | new recombination | `boundary_ce=0.75 + prototype + light stability` |
| `boundary075_prototype_calibrated` | new recombination | `boundary_ce=0.75 + prototype + light risk entropy + anti-confident risk` |

## 4. Outcome

模型层 Pareto outcome 仍然只使用模型分类指标：

| Outcome | Direction |
| --- | --- |
| accuracy | higher better |
| macro-F1 | higher better |
| ECE | lower better |
| VT/VF cross-errors | lower better |
| total errors | lower better |
| error migration penalty | lower better |

没有使用 V5D/recover/routing 指标。

## 5. Smoke run

已完成一个 smoke run：

```text
python -m src.run_model_layer_causal_pareto_search \
  --seeds 42 \
  --epochs 1 \
  --max-windows-per-record 2 \
  --candidates baseline prototype_guard boundary_risk boundary075_prototype \
  --out results/model_layer_causal_pareto_search_smoke_20260630
```

输出：

```text
results/model_layer_causal_pareto_search_smoke_20260630/
```

smoke 结果：

| Candidate | Accuracy delta | Macro-F1 delta | ECE delta | VT/VF cross-error delta | Total-error delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prototype_guard` | +0.0000 | -0.0137 | +0.0111 | 0 | 0 |
| `boundary_risk` | +0.0182 | +0.0545 | -0.0281 | -4 | -4 |
| `boundary075_prototype` | +0.0045 | +0.0126 | -0.0141 | -1 | -1 |

注意：这个 smoke 只用于验证链路，不用于模型结论。因为它只跑了 1 个 seed、1 epoch，并且每个 record 只取 2 个 windows。

## 6. 下一步完整验证命令

建议正式运行先用 3 seeds，而不是直接扩大到 10 seeds：

```text
python -m src.run_model_layer_causal_pareto_search \
  --seeds 42 43 44 \
  --epochs 30 \
  --candidates baseline prototype_guard boundary_risk boundary075_prototype boundary100_prototype boundary075_prototype_reg boundary075_prototype_stability boundary075_prototype_calibrated \
  --out results/model_layer_causal_pareto_search_full_20260630
```

如果 3 seeds 里出现稳定 Pareto candidate，再扩到 10 seeds：

```text
python -m src.run_model_layer_causal_pareto_search \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --epochs 30 \
  --candidates baseline prototype_guard boundary_risk boundary075_prototype boundary100_prototype boundary075_prototype_reg boundary075_prototype_stability boundary075_prototype_calibrated \
  --out results/model_layer_causal_pareto_search_10seed_20260630
```

## 7. 当前结论

当前已经完成的是：

> 建立并 smoke 验证了模型层因果-Pareto约束重组搜索器。它能把旧模型约束候选转化为可干预变量，自动生成同 seed risk targets，训练新组合候选，并输出 paired effect 与 Pareto summary。

当前还没有完成的是：

> 不能根据 smoke 结果宣称新模型优于旧模型。真正结论必须等待 3 seed 或 10 seed 完整运行。

