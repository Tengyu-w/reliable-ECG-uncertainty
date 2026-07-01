# 路由机制 OOD-style ECG Shift 压力测试

对应脚本：

```text
python -m src.run_route_ood_stress
```

输出目录：

```text
results/route_ood_stress_20260629/
```

说明：本文件中的 OOD 数字来自该旧 pilot 输出；表格中的 `complete_router_profile_*` 是按照新 taxonomy 对旧 policy 名称做的解释性重命名。新命名脚本已准备好，但完整 OOD rerun 尚未完成。

## 0. 重要修正

本测试比较的是完整路由 policy，不是裸 evidence model。

旧输出中的 `causal_pareto_wavelet_boundary_heavy` 应理解为：

> 由 wavelet-boundary-heavy evidence profile 包装成的完整 routing policy。

脚本已将未来输出命名改为：

- `complete_router_profile_wavelet_boundary_heavy`;
- `complete_router_profile_validity_boundary__sr_ventricular`;
- `complete_router_profile_wavelet_boundary__atypical_signal`.

这三个对象都包含 evidence weighting、candidate mask、budget selection 和 recover action，因此可以和 V5D 比较。

## 1. 为什么做这个测试

这个测试回应一个关键质疑：

> 当前路由机制是否只是适配了本数据集的波形、边界和表征特点？如果 ECG 测试样本发生合理分布偏移，路由是否仍然稳定？

由于没有外部 ECG 数据，本测试使用内部 OOD-style ECG shift。它不能替代外部验证，但可以作为伪外部压力测试，检查机制是否过度依赖 clean-set 特征。

## 2. ECG 结构边界

默认只使用相对保留 ECG 时间结构的扰动：

- `gaussian_noise`;
- `powerline_interference`;
- `baseline_wander`;
- `amplitude_scaling`;
- `clipping_saturation`;
- `time_scaling`.

没有把 `shuffled` 这类明显破坏 ECG 时间顺序的扰动纳入主测试。当前 severity 为 1 和 2，属于 pilot 级别。`clipping_saturation` severity 2 对波形破坏较强，应单独解释为强记录伪影压力场景。

## 3. 测试设计

固定 clean validation 上的路由设计，然后在 shifted ECG test 上评估：

```text
E[Y | do(complete_router = p), shift_condition = s]
```

pilot 规模：

| Item | Value |
| --- | ---: |
| Seeds | 42, 43, 44 |
| Budgets | 10%, 20%, 30% |
| ECG shifts | 6 |
| Severities | 1, 2 |
| Seed-level rows | 819 |
| Summary rows | 273 |

比较对象：

- `v5d_reserve_0pct`;
- `v5d_reserve_20pct`;
- `v5d_reserve_30pct`;
- `complete_router_profile_wavelet_boundary_heavy`；
- `complete_router_profile_validity_boundary__sr_ventricular`；
- `complete_router_profile_wavelet_boundary__atypical_signal`；
- `entropy_ranked_review`.

## 4. 动态与静态证据

脚本在 shifted ECG 上重新计算：

- shifted model prediction；
- shifted softmax / entropy / VT-VF ambiguity；
- shifted wavelet VT/VF boundary risk；
- shifted error labels and VT/VF cross-error labels。

当前 pilot 的限制：

- validity gate 暂时沿用 clean evidence；
- residual mechanism risk heads 暂时沿用 clean evidence；
- 因此这是 route-level OOD stress pilot，不是最终完整 OOD mechanism retraining。

## 5. 20% budget clean-set 对照

三 seed clean condition 中：

| Complete policy | VT/VF capture | All-error capture | Auto VT/VF error rate | Auto error rate |
| --- | ---: | ---: | ---: | ---: |
| `v5d_reserve_0pct` | 100.00% | 89.99% | 0.000% | 0.757% |
| `v5d_reserve_20pct` | 100.00% | 92.49% | 0.000% | 0.566% |
| `complete_router_profile_wavelet_boundary_heavy` | 99.83% | 90.51% | 0.008% | 0.717% |
| `v5d_reserve_30pct` | 99.83% | 90.72% | 0.008% | 0.701% |
| `complete_router_profile_validity_boundary__sr_ventricular` | 99.48% | 89.99% | 0.024% | 0.757% |
| `complete_router_profile_wavelet_boundary__atypical_signal` | 98.96% | 90.80% | 0.048% | 0.695% |
| `entropy_ranked_review` | 93.59% | 87.86% | 0.292% | 0.890% |

clean-set 上 V5D 仍是很强的完整路由 baseline，特别是 `reserve_20pct`。

## 6. 20% budget OOD shift 平均表现

六类 ECG shift、severity 1-2 平均后：

| Complete policy | VT/VF capture | All-error capture | Auto VT/VF error rate | Auto error rate |
| --- | ---: | ---: | ---: | ---: |
| `v5d_reserve_0pct` | 99.47% | 79.49% | 0.023% | 7.624% |
| `complete_router_profile_validity_boundary__sr_ventricular` | 99.43% | 71.67% | 0.025% | 9.193% |
| `v5d_reserve_20pct` | 99.39% | 75.09% | 0.033% | 8.596% |
| `v5d_reserve_30pct` | 99.08% | 76.85% | 0.051% | 8.335% |
| `complete_router_profile_wavelet_boundary_heavy` | 97.05% | 77.80% | 0.150% | 8.029% |
| `complete_router_profile_wavelet_boundary__atypical_signal` | 95.01% | 77.40% | 0.236% | 8.165% |
| `entropy_ranked_review` | 73.40% | 72.24% | 1.155% | 8.535% |

解释：

- V5D 在 shifted ECG 下仍然最强地保护 VT/VF capture；
- wavelet-heavy complete router 在 all-error capture 上有竞争力，但强波形扰动下 VT/VF capture 掉得更多；
- entropy baseline 明显更弱，说明单纯 uncertainty ranking 不够。

## 7. 去掉强 clipping 后的 ECG-preserving shift

如果去掉 `clipping_saturation`，只看较温和的 ECG-preserving shifts，20% budget 平均为：

| Complete policy | VT/VF capture | All-error capture | Auto VT/VF error rate | Auto error rate |
| --- | ---: | ---: | ---: | ---: |
| `v5d_reserve_0pct` | 100.00% | 89.86% | 0.000% | 1.115% |
| `v5d_reserve_20pct` | 99.66% | 85.50% | 0.018% | 1.821% |
| `complete_router_profile_wavelet_boundary_heavy` | 99.57% | 87.79% | 0.017% | 1.472% |
| `complete_router_profile_validity_boundary__sr_ventricular` | 99.41% | 81.97% | 0.024% | 2.216% |
| `v5d_reserve_30pct` | 99.38% | 87.71% | 0.032% | 1.421% |
| `complete_router_profile_wavelet_boundary__atypical_signal` | 97.90% | 87.65% | 0.089% | 1.484% |
| `entropy_ranked_review` | 86.65% | 81.84% | 0.566% | 1.932% |

这说明在更温和的 ECG shift 下：

- V5D 的 VT/VF 防线非常稳；
- wavelet-heavy complete router 的整体错误覆盖和 auto error 也有竞争力；
- 但还不能说 causal-Pareto complete router 全面超过 V5D。

## 8. 对“用自己的钥匙开自己的锁”的回应

当前 pilot 的结论更谨慎：

1. V5D 没有在 ECG shift 下立刻崩溃，说明它不是纯 clean-set artifact；
2. V5D 的强项主要是 VT/VF boundary 防守；
3. causal-Pareto complete routers 展示了其他目标上的可解释 trade-off，但不是全面胜出；
4. entropy baseline 在 shift 下明显不稳，支持机制化路由优于单一 uncertainty ranking。

因此当前不能写：

```text
causal-Pareto 全面优于 V5D。
```

更合适的写法：

```text
OOD-style ECG shift stress shows that V5D remains a strong VT/VF-protective hierarchical baseline, while complete causal-Pareto routing profiles expose alternative trade-offs that can improve overall error coverage under certain shift and budget regimes.
```

## 9. 下一步

- 跑 10-seed full OOD route stress；
- 在 shifted ECG 上重算 validity gate；
- 在 shifted ECG 上重算 residual mechanism heads；
- 分开报告 mild ECG shift 与 strong artifact shift；
- 增加 route composition stability，检查不同 shift 下到底哪条 recover action 在捕获错误。
