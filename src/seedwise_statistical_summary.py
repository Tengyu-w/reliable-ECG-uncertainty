from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _paired_summary(df: pd.DataFrame, baseline: str, comparator: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    base = df[df["variant"].eq(baseline)].set_index("seed")
    comp = df[df["variant"].eq(comparator)].set_index("seed")
    seeds = sorted(set(base.index).intersection(set(comp.index)))
    for metric in metrics:
        b = base.loc[seeds, metric].astype(float).to_numpy()
        c = comp.loc[seeds, metric].astype(float).to_numpy()
        diff = c - b
        if len(diff) >= 2:
            sem = stats.sem(diff)
            ci = stats.t.interval(0.95, len(diff) - 1, loc=float(np.mean(diff)), scale=sem) if sem > 0 else (float(np.mean(diff)), float(np.mean(diff)))
            p = stats.ttest_rel(c, b).pvalue
        else:
            ci = (float("nan"), float("nan"))
            p = float("nan")
        rows.append(
            {
                "comparison": f"{comparator}_minus_{baseline}",
                "metric": metric,
                "n_paired_seeds": len(seeds),
                "baseline_mean": float(np.mean(b)),
                "comparator_mean": float(np.mean(c)),
                "mean_difference": float(np.mean(diff)),
                "difference_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "paired_t_p_value": float(p),
                "seed_differences": ";".join(f"{seed}:{d:.6g}" for seed, d in zip(seeds, diff)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/mitigation_v3_key_ablation_summary_full_analysis/mitigation_run_level_metrics.csv"))
    parser.add_argument("--routing", type=Path, default=Path("results/mitigation_v3_key_ablation_summary_full_analysis/mitigation_routing_run_level.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/statistical_summary_20260609"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    routing = pd.read_csv(args.routing)
    metrics = metrics[metrics["variant"].isin(["baseline", "prototype_separation", "full_supervisor"])].copy()
    metric_cols = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors"]
    seedwise = metrics.sort_values(["seed", "variant"])[["seed", "variant", *metric_cols]]
    seedwise.to_csv(args.out / "classification_seedwise_table.csv", index=False)

    paired = pd.concat(
        [
            _paired_summary(metrics, "baseline", "prototype_separation", metric_cols),
            _paired_summary(metrics, "baseline", "full_supervisor", metric_cols),
        ],
        ignore_index=True,
    )
    paired.to_csv(args.out / "classification_paired_comparisons.csv", index=False)

    auto = routing[routing["decision"].eq("automatic_single_label")].copy()
    auto = auto[auto["variant"].isin(["baseline", "prototype_separation", "full_supervisor"])]
    auto_cols = ["rate", "error_rate", "vtvf_cross_error_rate", "vtvf_cross_errors"]
    auto_seedwise = auto.sort_values(["seed", "variant"])[["seed", "variant", *auto_cols]]
    auto_seedwise.to_csv(args.out / "automatic_route_seedwise_table.csv", index=False)
    auto_paired = pd.concat(
        [
            _paired_summary(auto, "baseline", "prototype_separation", auto_cols),
            _paired_summary(auto, "baseline", "full_supervisor", auto_cols),
        ],
        ignore_index=True,
    )
    auto_paired.to_csv(args.out / "automatic_route_paired_comparisons.csv", index=False)

    md = args.out / "statistical_summary_cn.md"
    lines = [
        "# Seed-wise and paired statistical summary",
        "",
        "说明：只有 3 个 paired seeds，因此 p-value 只能作为辅助描述，不能过度解释。更适合汇报 mean difference、seed-wise direction 和 95% CI。",
        "",
        "## Classification paired comparisons",
        "",
        paired.to_markdown(index=False),
        "",
        "## Automatic route paired comparisons",
        "",
        auto_paired.to_markdown(index=False),
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", args.out)
    print(paired)
    print(auto_paired)


if __name__ == "__main__":
    main()
