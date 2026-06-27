from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = list(range(42, 52))


def _top_budget_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = max(1, int(round(len(score) * budget)))
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n]] = True
    return mask


def _score_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {
        "entropy": df["entropy"].to_numpy(float),
        "msp_uncertainty": df["msp_uncertainty"].to_numpy(float),
        "low_softmax_margin": 1.0 - df["rank_margin"].to_numpy(float),
        "softmax_vtvf_ambiguity": df["softmax_vtvf_ambiguity"].to_numpy(float),
        "prototype_vtvf_ambiguity": df["proto_vtvf_ambiguity"].to_numpy(float),
        "knn_vtvf_mixing": df["knn_vtvf_mixing"].to_numpy(float),
        "knn_label_entropy": df["knn_label_entropy"].to_numpy(float),
        "latent_cluster_error_rate": df["latent_cluster_val_error_rate"].to_numpy(float),
        "latent_cluster_vtvf_cross_rate": df["latent_cluster_val_vtvf_cross_rate"].to_numpy(float),
        "model_disagreement": df["model_disagreement"].to_numpy(float)
        if "model_disagreement" in df.columns
        else np.zeros(len(df), dtype=float),
    }
    conformal_cols = [c for c in df.columns if c.startswith("conformal_set_size_")]
    if conformal_cols:
        scores["conformal_max_set_size"] = df[conformal_cols].max(axis=1).to_numpy(float)
        scores["conformal_mean_set_size"] = df[conformal_cols].mean(axis=1).to_numpy(float)
    if "any_error_risk" in df.columns:
        scores["learned_any_error_risk"] = df["any_error_risk"].to_numpy(float)
    if "vtvf_boundary_risk" in df.columns:
        scores["learned_vtvf_boundary_risk"] = df["vtvf_boundary_risk"].to_numpy(float)
    return scores


def _ranked_review_rows(seed: int, df: pd.DataFrame, budgets: list[float]) -> list[dict[str, float | int | str]]:
    errors = df["is_error"].to_numpy(bool)
    vtvf = df["is_vtvf_cross_error"].to_numpy(bool)
    rows: list[dict[str, float | int | str]] = []
    for method, score in _score_columns(df).items():
        for budget in budgets:
            mask = _top_budget_mask(score, budget)
            auto = ~mask
            rows.append(
                {
                    "seed": seed,
                    "budget": budget,
                    "method": method,
                    "policy_family": "ranked_review_score",
                    "action_rate": float(mask.mean()),
                    "all_error_addressed": float((errors & mask).sum() / max(errors.sum(), 1)),
                    "vtvf_cross_error_addressed": float((vtvf & mask).sum() / max(vtvf.sum(), 1)),
                    "automatic_unresolved_error_rate": float((errors & auto).mean()) if auto.any() else np.nan,
                    "automatic_unresolved_vtvf_cross_error_rate": float((vtvf & auto).mean()) if auto.any() else np.nan,
                }
            )
    return rows


def _policy_summary_rows(seed: int, seed_dir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    specs = [
        (
            "learned_total_risk_layered",
            "layered_policy_summary.csv",
            "total_risk_layered",
            {
                "action_rate": lambda r: float(r["review_rate"] + r["vtvf_set_rate"]),
                "all_error_addressed": "all_error_addressed_by_review_or_set",
                "vtvf_cross_error_addressed": "vtvf_cross_error_addressed_by_review_or_set",
                "automatic_unresolved_error_rate": "automatic_unresolved_error_rate",
                "automatic_unresolved_vtvf_cross_error_rate": "automatic_unresolved_vtvf_cross_error_rate",
            },
        ),
        (
            "fixed_mechanism_router_v3b",
            "mechanism_layered_policy_summary.csv",
            "mechanism_router",
            {
                "action_rate": "mechanism_action_rate",
                "all_error_addressed": "all_error_addressed",
                "vtvf_cross_error_addressed": "vtvf_cross_error_addressed",
                "automatic_unresolved_error_rate": "automatic_unresolved_error_rate",
                "automatic_unresolved_vtvf_cross_error_rate": "automatic_unresolved_vtvf_cross_error_rate",
            },
        ),
        (
            "optimized_mechanism_router_v4",
            "optimized_mechanism_layered_policy_summary.csv",
            "mechanism_router",
            {
                "action_rate": "mechanism_action_rate",
                "all_error_addressed": "all_error_addressed",
                "vtvf_cross_error_addressed": "vtvf_cross_error_addressed",
                "automatic_unresolved_error_rate": "automatic_unresolved_error_rate",
                "automatic_unresolved_vtvf_cross_error_rate": "automatic_unresolved_vtvf_cross_error_rate",
            },
        ),
    ]
    for method, filename, family, mapping in specs:
        path = seed_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            out: dict[str, float | int | str] = {
                "seed": seed,
                "budget": float(row["budget"]),
                "method": method,
                "policy_family": family,
            }
            for out_col, source in mapping.items():
                if callable(source):
                    out[out_col] = source(row)
                else:
                    out[out_col] = float(row[source])
            rows.append(out)
    return rows


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    out_dir = args.out or (args.routing_dir / "baseline_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        seed_dir = args.routing_dir / f"seed{seed}"
        evidence_path = seed_dir / "evidence_scores_test.csv"
        if not evidence_path.exists():
            continue
        evidence = pd.read_csv(evidence_path)
        rows.extend(_ranked_review_rows(seed, evidence, args.budgets))
        rows.extend(_policy_summary_rows(seed, seed_dir))

    all_df = pd.DataFrame(rows)
    all_df.to_csv(out_dir / "all_seed_routing_baseline_comparison.csv", index=False)
    metric_cols = [
        "action_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
    ]
    summary = _mean_std(all_df, ["method", "policy_family", "budget"], metric_cols)
    summary.to_csv(out_dir / "routing_baseline_mean_std.csv", index=False)

    opt = all_df[all_df["method"].eq("optimized_mechanism_router_v4")]
    paired_rows = []
    for method in sorted(set(all_df["method"]) - {"optimized_mechanism_router_v4"}):
        other = all_df[all_df["method"].eq(method)]
        merged = opt.merge(other, on=["seed", "budget"], suffixes=("_opt", "_baseline"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "baseline_method": method,
                    "delta_all_error_addressed_opt_minus_baseline": row["all_error_addressed_opt"]
                    - row["all_error_addressed_baseline"],
                    "delta_vtvf_cross_error_addressed_opt_minus_baseline": row["vtvf_cross_error_addressed_opt"]
                    - row["vtvf_cross_error_addressed_baseline"],
                    "delta_unresolved_vtvf_opt_minus_baseline": row[
                        "automatic_unresolved_vtvf_cross_error_rate_opt"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_baseline"],
                    "delta_action_rate_opt_minus_baseline": row["action_rate_opt"] - row["action_rate_baseline"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_optimized_mechanism_vs_baselines.csv", index=False)
    if not paired.empty:
        paired_summary = _mean_std(
            paired.rename(columns={"baseline_method": "method"}),
            ["method", "budget"],
            [
                "delta_all_error_addressed_opt_minus_baseline",
                "delta_vtvf_cross_error_addressed_opt_minus_baseline",
                "delta_unresolved_vtvf_opt_minus_baseline",
                "delta_action_rate_opt_minus_baseline",
            ],
        )
        paired_summary.to_csv(out_dir / "paired_optimized_mechanism_vs_baselines_mean_std.csv", index=False)
    print(summary.sort_values(["budget", "vtvf_cross_error_addressed_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
