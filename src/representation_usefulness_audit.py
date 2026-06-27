from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evidence_informed_recovery_routing import (
    _feature_columns,
    _mechanism_feature_specs,
    _optimized_mechanism_policy_summary,
)


SEEDS = list(range(42, 52))
REPRESENTATION_PREFIXES = (
    "proto_",
    "nearest_",
    "min_proto",
    "knn_",
    "abs_proto",
    "prior_calibration_d_",
    "prior_calibration_nearest_",
    "prior_calibration_knn_",
    "prior_calibration_proto_",
)


class ConstantRiskModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        p = np.full(len(features), self.value, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def _fit_binary_risk_model(x: pd.DataFrame, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        return ConstantRiskModel(float(y.mean()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(x, y)
    return model


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _is_representation_feature(col: str) -> bool:
    return col.startswith(REPRESENTATION_PREFIXES)


def _variant_features(features: list[str], variant: str) -> list[str]:
    if variant == "full":
        return features
    if variant == "no_representation":
        return [col for col in features if not _is_representation_feature(col)]
    if variant == "representation_only":
        return [col for col in features if _is_representation_feature(col)]
    raise ValueError(f"Unknown variant: {variant}")


def _fit_variant_heads(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    specs = _mechanism_feature_specs(val_df)
    val_out = val_df.copy()
    test_out = test_df.copy()
    rows = []
    fitted_specs: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        features = _variant_features(spec["features"], variant)
        if not features:
            continue
        target = spec["target"]
        model = _fit_binary_risk_model(val_out[features], val_out[target].to_numpy(int))
        val_score = model.predict_proba(val_out[features])[:, 1]
        test_score = model.predict_proba(test_out[features])[:, 1]
        score_col = f"{name}_mechanism_risk"
        val_out[score_col] = val_score
        test_out[score_col] = test_score
        val_positive = int(val_out[target].sum())
        test_positive = int(test_out[target].sum())
        fitted_specs[name] = {
            **spec,
            "features": features,
            "score_col": score_col,
            "n_features": len(features),
            "val_positive": val_positive,
            "test_positive": test_positive,
            "enabled_for_routing": val_positive >= 5,
        }
        rows.append(
            {
                "variant": variant,
                "mechanism": name,
                "target": target,
                "n_features": len(features),
                "val_positive": val_positive,
                "test_positive": test_positive,
                "enabled_for_routing": val_positive >= 5,
                "test_auroc": _safe_auc(test_out[target].to_numpy(int), test_score),
                "test_aupr": _safe_aupr(test_out[target].to_numpy(int), test_score),
            }
        )
    return val_out, test_out, pd.DataFrame(rows), fitted_specs


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _run_seed(seed_dir: Path, seed: int, budgets: list[float], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = pd.read_csv(seed_dir / "evidence_scores_val.csv")
    test_df = pd.read_csv(seed_dir / "evidence_scores_test.csv")
    head_rows = []
    policy_rows = []
    for variant in ["full", "no_representation", "representation_only"]:
        val_variant, test_variant, head_df, specs = _fit_variant_heads(val_df, test_df, variant)
        policy_df, _, _ = _optimized_mechanism_policy_summary(val_variant, test_variant, specs, budgets)
        head_df["seed"] = seed
        policy_df["seed"] = seed
        policy_df["variant"] = variant
        head_rows.append(head_df)
        policy_rows.append(policy_df)
    seed_heads = pd.concat(head_rows, ignore_index=True)
    seed_policy = pd.concat(policy_rows, ignore_index=True)
    seed_heads.to_csv(out_dir / f"seed{seed}_representation_variant_heads.csv", index=False)
    seed_policy.to_csv(out_dir / f"seed{seed}_representation_variant_policy.csv", index=False)
    return seed_heads, seed_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    out_dir = args.out or (args.routing_dir / "representation_usefulness_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_heads = []
    all_policy = []
    for seed in SEEDS:
        seed_dir = args.routing_dir / f"seed{seed}"
        if not (seed_dir / "evidence_scores_val.csv").exists():
            continue
        heads, policy = _run_seed(seed_dir, seed, args.budgets, out_dir)
        all_heads.append(heads)
        all_policy.append(policy)
        print(f"completed seed{seed}")

    heads_df = pd.concat(all_heads, ignore_index=True)
    policy_df = pd.concat(all_policy, ignore_index=True)
    heads_df.to_csv(out_dir / "all_seed_representation_variant_heads.csv", index=False)
    policy_df.to_csv(out_dir / "all_seed_representation_variant_policy.csv", index=False)

    _mean_std(
        heads_df,
        ["variant", "mechanism"],
        ["n_features", "test_auroc", "test_aupr"],
    ).to_csv(out_dir / "representation_variant_head_mean_std.csv", index=False)
    _mean_std(
        policy_df,
        ["variant", "budget"],
        [
            "mechanism_action_rate",
            "all_error_addressed",
            "vtvf_cross_error_addressed",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
        ],
    ).to_csv(out_dir / "representation_variant_policy_mean_std.csv", index=False)

    paired_rows = []
    full = policy_df[policy_df["variant"].eq("full")]
    for variant in ["no_representation", "representation_only"]:
        other = policy_df[policy_df["variant"].eq(variant)]
        merged = full.merge(other, on=["seed", "budget"], suffixes=("_full", "_variant"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "variant": variant,
                    "delta_all_error_addressed_full_minus_variant": row["all_error_addressed_full"]
                    - row["all_error_addressed_variant"],
                    "delta_vtvf_cross_error_addressed_full_minus_variant": row["vtvf_cross_error_addressed_full"]
                    - row["vtvf_cross_error_addressed_variant"],
                    "delta_unresolved_vtvf_full_minus_variant": row[
                        "automatic_unresolved_vtvf_cross_error_rate_full"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_variant"],
                    "delta_action_rate_full_minus_variant": row["mechanism_action_rate_full"]
                    - row["mechanism_action_rate_variant"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_full_vs_representation_variants.csv", index=False)
    _mean_std(
        paired,
        ["variant", "budget"],
        [
            "delta_all_error_addressed_full_minus_variant",
            "delta_vtvf_cross_error_addressed_full_minus_variant",
            "delta_unresolved_vtvf_full_minus_variant",
            "delta_action_rate_full_minus_variant",
        ],
    ).to_csv(out_dir / "paired_full_vs_representation_variants_mean_std.csv", index=False)
    print(pd.read_csv(out_dir / "representation_variant_policy_mean_std.csv").to_string(index=False))


if __name__ == "__main__":
    main()
