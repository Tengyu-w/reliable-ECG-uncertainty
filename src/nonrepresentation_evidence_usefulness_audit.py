from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evidence_informed_recovery_routing import (
    _mechanism_feature_specs,
    _optimized_mechanism_policy_summary,
)


SEEDS = list(range(42, 52))


def _starts(col: str, prefixes: tuple[str, ...]) -> bool:
    return col.startswith(prefixes)


def _contains(col: str, tokens: tuple[str, ...]) -> bool:
    return any(token in col for token in tokens)


def _regularity(col: str) -> bool:
    return _starts(col, ("regularity_",))


def _risk_target(col: str) -> bool:
    return _starts(col, ("risk_target_",))


def _softmax_boundary(col: str) -> bool:
    prefixes = (
        "prob_",
        "temperature_prob_",
        "max_prob",
        "temperature_max_prob",
        "msp_",
        "entropy",
        "temperature_entropy",
        "rank_margin",
        "ventricular_",
        "softmax_",
        "abs_prob_",
        "abs_logit_",
        "pred_is_",
        "top2_",
        "prior_calibration_prob_",
        "prior_calibration_temperature_prob_",
        "prior_calibration_max_prob",
        "prior_calibration_temperature_max_prob",
        "prior_calibration_msp_",
        "prior_calibration_entropy",
        "prior_calibration_temperature_entropy",
        "prior_calibration_ventricular_",
        "prior_calibration_softmax_",
        "prior_calibration_abs_prob_",
        "prior_calibration_abs_logit_",
        "prior_calibration_pred_is_",
        "prior_calibration_top2_",
    )
    return _starts(col, prefixes)


def _local_instability(col: str) -> bool:
    return _starts(col, ("knn_", "prior_calibration_knn_")) or col in {
        "risk_target_knn",
        "risk_target_local_instability",
        "risk_target_vtvf_mixing",
    }


def _latent_cluster(col: str) -> bool:
    return _starts(col, ("latent_cluster",))


def _model_disagreement(col: str) -> bool:
    return _starts(col, ("second_", "model_disagreement"))


def _historical_diagnostics(col: str) -> bool:
    return _starts(col, ("risk_target_", "prior_calibration_"))


def _prototype_geometry(col: str) -> bool:
    return _starts(
        col,
        (
            "proto_",
            "nearest_",
            "min_proto",
            "abs_proto",
            "prior_calibration_d_",
            "prior_calibration_nearest_",
        ),
    ) or col == "risk_target_prototype"


FAMILY_PREDICATES: dict[str, Callable[[str], bool]] = {
    "regularity": _regularity,
    "risk_target_components": _risk_target,
    "softmax_boundary": _softmax_boundary,
    "local_instability_knn": _local_instability,
    "latent_cluster": _latent_cluster,
    "model_disagreement": _model_disagreement,
    "historical_diagnostics": _historical_diagnostics,
    "prototype_geometry": _prototype_geometry,
}


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


def _variant_features(features: list[str], family: str, mode: str) -> list[str]:
    predicate = FAMILY_PREDICATES[family]
    if mode == "full":
        return features
    if mode == "without":
        return [col for col in features if not predicate(col)]
    if mode == "only":
        return [col for col in features if predicate(col)]
    raise ValueError(f"Unknown mode: {mode}")


def _fit_variant_heads(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    family: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    specs = _mechanism_feature_specs(val_df)
    val_out = val_df.copy()
    test_out = test_df.copy()
    rows = []
    fitted_specs: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        features = _variant_features(spec["features"], family, mode)
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
                "family": family,
                "mode": mode,
                "variant": f"{mode}_{family}" if mode != "full" else "full",
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


def _run_seed(seed_dir: Path, seed: int, families: list[str], budgets: list[float], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = pd.read_csv(seed_dir / "evidence_scores_val.csv")
    test_df = pd.read_csv(seed_dir / "evidence_scores_test.csv")
    head_rows = []
    policy_rows = []
    for family in families:
        for mode in ["full", "without", "only"]:
            if mode == "full" and family != families[0]:
                continue
            val_variant, test_variant, head_df, specs = _fit_variant_heads(val_df, test_df, family, mode)
            policy_df, _, _ = _optimized_mechanism_policy_summary(val_variant, test_variant, specs, budgets)
            variant = "full" if mode == "full" else f"{mode}_{family}"
            head_df["seed"] = seed
            policy_df["seed"] = seed
            policy_df["family"] = family
            policy_df["mode"] = mode
            policy_df["variant"] = variant
            head_rows.append(head_df)
            policy_rows.append(policy_df)
    seed_heads = pd.concat(head_rows, ignore_index=True)
    seed_policy = pd.concat(policy_rows, ignore_index=True)
    seed_heads.to_csv(out_dir / f"seed{seed}_evidence_family_heads.csv", index=False)
    seed_policy.to_csv(out_dir / f"seed{seed}_evidence_family_policy.csv", index=False)
    return seed_heads, seed_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--families", nargs="+", default=list(FAMILY_PREDICATES))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    out_dir = args.out or (args.routing_dir / "nonrepresentation_evidence_usefulness_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    families = [family for family in args.families if family in FAMILY_PREDICATES]
    if not families:
        raise ValueError("No valid evidence families selected.")

    all_heads = []
    all_policy = []
    for seed in SEEDS:
        seed_dir = args.routing_dir / f"seed{seed}"
        if not (seed_dir / "evidence_scores_val.csv").exists():
            continue
        heads, policy = _run_seed(seed_dir, seed, families, args.budgets, out_dir)
        all_heads.append(heads)
        all_policy.append(policy)
        print(f"completed seed{seed}")

    heads_df = pd.concat(all_heads, ignore_index=True)
    policy_df = pd.concat(all_policy, ignore_index=True)
    heads_df.to_csv(out_dir / "all_seed_evidence_family_heads.csv", index=False)
    policy_df.to_csv(out_dir / "all_seed_evidence_family_policy.csv", index=False)

    _mean_std(
        heads_df,
        ["variant", "family", "mode", "mechanism"],
        ["n_features", "test_auroc", "test_aupr"],
    ).to_csv(out_dir / "evidence_family_head_mean_std.csv", index=False)
    _mean_std(
        policy_df,
        ["variant", "family", "mode", "budget"],
        [
            "mechanism_action_rate",
            "all_error_addressed",
            "vtvf_cross_error_addressed",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
        ],
    ).to_csv(out_dir / "evidence_family_policy_mean_std.csv", index=False)

    full = policy_df[policy_df["variant"].eq("full")]
    paired_rows = []
    for variant in sorted(set(policy_df["variant"]) - {"full"}):
        other = policy_df[policy_df["variant"].eq(variant)]
        merged = full.merge(other, on=["seed", "budget"], suffixes=("_full", "_variant"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "variant": variant,
                    "family": str(row["family_variant"]),
                    "mode": str(row["mode_variant"]),
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
    paired.to_csv(out_dir / "paired_full_vs_evidence_family_variants.csv", index=False)
    _mean_std(
        paired,
        ["variant", "family", "mode", "budget"],
        [
            "delta_all_error_addressed_full_minus_variant",
            "delta_vtvf_cross_error_addressed_full_minus_variant",
            "delta_unresolved_vtvf_full_minus_variant",
            "delta_action_rate_full_minus_variant",
        ],
    ).to_csv(out_dir / "paired_full_vs_evidence_family_variants_mean_std.csv", index=False)

    print(pd.read_csv(out_dir / "evidence_family_policy_mean_std.csv").to_string(index=False))


if __name__ == "__main__":
    main()
