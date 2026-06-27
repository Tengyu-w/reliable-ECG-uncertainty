from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_duplicate_family_groups, load_rhythm_windows, make_splits
from .models import FixedWaveletFilterBank1D


SEEDS = list(range(42, 52))
CLASS_LABELS = {0: "SR", 1: "VT", 2: "VF"}
WAVELET_FEATURE_NAMES: list[str] = []
for scale in ["s2", "s4", "s8"]:
    for atom in ["mexican_hat", "slope", "oscillation"]:
        for stat in ["mean", "std", "p95", "max", "entropy"]:
            WAVELET_FEATURE_NAMES.append(f"wavelet_{scale}_{atom}_{stat}")
for scale in ["s2", "s4", "s8"]:
    WAVELET_FEATURE_NAMES.append(f"wavelet_{scale}_osc_to_shape_energy")
    WAVELET_FEATURE_NAMES.append(f"wavelet_{scale}_slope_to_shape_energy")
WAVELET_FEATURE_NAMES.extend(
    [
        "wavelet_fine_to_coarse_energy",
        "wavelet_mid_to_coarse_energy",
        "wavelet_oscillation_fraction",
        "wavelet_slope_fraction",
        "wavelet_shape_fraction",
    ]
)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _top_budget_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = max(1, int(round(len(score) * budget)))
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n]] = True
    return mask


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


def _extract_wavelet_features(x: np.ndarray, batch_size: int = 512) -> pd.DataFrame:
    bank = FixedWaveletFilterBank1D()
    bank.eval()
    rows: list[np.ndarray] = []
    eps = 1e-8
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float()
            response = bank(xb).cpu().numpy()
            flat = response.reshape(response.shape[0], response.shape[1], -1)
            mean = flat.mean(axis=2)
            std = flat.std(axis=2)
            p95 = np.percentile(flat, 95, axis=2)
            maxv = flat.max(axis=2)
            mass = flat / np.maximum(flat.sum(axis=2, keepdims=True), eps)
            entropy = -np.sum(mass * np.log(np.maximum(mass, eps)), axis=2) / np.log(flat.shape[2])
            stats = np.stack([mean, std, p95, maxv, entropy], axis=2).reshape(len(flat), -1)

            energy = np.square(flat).mean(axis=2)
            shaped = energy.reshape(len(flat), 3, 3)
            derived = []
            for scale_idx in range(3):
                shape_energy = shaped[:, scale_idx, 0]
                slope_energy = shaped[:, scale_idx, 1]
                osc_energy = shaped[:, scale_idx, 2]
                derived.append(osc_energy / np.maximum(shape_energy, eps))
                derived.append(slope_energy / np.maximum(shape_energy, eps))
            scale_energy = shaped.sum(axis=2)
            atom_energy = shaped.sum(axis=1)
            total = np.maximum(atom_energy.sum(axis=1), eps)
            derived.extend(
                [
                    scale_energy[:, 0] / np.maximum(scale_energy[:, 2], eps),
                    scale_energy[:, 1] / np.maximum(scale_energy[:, 2], eps),
                    atom_energy[:, 2] / total,
                    atom_energy[:, 1] / total,
                    atom_energy[:, 0] / total,
                ]
            )
            rows.append(np.concatenate([stats, np.stack(derived, axis=1)], axis=1))
    return pd.DataFrame(np.concatenate(rows, axis=0).astype(np.float32), columns=WAVELET_FEATURE_NAMES)


def _reconstruct_split(seed: int, mat: Path, max_windows_per_record: int | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    dataset = load_rhythm_windows(mat, max_windows_per_record=max_windows_per_record)
    groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    splits = make_splits(dataset.x, dataset.y, groups=groups, seed=seed)
    return {
        "val": (splits.x_val, splits.y_val),
        "test": (splits.x_test, splits.y_test),
    }


def _fit_binary_risk(x: pd.DataFrame, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        value = float(y.mean())

        class ConstantModel:
            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                p = np.full(len(features), value, dtype=np.float32)
                return np.stack([1.0 - p, p], axis=1)

        return ConstantModel()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"))
    model.fit(x, y)
    return model


def _add_wavelet_risk_heads(val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    targets = {
        "wavelet_any_error_risk": "is_error",
        "wavelet_vtvf_boundary_risk": "is_vtvf_cross_error",
    }
    for score_col, target in targets.items():
        model = _fit_binary_risk(val_df[WAVELET_FEATURE_NAMES], val_df[target].to_numpy(int))
        val_score = model.predict_proba(val_df[WAVELET_FEATURE_NAMES])[:, 1]
        test_score = model.predict_proba(test_df[WAVELET_FEATURE_NAMES])[:, 1]
        val_df[score_col] = val_score
        test_df[score_col] = test_score
        rows.append(
            {
                "score": score_col,
                "target": target,
                "n_features": len(WAVELET_FEATURE_NAMES),
                "val_positive": int(val_df[target].sum()),
                "test_positive": int(test_df[target].sum()),
                "test_auroc": _safe_auc(test_df[target].to_numpy(int), test_score),
                "test_aupr": _safe_aupr(test_df[target].to_numpy(int), test_score),
            }
        )
    return val_df, test_df, pd.DataFrame(rows)


def _ranked_review_rows(seed: int, df: pd.DataFrame, budgets: list[float]) -> list[dict[str, float | int | str]]:
    errors = df["is_error"].to_numpy(bool)
    vtvf = df["is_vtvf_cross_error"].to_numpy(bool)
    scores = {
        "wavelet_any_error_risk": df["wavelet_any_error_risk"].to_numpy(float),
        "wavelet_vtvf_boundary_risk": df["wavelet_vtvf_boundary_risk"].to_numpy(float),
        "wavelet_fine_to_coarse_energy": df["wavelet_fine_to_coarse_energy"].to_numpy(float),
        "wavelet_oscillation_fraction": df["wavelet_oscillation_fraction"].to_numpy(float),
        "wavelet_shape_fraction": df["wavelet_shape_fraction"].to_numpy(float),
    }
    rows = []
    for method, score in scores.items():
        for budget in budgets:
            mask = _top_budget_mask(score, budget)
            auto = ~mask
            rows.append(
                {
                    "seed": seed,
                    "budget": budget,
                    "method": method,
                    "policy_family": "wavelet_ranked_review_score",
                    "action_rate": float(mask.mean()),
                    "all_error_addressed": float((errors & mask).sum() / max(errors.sum(), 1)),
                    "vtvf_cross_error_addressed": float((vtvf & mask).sum() / max(vtvf.sum(), 1)),
                    "automatic_unresolved_error_rate": float((errors & auto).mean()) if auto.any() else np.nan,
                    "automatic_unresolved_vtvf_cross_error_rate": float((vtvf & auto).mean()) if auto.any() else np.nan,
                }
            )
    return rows


def _base_specs(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs = {
        "vtvf_boundary": {
            "score_col": "vtvf_boundary_mechanism_risk",
            "target": "is_vtvf_cross_error",
            "action": "vtvf_boundary_set",
        },
        "sr_ventricular": {
            "score_col": "sr_ventricular_mechanism_risk",
            "target": "is_sr_ventricular_error",
            "action": "sr_ventricular_review",
        },
        "representation_conflict": {
            "score_col": "representation_conflict_mechanism_risk",
            "target": "is_representation_conflict_error",
            "action": "representation_review",
        },
        "atypical_signal": {
            "score_col": "atypical_signal_mechanism_risk",
            "target": "is_atypical_signal_error",
            "action": "atypical_review",
        },
        "wavelet_boundary": {
            "score_col": "wavelet_vtvf_boundary_risk",
            "target": "is_vtvf_cross_error",
            "action": "vtvf_boundary_set",
        },
    }
    return {name: spec for name, spec in specs.items() if spec["score_col"] in df.columns and spec["target"] in df.columns}


def _candidate_mask(df: pd.DataFrame, val_df: pd.DataFrame, name: str) -> np.ndarray:
    if name in {"vtvf_boundary", "wavelet_boundary"} and "is_vtvf_candidate" in df.columns:
        return df["is_vtvf_candidate"].to_numpy(bool)
    return np.ones(len(df), dtype=bool)


def _candidate_tuples(
    df: pd.DataFrame,
    val_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> list[tuple[float, int, str]]:
    candidates = []
    for name, spec in specs.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        mask = _candidate_mask(df, val_df, name)
        scores = df[spec["score_col"]].to_numpy(float) * weight
        for idx in np.flatnonzero(mask):
            candidates.append((float(scores[idx]), int(idx), name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _greedy_unique_selection(candidates: list[tuple[float, int, str]], n_select: int) -> list[tuple[int, str, float]]:
    selected = []
    used = set()
    for score, idx, name in candidates:
        if idx in used:
            continue
        used.add(idx)
        selected.append((idx, name, score))
        if len(selected) >= n_select:
            break
    return selected


def _utility_for_selection(df: pd.DataFrame, selected: list[tuple[int, str, float]], specs: dict[str, dict[str, Any]]) -> float:
    utility = 0.0
    for idx, route, _ in selected:
        utility += float(bool(df.iloc[idx]["is_error"]))
        utility += float(bool(df.iloc[idx]["is_vtvf_cross_error"]))
        target = specs[route]["target"]
        utility += float(bool(df.iloc[idx][target]))
    return float(utility)


def _optimize_weights(val_df: pd.DataFrame, specs: dict[str, dict[str, Any]], budget: float) -> tuple[dict[str, float], dict[str, Any]]:
    templates = [
        ("v4_equal_plus_wavelet", {"vtvf_boundary": 1, "sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 1, "wavelet_boundary": 1}),
        ("v4_boundary_wavelet_pair", {"vtvf_boundary": 2, "sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 1, "wavelet_boundary": 2}),
        ("wavelet_boundary_heavy", {"vtvf_boundary": 1, "sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 1, "wavelet_boundary": 4}),
        ("soft_boundary_heavy_no_wavelet", {"vtvf_boundary": 4, "sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 1, "wavelet_boundary": 0}),
        ("wavelet_only", {"wavelet_boundary": 1}),
        ("vtvf_only", {"vtvf_boundary": 1}),
        ("atypical_wavelet", {"atypical_signal": 2, "wavelet_boundary": 3}),
        ("representation_wavelet", {"representation_conflict": 2, "wavelet_boundary": 3}),
        ("all_error_with_wavelet", {"vtvf_boundary": 2, "sr_ventricular": 2, "representation_conflict": 2, "atypical_signal": 2, "wavelet_boundary": 1}),
    ]
    n_select = max(1, int(round(len(val_df) * budget)))
    best_name = ""
    best_score = -1.0
    best_weights = {name: 0.0 for name in specs}
    for name, template in templates:
        weights = {spec_name: float(template.get(spec_name, 0.0)) for spec_name in specs}
        if not any(value > 0 for value in weights.values()):
            continue
        selected = _greedy_unique_selection(_candidate_tuples(val_df, val_df, specs, weights), n_select)
        score = _utility_for_selection(val_df, selected, specs)
        active = sum(value > 0 for value in weights.values())
        best_active = sum(value > 0 for value in best_weights.values())
        if score > best_score or (np.isclose(score, best_score) and active < best_active):
            best_name = name
            best_score = score
            best_weights = weights
    return best_weights, {"selected_profile": best_name, "validation_utility": best_score, "weights": best_weights}


def _assign_v5_route(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weights, optimization = _optimize_weights(val_df, specs, budget)
    n_select = max(1, int(round(len(test_df) * budget)))
    selected = _greedy_unique_selection(_candidate_tuples(test_df, val_df, specs, weights), n_select)
    routed = test_df.copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    weighted_scores = np.zeros(len(routed), dtype=np.float32)
    for idx, route, score in selected:
        actions[idx] = specs[route]["action"]
        routes[idx] = route
        weighted_scores[idx] = score

    output_set = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    review = actions != "single_label"
    vtvf_set = actions == "vtvf_boundary_set"
    output_set[review] = "review"
    output_set[vtvf_set] = "{VT,VF}"
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    routed["mechanism_output_set"] = output_set
    routed["mechanism_budget"] = budget
    routed["mechanism_strategy"] = "v5_wavelet_boundary_optimized"
    routed["mechanism_weighted_score"] = weighted_scores
    return routed, {
        "budget": budget,
        "optimization": optimization,
        "test_route_counts": pd.Series(routes[routes != "single_label"]).value_counts().to_dict(),
    }


def _summarize_routing(seed: int, routed: pd.DataFrame, budget: float) -> dict[str, float | int | str]:
    y = routed["y_true"].to_numpy(int)
    pred = routed["y_pred"].to_numpy(int)
    is_error = routed["is_error"].to_numpy(bool)
    is_vtvf = routed["is_vtvf_cross_error"].to_numpy(bool)
    action = routed["mechanism_action"].to_numpy(str)
    single = action == "single_label"
    addressed = ~single
    unresolved_error = single & (y != pred)
    unresolved_vtvf = single & is_vtvf
    row: dict[str, float | int | str] = {
        "seed": seed,
        "budget": budget,
        "method": "v5_wavelet_boundary_router",
        "policy_family": "mechanism_router",
        "action_rate": float(addressed.mean()),
        "vtvf_set_rate": float((action == "vtvf_boundary_set").mean()),
        "all_error_addressed": float((is_error & addressed).sum() / max(is_error.sum(), 1)),
        "vtvf_cross_error_addressed": float((is_vtvf & addressed).sum() / max(is_vtvf.sum(), 1)),
        "automatic_unresolved_error_rate": float(unresolved_error.mean()),
        "automatic_unresolved_vtvf_cross_error_rate": float(unresolved_vtvf.mean()),
        "single_label_error_rate_after_routing": float(unresolved_error.sum() / max(single.sum(), 1)),
        "single_label_vtvf_cross_error_rate_after_routing": float(unresolved_vtvf.sum() / max(single.sum(), 1)),
    }
    for route, count in routed.loc[addressed, "mechanism_route"].value_counts().items():
        row[f"route_count_{route}"] = int(count)
    return row


def _read_v4_summary(seed: int, seed_dir: Path) -> pd.DataFrame:
    path = seed_dir / "optimized_mechanism_layered_policy_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "seed": seed,
            "budget": df["budget"].astype(float),
            "method": "optimized_mechanism_router_v4",
            "policy_family": "mechanism_router",
            "action_rate": df["mechanism_action_rate"].astype(float),
            "vtvf_set_rate": df["vtvf_set_rate"].astype(float),
            "all_error_addressed": df["all_error_addressed"].astype(float),
            "vtvf_cross_error_addressed": df["vtvf_cross_error_addressed"].astype(float),
            "automatic_unresolved_error_rate": df["automatic_unresolved_error_rate"].astype(float),
            "automatic_unresolved_vtvf_cross_error_rate": df["automatic_unresolved_vtvf_cross_error_rate"].astype(float),
        }
    )


def _run_seed(seed: int, routing_dir: Path, out_dir: Path, mat: Path, budgets: list[float], max_windows_per_record: int | None) -> dict[str, Any]:
    seed_dir = routing_dir / f"seed{seed}"
    val_path = seed_dir / "evidence_scores_val.csv"
    test_path = seed_dir / "evidence_scores_test.csv"
    if not val_path.exists() or not test_path.exists():
        return {"seed": seed, "status": "missing_v4_evidence"}
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    raw = _reconstruct_split(seed, mat, max_windows_per_record)
    alignment = {
        "seed": seed,
        "val_y_aligned": bool(len(raw["val"][1]) == len(val_df) and np.array_equal(raw["val"][1], val_df["y_true"].to_numpy(int))),
        "test_y_aligned": bool(len(raw["test"][1]) == len(test_df) and np.array_equal(raw["test"][1], test_df["y_true"].to_numpy(int))),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    if not alignment["val_y_aligned"] or not alignment["test_y_aligned"]:
        return {"seed": seed, "status": "split_alignment_failed", **alignment}

    val_wavelet = _extract_wavelet_features(raw["val"][0])
    test_wavelet = _extract_wavelet_features(raw["test"][0])
    val_df = pd.concat([val_df.reset_index(drop=True), val_wavelet], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), test_wavelet], axis=1)
    val_df, test_df, head_df = _add_wavelet_risk_heads(val_df, test_df)
    specs = _base_specs(val_df)

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(seed_out / "evidence_scores_val_with_wavelet.csv", index=False)
    test_df.to_csv(seed_out / "evidence_scores_test_with_wavelet.csv", index=False)
    head_df.to_csv(seed_out / "wavelet_risk_head_summary.csv", index=False)

    ranking_rows = _ranked_review_rows(seed, test_df, budgets)
    v5_rows = []
    diagnostics = []
    routed_frames = []
    for budget in budgets:
        routed, info = _assign_v5_route(val_df, test_df, specs, budget)
        v5_rows.append(_summarize_routing(seed, routed, budget))
        diagnostics.append({"seed": seed, **info})
        routed["seed"] = seed
        routed["budget"] = budget
        routed_frames.append(routed)
    pd.DataFrame(ranking_rows).to_csv(seed_out / "wavelet_ranked_policy_summary.csv", index=False)
    pd.DataFrame(v5_rows).to_csv(seed_out / "v5_wavelet_mechanism_policy_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_json(seed_out / "v5_wavelet_budget_diagnostics.json", orient="records", indent=2)
    pd.concat(routed_frames, ignore_index=True).to_csv(seed_out / "v5_wavelet_routing_assignments_test.csv", index=False)
    return {"seed": seed, "status": "completed", **alignment, "out": str(seed_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed wavelet/time-frequency evidence as a mechanism routing expert.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    out_dir = args.out or (args.routing_dir / "wavelet_boundary_routing_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    all_rows = []
    head_rows = []
    for seed in SEEDS:
        info = _run_seed(seed, args.routing_dir, out_dir, args.mat, args.budgets, args.max_windows_per_record)
        manifest.append(info)
        seed_out = out_dir / f"seed{seed}"
        for filename in ["wavelet_ranked_policy_summary.csv", "v5_wavelet_mechanism_policy_summary.csv"]:
            path = seed_out / filename
            if path.exists():
                all_rows.append(pd.read_csv(path))
        v4 = _read_v4_summary(seed, args.routing_dir / f"seed{seed}")
        if not v4.empty:
            all_rows.append(v4)
        head_path = seed_out / "wavelet_risk_head_summary.csv"
        if head_path.exists():
            heads = pd.read_csv(head_path)
            heads["seed"] = seed
            head_rows.append(heads)

    pd.DataFrame(manifest).to_csv(out_dir / "wavelet_boundary_alignment_manifest.csv", index=False)
    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_df.to_csv(out_dir / "all_seed_wavelet_boundary_policy_summary.csv", index=False)
    head_df = pd.concat(head_rows, ignore_index=True) if head_rows else pd.DataFrame()
    head_df.to_csv(out_dir / "all_seed_wavelet_risk_head_summary.csv", index=False)

    metric_cols = [
        "action_rate",
        "vtvf_set_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
    ]
    if not all_df.empty:
        summary = _mean_std(all_df, ["method", "policy_family", "budget"], metric_cols)
        summary.to_csv(out_dir / "wavelet_boundary_policy_mean_std.csv", index=False)
    if not head_df.empty:
        _mean_std(head_df, ["score", "target"], ["test_auroc", "test_aupr", "val_positive", "test_positive"]).to_csv(
            out_dir / "wavelet_risk_head_mean_std.csv", index=False
        )

    paired_rows = []
    if not all_df.empty:
        v5 = all_df[all_df["method"].eq("v5_wavelet_boundary_router")]
        for method in sorted(set(all_df["method"]) - {"v5_wavelet_boundary_router"}):
            other = all_df[all_df["method"].eq(method)]
            merged = v5.merge(other, on=["seed", "budget"], suffixes=("_v5", "_baseline"))
            for _, row in merged.iterrows():
                paired_rows.append(
                    {
                        "seed": int(row["seed"]),
                        "budget": float(row["budget"]),
                        "baseline_method": method,
                        "delta_all_error_addressed_v5_minus_baseline": row["all_error_addressed_v5"]
                        - row["all_error_addressed_baseline"],
                        "delta_vtvf_cross_error_addressed_v5_minus_baseline": row["vtvf_cross_error_addressed_v5"]
                        - row["vtvf_cross_error_addressed_baseline"],
                        "delta_unresolved_vtvf_v5_minus_baseline": row[
                            "automatic_unresolved_vtvf_cross_error_rate_v5"
                        ]
                        - row["automatic_unresolved_vtvf_cross_error_rate_baseline"],
                        "delta_action_rate_v5_minus_baseline": row["action_rate_v5"] - row["action_rate_baseline"],
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_v5_wavelet_vs_baselines.csv", index=False)
    if not paired.empty:
        _mean_std(
            paired.rename(columns={"baseline_method": "method"}),
            ["method", "budget"],
            [
                "delta_all_error_addressed_v5_minus_baseline",
                "delta_vtvf_cross_error_addressed_v5_minus_baseline",
                "delta_unresolved_vtvf_v5_minus_baseline",
                "delta_action_rate_v5_minus_baseline",
            ],
        ).to_csv(out_dir / "paired_v5_wavelet_vs_baselines_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "out_dir": str(out_dir),
        "mat": str(args.mat),
        "budgets": args.budgets,
        "manifest": manifest,
        "n_policy_rows": int(len(all_df)),
        "n_head_rows": int(len(head_df)),
    }
    (out_dir / "wavelet_boundary_routing_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all_df.empty:
        print(
            pd.read_csv(out_dir / "wavelet_boundary_policy_mean_std.csv")
            .sort_values(["budget", "vtvf_cross_error_addressed_mean"], ascending=[True, False])
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
