from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .boundary_first_router_v5b import (
    CLASS_LABELS,
    SEEDS,
    _discover_validity_runs,
    _load_seed_frames,
    _mean_std,
    _score_dict,
    _top_budget_mask,
)


RESIDUAL_SPECS = {
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
    "hidden_confident": {
        "score_col": "hidden_confident_mechanism_risk",
        "target": "is_hidden_confident_error",
        "action": "hidden_failure_review",
    },
}


def _active_residual_specs(val_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs = {}
    for name, spec in RESIDUAL_SPECS.items():
        if spec["score_col"] not in val_df.columns or spec["target"] not in val_df.columns:
            continue
        val_positive = int(val_df[spec["target"]].sum())
        if val_positive < 5:
            continue
        specs[name] = {**spec, "val_positive": val_positive}
    return specs


def _residual_candidate_mask(df: pd.DataFrame, selected: np.ndarray, name: str) -> np.ndarray:
    mask = ~selected
    if name == "hidden_confident" and "max_prob" in df.columns:
        mask = mask & (df["max_prob"].to_numpy(float) >= np.nanquantile(df["max_prob"].to_numpy(float), 0.50))
    return mask


def _residual_candidate_tuples(
    df: pd.DataFrame,
    selected: np.ndarray,
    specs: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> list[tuple[float, int, str]]:
    candidates = []
    for name, spec in specs.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        mask = _residual_candidate_mask(df, selected, name)
        score = df[spec["score_col"]].to_numpy(float) * weight
        for idx in np.flatnonzero(mask):
            candidates.append((float(score[idx]), int(idx), name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _greedy_unique(candidates: list[tuple[float, int, str]], n_select: int) -> list[tuple[int, str, float]]:
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


def _residual_utility(df: pd.DataFrame, picks: list[tuple[int, str, float]], specs: dict[str, dict[str, Any]]) -> float:
    utility = 0.0
    for idx, name, _ in picks:
        utility += float(bool(df.iloc[idx]["is_error"]))
        target = specs[name]["target"]
        utility += float(bool(df.iloc[idx][target]))
    return float(utility)


def _optimize_residual_weights(
    val_df: pd.DataFrame,
    val_stage1_selected: np.ndarray,
    specs: dict[str, dict[str, Any]],
    n_select: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    templates = [
        ("equal_residual", {"sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 1, "hidden_confident": 1}),
        ("sr_heavy", {"sr_ventricular": 4, "representation_conflict": 1, "atypical_signal": 1, "hidden_confident": 1}),
        ("representation_heavy", {"sr_ventricular": 1, "representation_conflict": 4, "atypical_signal": 1, "hidden_confident": 1}),
        ("atypical_heavy", {"sr_ventricular": 1, "representation_conflict": 1, "atypical_signal": 4, "hidden_confident": 1}),
        ("sr_atypical", {"sr_ventricular": 3, "atypical_signal": 3}),
        ("representation_atypical", {"representation_conflict": 3, "atypical_signal": 3}),
        ("sr_only", {"sr_ventricular": 1}),
        ("representation_only", {"representation_conflict": 1}),
        ("atypical_only", {"atypical_signal": 1}),
    ]
    if n_select <= 0 or not specs:
        return {name: 0.0 for name in specs}, {"selected_profile": "no_residual_budget", "validation_utility": 0.0}
    best_profile = ""
    best_score = -1.0
    best_weights = {name: 0.0 for name in specs}
    for profile, template in templates:
        weights = {name: float(template.get(name, 0.0)) for name in specs}
        if not any(value > 0 for value in weights.values()):
            continue
        picks = _greedy_unique(_residual_candidate_tuples(val_df, val_stage1_selected, specs, weights), n_select)
        utility = _residual_utility(val_df, picks, specs)
        active = sum(value > 0 for value in weights.values())
        best_active = sum(value > 0 for value in best_weights.values())
        if utility > best_score or (np.isclose(utility, best_score) and active < best_active):
            best_score = utility
            best_profile = profile
            best_weights = weights
    return best_weights, {"selected_profile": best_profile, "validation_utility": best_score, "weights": best_weights}


def _assign_v5c(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_boundary_score: np.ndarray,
    test_boundary_score: np.ndarray,
    budget: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_total_val = max(1, int(round(len(val_df) * budget)))
    n_total_test = max(1, int(round(len(test_df) * budget)))
    candidate_val = val_df["is_vtvf_candidate"].to_numpy(bool)
    candidate_test = test_df["is_vtvf_candidate"].to_numpy(bool)
    val_stage1 = _top_budget_mask(val_boundary_score, budget, candidate=candidate_val)
    test_stage1 = _top_budget_mask(test_boundary_score, budget, candidate=candidate_test)
    residual_val_slots = max(0, n_total_val - int(val_stage1.sum()))
    residual_test_slots = max(0, n_total_test - int(test_stage1.sum()))
    specs = _active_residual_specs(val_df)
    weights, optimization = _optimize_residual_weights(val_df, val_stage1, specs, residual_val_slots)
    residual_picks = _greedy_unique(
        _residual_candidate_tuples(test_df, test_stage1, specs, weights),
        residual_test_slots,
    )

    routed = test_df.copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    output = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    stage = np.full(len(routed), "none", dtype=object)
    score = np.zeros(len(routed), dtype=np.float32)

    actions[test_stage1] = "vtvf_boundary_set"
    routes[test_stage1] = "boundary_first"
    output[test_stage1] = "{VT,VF}"
    stage[test_stage1] = "stage1_boundary"
    score[test_stage1] = test_boundary_score[test_stage1]

    for idx, name, pick_score in residual_picks:
        actions[idx] = specs[name]["action"]
        routes[idx] = name
        output[idx] = "review"
        stage[idx] = "stage2_residual"
        score[idx] = float(pick_score)

    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    routed["mechanism_output_set"] = output
    routed["hierarchical_stage"] = stage
    routed["hierarchical_score"] = score
    routed["mechanism_strategy"] = "v5c_boundary_first_then_residual"
    routed["budget"] = budget
    diagnostics = {
        "budget": budget,
        "requested_total_slots": n_total_test,
        "stage1_slots": int(test_stage1.sum()),
        "stage2_slots": len(residual_picks),
        "selected_slots": int((actions != "single_label").sum()),
        "residual_optimization": optimization,
        "stage2_route_counts": pd.Series([name for _, name, _ in residual_picks]).value_counts().to_dict()
        if residual_picks
        else {},
    }
    return routed, diagnostics


def _summarize(seed: int, routed: pd.DataFrame, budget: float, method: str) -> dict[str, Any]:
    y = routed["y_true"].to_numpy(int)
    pred = routed["y_pred"].to_numpy(int)
    is_error = routed["is_error"].to_numpy(bool)
    is_vtvf = routed["is_vtvf_cross_error"].to_numpy(bool)
    action = routed["mechanism_action"].to_numpy(str)
    single = action == "single_label"
    selected = ~single
    stage = routed["hierarchical_stage"].to_numpy(str)
    unresolved_error = single & (y != pred)
    unresolved_vtvf = single & is_vtvf
    row: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "policy_family": "hierarchical_router",
        "budget": budget,
        "action_rate": float(selected.mean()),
        "stage1_boundary_rate": float((stage == "stage1_boundary").mean()),
        "stage2_residual_rate": float((stage == "stage2_residual").mean()),
        "vtvf_set_rate": float((action == "vtvf_boundary_set").mean()),
        "all_error_addressed": float((is_error & selected).sum() / max(is_error.sum(), 1)),
        "vtvf_cross_error_addressed": float((is_vtvf & selected).sum() / max(is_vtvf.sum(), 1)),
        "automatic_unresolved_error_rate": float(unresolved_error.mean()),
        "automatic_unresolved_vtvf_cross_error_rate": float(unresolved_vtvf.mean()),
        "single_label_error_rate_after_routing": float(unresolved_error.sum() / max(single.sum(), 1)),
        "single_label_vtvf_cross_error_rate_after_routing": float(unresolved_vtvf.sum() / max(single.sum(), 1)),
    }
    for route, count in routed.loc[selected, "mechanism_route"].value_counts().items():
        row[f"route_count_{route}"] = int(count)
    return row


def _read_baselines(seed: int, routing_dir: Path, boundary_dir: Path) -> list[dict[str, Any]]:
    rows = []
    v4 = routing_dir / f"seed{seed}" / "optimized_mechanism_layered_policy_summary.csv"
    if v4.exists():
        df = pd.read_csv(v4)
        for _, row in df.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "method": "optimized_mechanism_router_v4",
                    "policy_family": "mechanism_router",
                    "budget": float(row["budget"]),
                    "action_rate": float(row["mechanism_action_rate"]),
                    "stage1_boundary_rate": np.nan,
                    "stage2_residual_rate": np.nan,
                    "vtvf_set_rate": float(row["vtvf_set_rate"]),
                    "all_error_addressed": float(row["all_error_addressed"]),
                    "vtvf_cross_error_addressed": float(row["vtvf_cross_error_addressed"]),
                    "automatic_unresolved_error_rate": float(row["automatic_unresolved_error_rate"]),
                    "automatic_unresolved_vtvf_cross_error_rate": float(row["automatic_unresolved_vtvf_cross_error_rate"]),
                }
            )
    v5b = boundary_dir / f"seed{seed}" / "boundary_first_policy_summary.csv"
    if v5b.exists():
        df = pd.read_csv(v5b)
        df = df[df["method"].eq("mean_softmax_validity_wavelet")]
        for _, row in df.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "method": "boundary_first_v5b",
                    "policy_family": "boundary_first_set",
                    "budget": float(row["budget"]),
                    "action_rate": float(row["action_rate"]),
                    "stage1_boundary_rate": float(row["action_rate"]),
                    "stage2_residual_rate": 0.0,
                    "vtvf_set_rate": float(row["vtvf_set_rate"]),
                    "all_error_addressed": float(row["all_error_addressed"]),
                    "vtvf_cross_error_addressed": float(row["vtvf_cross_error_addressed"]),
                    "automatic_unresolved_error_rate": float(row["automatic_unresolved_error_rate"]),
                    "automatic_unresolved_vtvf_cross_error_rate": float(row["automatic_unresolved_vtvf_cross_error_rate"]),
                }
            )
    return rows


def _run_seed(
    seed: int,
    routing_dir: Path,
    wavelet_dir: Path,
    boundary_dir: Path,
    validity_run: Path,
    out_dir: Path,
    budgets: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    val_df, test_df, alignment = _load_seed_frames(seed, routing_dir, wavelet_dir, validity_run)
    val_scores, test_scores, _ = _score_dict(val_df, test_df)
    val_boundary_score = val_scores["mean_softmax_validity_wavelet"]
    test_boundary_score = test_scores["mean_softmax_validity_wavelet"]
    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    routed_frames = []
    for budget in budgets:
        routed, info = _assign_v5c(val_df, test_df, val_boundary_score, test_boundary_score, budget)
        rows.append(_summarize(seed, routed, budget, "hierarchical_router_v5c"))
        diagnostics.append({"seed": seed, **info})
        routed_frames.append(routed)
    rows.extend(_read_baselines(seed, routing_dir, boundary_dir))
    pd.DataFrame(rows).to_csv(seed_out / "v5c_hierarchical_policy_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_json(seed_out / "v5c_hierarchical_diagnostics.json", orient="records", indent=2)
    pd.concat(routed_frames, ignore_index=True).to_csv(seed_out / "v5c_hierarchical_routing_assignments_test.csv", index=False)
    return rows, diagnostics, alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="v5c: boundary-first route followed by residual mechanism routing.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--wavelet-dir", type=Path, default=None)
    parser.add_argument("--boundary-dir", type=Path, default=None)
    parser.add_argument("--validity-root", type=Path, default=Path("results/cnn_tcn_validity_20260626"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    wavelet_dir = args.wavelet_dir or (args.routing_dir / "wavelet_boundary_routing_audit")
    boundary_dir = args.boundary_dir or (args.routing_dir / "boundary_first_router_v5b")
    out_dir = args.out or (args.routing_dir / "hierarchical_router_v5c")
    out_dir.mkdir(parents=True, exist_ok=True)
    validity_runs = _discover_validity_runs(args.validity_root)

    all_rows = []
    all_diagnostics = []
    manifest = []
    for seed in SEEDS:
        if seed not in validity_runs:
            manifest.append({"seed": seed, "status": "missing_validity_run"})
            continue
        rows, diagnostics, alignment = _run_seed(
            seed,
            args.routing_dir,
            wavelet_dir,
            boundary_dir,
            validity_runs[seed],
            out_dir,
            args.budgets,
        )
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)
        manifest.append({"seed": seed, "status": "completed", **alignment})

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_seed_v5c_hierarchical_policy_summary.csv", index=False)
    pd.DataFrame(all_diagnostics).to_json(out_dir / "all_seed_v5c_hierarchical_diagnostics.json", orient="records", indent=2)
    pd.DataFrame(manifest).to_csv(out_dir / "v5c_alignment_manifest.csv", index=False)

    metrics = [
        "action_rate",
        "stage1_boundary_rate",
        "stage2_residual_rate",
        "vtvf_set_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
        "single_label_error_rate_after_routing",
        "single_label_vtvf_cross_error_rate_after_routing",
    ]
    summary = _mean_std(all_df, ["method", "policy_family", "budget"], [c for c in metrics if c in all_df.columns])
    summary.to_csv(out_dir / "v5c_hierarchical_policy_mean_std.csv", index=False)

    paired_rows = []
    main = all_df[all_df["method"].eq("hierarchical_router_v5c")]
    for method in sorted(set(all_df["method"]) - {"hierarchical_router_v5c"}):
        other = all_df[all_df["method"].eq(method)]
        merged = main.merge(other, on=["seed", "budget"], suffixes=("_v5c", "_baseline"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "baseline_method": method,
                    "delta_action_rate_v5c_minus_baseline": row["action_rate_v5c"] - row["action_rate_baseline"],
                    "delta_all_error_addressed_v5c_minus_baseline": row["all_error_addressed_v5c"]
                    - row["all_error_addressed_baseline"],
                    "delta_vtvf_cross_error_addressed_v5c_minus_baseline": row["vtvf_cross_error_addressed_v5c"]
                    - row["vtvf_cross_error_addressed_baseline"],
                    "delta_unresolved_error_v5c_minus_baseline": row["automatic_unresolved_error_rate_v5c"]
                    - row["automatic_unresolved_error_rate_baseline"],
                    "delta_unresolved_vtvf_v5c_minus_baseline": row[
                        "automatic_unresolved_vtvf_cross_error_rate_v5c"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_baseline"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_v5c_vs_baselines.csv", index=False)
    if not paired.empty:
        _mean_std(
            paired.rename(columns={"baseline_method": "method"}),
            ["method", "budget"],
            [
                "delta_action_rate_v5c_minus_baseline",
                "delta_all_error_addressed_v5c_minus_baseline",
                "delta_vtvf_cross_error_addressed_v5c_minus_baseline",
                "delta_unresolved_error_v5c_minus_baseline",
                "delta_unresolved_vtvf_v5c_minus_baseline",
            ],
        ).to_csv(out_dir / "paired_v5c_vs_baselines_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "wavelet_dir": str(wavelet_dir),
        "boundary_dir": str(boundary_dir),
        "validity_root": str(args.validity_root),
        "out_dir": str(out_dir),
        "budgets": args.budgets,
        "n_rows": int(len(all_df)),
        "manifest": manifest,
    }
    (out_dir / "v5c_hierarchical_router_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.sort_values(["budget", "vtvf_cross_error_addressed_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
