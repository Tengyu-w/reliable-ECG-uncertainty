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
)
from .hierarchical_router_v5c import (
    _active_residual_specs,
    _greedy_unique,
    _optimize_residual_weights,
    _read_baselines,
    _residual_candidate_tuples,
)


def _top_n_mask(score: np.ndarray, n_select: int, candidate: np.ndarray | None = None) -> np.ndarray:
    if candidate is None:
        candidate = np.ones(len(score), dtype=bool)
    eligible = np.flatnonzero(candidate)
    n = min(len(eligible), max(0, n_select))
    mask = np.zeros(len(score), dtype=bool)
    if n <= 0:
        return mask
    ordered = eligible[np.argsort(-score[eligible])]
    mask[ordered[:n]] = True
    return mask


def _assign_v5d(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_boundary_score: np.ndarray,
    test_boundary_score: np.ndarray,
    budget: float,
    reserve_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_total_val = max(1, int(round(len(val_df) * budget)))
    n_total_test = max(1, int(round(len(test_df) * budget)))
    reserve_val = int(round(n_total_val * reserve_fraction))
    reserve_test = int(round(n_total_test * reserve_fraction))
    requested_stage1_val = max(0, n_total_val - reserve_val)
    requested_stage1_test = max(0, n_total_test - reserve_test)

    candidate_val = val_df["is_vtvf_candidate"].to_numpy(bool)
    candidate_test = test_df["is_vtvf_candidate"].to_numpy(bool)
    val_stage1 = _top_n_mask(val_boundary_score, requested_stage1_val, candidate=candidate_val)
    test_stage1 = _top_n_mask(test_boundary_score, requested_stage1_test, candidate=candidate_test)

    residual_val_slots = max(0, n_total_val - int(val_stage1.sum()))
    residual_test_slots = max(0, n_total_test - int(test_stage1.sum()))
    specs = _active_residual_specs(val_df)
    weights, optimization = _optimize_residual_weights(val_df, val_stage1, specs, residual_val_slots)
    residual_picks = _greedy_unique(
        _residual_candidate_tuples(test_df, test_stage1, specs, weights),
        residual_test_slots,
    )

    routed = test_df[
        [
            "sample_id",
            "split",
            "y_true",
            "y_pred",
            "is_error",
            "is_vtvf_cross_error",
            "is_sr_ventricular_error",
            "is_representation_conflict_error",
            "is_atypical_signal_error",
            "is_hidden_confident_error",
        ]
    ].copy()
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
    routed["mechanism_strategy"] = "v5d_reserved_residual_budget"
    routed["budget"] = budget
    routed["reserve_fraction"] = reserve_fraction
    diagnostics = {
        "budget": budget,
        "reserve_fraction": reserve_fraction,
        "requested_total_slots": n_total_test,
        "requested_stage1_slots": requested_stage1_test,
        "requested_residual_reserve_slots": reserve_test,
        "stage1_slots": int(test_stage1.sum()),
        "stage2_slots": len(residual_picks),
        "selected_slots": int((actions != "single_label").sum()),
        "residual_optimization": optimization,
        "stage2_route_counts": pd.Series([name for _, name, _ in residual_picks]).value_counts().to_dict()
        if residual_picks
        else {},
    }
    return routed, diagnostics


def _summarize(seed: int, routed: pd.DataFrame, budget: float, reserve_fraction: float) -> dict[str, Any]:
    y = routed["y_true"].to_numpy(int)
    pred = routed["y_pred"].to_numpy(int)
    is_error = routed["is_error"].to_numpy(bool)
    is_vtvf = routed["is_vtvf_cross_error"].to_numpy(bool)
    action = routed["mechanism_action"].to_numpy(str)
    stage = routed["hierarchical_stage"].to_numpy(str)
    selected = action != "single_label"
    single = ~selected
    unresolved_error = single & (y != pred)
    unresolved_vtvf = single & is_vtvf
    row: dict[str, Any] = {
        "seed": seed,
        "method": f"hierarchical_router_v5d_reserve_{int(round(reserve_fraction * 100))}pct",
        "policy_family": "hierarchical_router",
        "budget": budget,
        "reserve_fraction": reserve_fraction,
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


def _run_seed(
    seed: int,
    routing_dir: Path,
    wavelet_dir: Path,
    boundary_dir: Path,
    validity_run: Path,
    out_dir: Path,
    budgets: list[float],
    reserve_fractions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    val_df, test_df, alignment = _load_seed_frames(seed, routing_dir, wavelet_dir, validity_run)
    val_scores, test_scores, _ = _score_dict(val_df, test_df)
    val_boundary_score = val_scores["mean_softmax_validity_wavelet"]
    test_boundary_score = test_scores["mean_softmax_validity_wavelet"]

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    compact_frames = []
    for reserve_fraction in reserve_fractions:
        for budget in budgets:
            routed, info = _assign_v5d(
                val_df,
                test_df,
                val_boundary_score,
                test_boundary_score,
                budget,
                reserve_fraction,
            )
            rows.append(_summarize(seed, routed, budget, reserve_fraction))
            diagnostics.append({"seed": seed, **info})
            compact_frames.append(routed)
    rows.extend(_read_baselines(seed, routing_dir, boundary_dir))
    pd.DataFrame(rows).to_csv(seed_out / "v5d_reserved_policy_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_json(seed_out / "v5d_reserved_diagnostics.json", orient="records", indent=2)
    pd.concat(compact_frames, ignore_index=True).to_csv(seed_out / "v5d_reserved_routing_assignments_test_compact.csv", index=False)
    return rows, diagnostics, alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="v5d: hierarchical router with reserved residual-stage budget.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--wavelet-dir", type=Path, default=None)
    parser.add_argument("--boundary-dir", type=Path, default=None)
    parser.add_argument("--validity-root", type=Path, default=Path("results/cnn_tcn_validity_20260626"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--reserve-fractions", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    wavelet_dir = args.wavelet_dir or (args.routing_dir / "wavelet_boundary_routing_audit")
    boundary_dir = args.boundary_dir or (args.routing_dir / "boundary_first_router_v5b")
    out_dir = args.out or (args.routing_dir / "hierarchical_router_v5d_reserved")
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
            args.reserve_fractions,
        )
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)
        manifest.append({"seed": seed, "status": "completed", **alignment})

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_seed_v5d_reserved_policy_summary.csv", index=False)
    pd.DataFrame(all_diagnostics).to_json(out_dir / "all_seed_v5d_reserved_diagnostics.json", orient="records", indent=2)
    pd.DataFrame(manifest).to_csv(out_dir / "v5d_alignment_manifest.csv", index=False)

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
    summary = _mean_std(
        all_df,
        ["method", "policy_family", "reserve_fraction", "budget"],
        [c for c in metrics if c in all_df.columns],
    )
    summary.to_csv(out_dir / "v5d_reserved_policy_mean_std.csv", index=False)

    paired_rows = []
    v5d = all_df[all_df["method"].str.startswith("hierarchical_router_v5d_reserve_")]
    baselines = all_df[~all_df["method"].str.startswith("hierarchical_router_v5d_reserve_")]
    for method in sorted(baselines["method"].unique()):
        other = baselines[baselines["method"].eq(method)]
        merged = v5d.merge(other, on=["seed", "budget"], suffixes=("_v5d", "_baseline"))
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "reserve_fraction": float(row["reserve_fraction_v5d"]),
                    "v5d_method": row["method_v5d"],
                    "baseline_method": method,
                    "delta_action_rate_v5d_minus_baseline": row["action_rate_v5d"] - row["action_rate_baseline"],
                    "delta_all_error_addressed_v5d_minus_baseline": row["all_error_addressed_v5d"]
                    - row["all_error_addressed_baseline"],
                    "delta_vtvf_cross_error_addressed_v5d_minus_baseline": row["vtvf_cross_error_addressed_v5d"]
                    - row["vtvf_cross_error_addressed_baseline"],
                    "delta_unresolved_error_v5d_minus_baseline": row["automatic_unresolved_error_rate_v5d"]
                    - row["automatic_unresolved_error_rate_baseline"],
                    "delta_unresolved_vtvf_v5d_minus_baseline": row[
                        "automatic_unresolved_vtvf_cross_error_rate_v5d"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_baseline"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_v5d_reserved_vs_baselines.csv", index=False)
    if not paired.empty:
        _mean_std(
            paired.rename(columns={"baseline_method": "method"}),
            ["method", "reserve_fraction", "budget"],
            [
                "delta_action_rate_v5d_minus_baseline",
                "delta_all_error_addressed_v5d_minus_baseline",
                "delta_vtvf_cross_error_addressed_v5d_minus_baseline",
                "delta_unresolved_error_v5d_minus_baseline",
                "delta_unresolved_vtvf_v5d_minus_baseline",
            ],
        ).to_csv(out_dir / "paired_v5d_reserved_vs_baselines_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "wavelet_dir": str(wavelet_dir),
        "boundary_dir": str(boundary_dir),
        "validity_root": str(args.validity_root),
        "out_dir": str(out_dir),
        "budgets": args.budgets,
        "reserve_fractions": args.reserve_fractions,
        "n_rows": int(len(all_df)),
        "manifest": manifest,
    }
    (out_dir / "v5d_reserved_router_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.sort_values(["budget", "reserve_fraction"]).to_string(index=False))


if __name__ == "__main__":
    main()
