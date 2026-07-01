from __future__ import annotations

import argparse
import json
from itertools import product
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
    _robust_scale_from_val,
)
from .hierarchical_router_v5c import _active_residual_specs, _greedy_unique, _residual_candidate_tuples


DEFAULT_ROUTING_DIR = Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627")
DEFAULT_VALIDITY_ROOT = Path("results/cnn_tcn_validity_20260626")
DEFAULT_OUT_DIR = Path("results/v5d_causal_pareto_weight_upgrade_20260629")


STAGE1_EVIDENCE = {
    "softmax_boundary": "softmax_vtvf_ambiguity",
    "validity_boundary": "validity_v1_gate_x_boundary",
    "wavelet_boundary": "wavelet_vtvf_boundary_risk",
    "prototype_boundary": "proto_vtvf_ambiguity",
    "knn_boundary": "knn_vtvf_mixing",
}


def _available_stage1_evidence(df: pd.DataFrame) -> dict[str, str]:
    return {name: col for name, col in STAGE1_EVIDENCE.items() if col in df.columns}


def _stage1_profiles(evidence: dict[str, str]) -> list[tuple[str, dict[str, float]]]:
    names = list(evidence)
    templates: list[tuple[str, dict[str, float]]] = [
        ("stage1_equal_all", {name: 1.0 for name in names}),
        (
            "stage1_original_softmax_validity_wavelet",
            {name: (1.0 if name in {"softmax_boundary", "validity_boundary", "wavelet_boundary"} else 0.0) for name in names},
        ),
        ("stage1_boundary_model_heavy", {name: (4.0 if name == "softmax_boundary" else 1.0) for name in names}),
        ("stage1_validity_heavy", {name: (4.0 if name == "validity_boundary" else 1.0) for name in names}),
        ("stage1_wavelet_heavy", {name: (4.0 if name == "wavelet_boundary" else 1.0) for name in names}),
        (
            "stage1_embedding_geometry_heavy",
            {name: (4.0 if name in {"prototype_boundary", "knn_boundary"} else 1.0) for name in names},
        ),
        (
            "stage1_wavelet_embedding_pair",
            {name: (1.0 if name in {"wavelet_boundary", "prototype_boundary", "knn_boundary"} else 0.0) for name in names},
        ),
        (
            "stage1_validity_embedding_pair",
            {name: (1.0 if name in {"validity_boundary", "prototype_boundary", "knn_boundary"} else 0.0) for name in names},
        ),
    ]
    for name in names:
        templates.append((f"stage1_{name}_only", {x: (1.0 if x == name else 0.0) for x in names}))
        templates.append((f"stage1_{name}_heavy", {x: (4.0 if x == name else 1.0) for x in names}))

    out: list[tuple[str, dict[str, float]]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for profile, weights in templates:
        active = {name: float(weights.get(name, 0.0)) for name in names}
        if not any(v > 0 for v in active.values()):
            continue
        key = tuple(sorted((name, round(value, 6)) for name, value in active.items() if value > 0))
        if key in seen:
            continue
        seen.add(key)
        out.append((profile, active))
    return out


def _residual_profiles(specs: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    names = list(specs)
    templates: list[tuple[str, dict[str, float]]] = [
        ("stage2_equal_residual", {name: 1.0 for name in names}),
        ("stage2_sr_heavy", {name: (4.0 if name == "sr_ventricular" else 1.0) for name in names}),
        (
            "stage2_representation_heavy",
            {name: (4.0 if name == "representation_conflict" else 1.0) for name in names},
        ),
        ("stage2_atypical_heavy", {name: (4.0 if name == "atypical_signal" else 1.0) for name in names}),
        (
            "stage2_sr_representation_pair",
            {name: (1.0 if name in {"sr_ventricular", "representation_conflict"} else 0.0) for name in names},
        ),
        (
            "stage2_sr_atypical_pair",
            {name: (1.0 if name in {"sr_ventricular", "atypical_signal"} else 0.0) for name in names},
        ),
        (
            "stage2_representation_atypical_pair",
            {name: (1.0 if name in {"representation_conflict", "atypical_signal"} else 0.0) for name in names},
        ),
    ]
    for name in names:
        templates.append((f"stage2_{name}_only", {x: (1.0 if x == name else 0.0) for x in names}))
        templates.append((f"stage2_{name}_heavy", {x: (4.0 if x == name else 1.0) for x in names}))

    out: list[tuple[str, dict[str, float]]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for profile, weights in templates:
        active = {name: float(weights.get(name, 0.0)) for name in names}
        if not any(v > 0 for v in active.values()):
            continue
        key = tuple(sorted((name, round(value, 6)) for name, value in active.items() if value > 0))
        if key in seen:
            continue
        seen.add(key)
        out.append((profile, active))
    if not out:
        out.append(("stage2_no_active_residual", {}))
    return out


def _scaled_evidence(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    evidence: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    val_scaled: dict[str, np.ndarray] = {}
    test_scaled: dict[str, np.ndarray] = {}
    for name, col in evidence.items():
        val_score = val_df[col].to_numpy(float)
        test_score = test_df[col].to_numpy(float)
        val_scaled[name] = _robust_scale_from_val(val_score, val_score)
        test_scaled[name] = _robust_scale_from_val(val_score, test_score)
    return val_scaled, test_scaled


def _weighted_score(scaled: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    active = [(name, float(weight)) for name, weight in weights.items() if weight > 0 and name in scaled]
    if not active:
        return np.zeros(len(next(iter(scaled.values()))), dtype=np.float32)
    total_weight = sum(weight for _, weight in active)
    score = np.zeros(len(next(iter(scaled.values()))), dtype=np.float32)
    for name, weight in active:
        score += scaled[name] * (weight / total_weight)
    return score.astype(np.float32)


def _top_n_mask(score: np.ndarray, n_select: int, candidate: np.ndarray | None = None) -> np.ndarray:
    if candidate is None:
        candidate = np.ones(len(score), dtype=bool)
    eligible = np.flatnonzero(candidate)
    n = min(len(eligible), max(0, int(n_select)))
    mask = np.zeros(len(score), dtype=bool)
    if n <= 0:
        return mask
    ordered = eligible[np.argsort(-score[eligible])]
    mask[ordered[:n]] = True
    return mask


def _route(
    df: pd.DataFrame,
    stage1_score: np.ndarray,
    stage2_specs: dict[str, dict[str, Any]],
    stage2_weights: dict[str, float],
    budget: float,
    reserve_fraction: float,
) -> pd.DataFrame:
    total_slots = max(1, int(round(len(df) * budget)))
    reserve_slots = int(round(total_slots * reserve_fraction))
    stage1_slots = max(0, total_slots - reserve_slots)
    candidate = df["is_vtvf_candidate"].to_numpy(bool)
    stage1_mask = _top_n_mask(stage1_score, stage1_slots, candidate)
    residual_slots = max(0, total_slots - int(stage1_mask.sum()))
    residual_picks = _greedy_unique(
        _residual_candidate_tuples(df, stage1_mask, stage2_specs, stage2_weights),
        residual_slots,
    )

    keep = [
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
    routed = df[keep].copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    output = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    stage = np.full(len(routed), "none", dtype=object)
    score = np.zeros(len(routed), dtype=np.float32)

    actions[stage1_mask] = "vtvf_boundary_set"
    routes[stage1_mask] = "boundary_first"
    output[stage1_mask] = "{VT,VF}"
    stage[stage1_mask] = "stage1_boundary"
    score[stage1_mask] = stage1_score[stage1_mask]

    for idx, name, pick_score in residual_picks:
        actions[idx] = stage2_specs[name]["action"]
        routes[idx] = name
        output[idx] = "review"
        stage[idx] = "stage2_residual"
        score[idx] = float(pick_score)

    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    routed["mechanism_output_set"] = output
    routed["hierarchical_stage"] = stage
    routed["hierarchical_score"] = score
    return routed


def _safe_bool(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].to_numpy(bool)


def _summarize(
    seed: int,
    policy_id: str,
    stage1_profile: str,
    stage2_profile: str,
    budget: float,
    reserve_fraction: float,
    routed: pd.DataFrame,
) -> dict[str, Any]:
    selected = routed["mechanism_action"].to_numpy(str) != "single_label"
    single = ~selected
    stage = routed["hierarchical_stage"].to_numpy(str)
    is_error = routed["is_error"].to_numpy(bool)
    is_vtvf = routed["is_vtvf_cross_error"].to_numpy(bool)
    row: dict[str, Any] = {
        "seed": seed,
        "method": "v5d_causal_pareto_weight_upgrade",
        "policy_family": "v5d_stage1_stage2_weight_upgrade",
        "policy_id": policy_id,
        "stage1_profile": stage1_profile,
        "stage2_profile": stage2_profile,
        "budget": budget,
        "reserve_fraction": reserve_fraction,
        "action_rate": float(selected.mean()),
        "stage1_boundary_rate": float((stage == "stage1_boundary").mean()),
        "stage2_residual_rate": float((stage == "stage2_residual").mean()),
        "all_error_capture": float((is_error & selected).sum() / max(is_error.sum(), 1)),
        "vtvf_capture": float((is_vtvf & selected).sum() / max(is_vtvf.sum(), 1)),
        "auto_error_rate": float((is_error & single).mean()) if single.any() else np.nan,
        "auto_vtvf_error_rate": float((is_vtvf & single).mean()) if single.any() else np.nan,
        "single_label_error_rate": float((is_error & single).sum() / max(single.sum(), 1)),
        "single_label_vtvf_error_rate": float((is_vtvf & single).sum() / max(single.sum(), 1)),
        "route_diversity": int(pd.Series(routed.loc[selected, "mechanism_route"]).nunique()) if selected.any() else 0,
    }
    for route, count in routed.loc[selected, "mechanism_route"].value_counts().items():
        row[f"route_count_{route}"] = int(count)
    targets = {
        "boundary_first": "is_vtvf_cross_error",
        "sr_ventricular": "is_sr_ventricular_error",
        "representation_conflict": "is_representation_conflict_error",
        "atypical_signal": "is_atypical_signal_error",
        "hidden_confident": "is_hidden_confident_error",
    }
    for route, target in targets.items():
        route_mask = routed["mechanism_route"].eq(route).to_numpy(bool)
        target_mask = _safe_bool(routed, target)
        row[f"target_capture_{route}"] = float((target_mask & route_mask).sum() / max(target_mask.sum(), 1))
        row[f"route_rate_{route}"] = float(route_mask.mean())
    return row


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    better_or_equal = (
        left["vtvf_capture"] >= right["vtvf_capture"]
        and left["all_error_capture"] >= right["all_error_capture"]
        and left["auto_vtvf_error_rate"] <= right["auto_vtvf_error_rate"]
        and left["auto_error_rate"] <= right["auto_error_rate"]
    )
    strictly = (
        left["vtvf_capture"] > right["vtvf_capture"]
        or left["all_error_capture"] > right["all_error_capture"]
        or left["auto_vtvf_error_rate"] < right["auto_vtvf_error_rate"]
        or left["auto_error_rate"] < right["auto_error_rate"]
    )
    return bool(better_or_equal and strictly)


def _mark_pareto(df: pd.DataFrame, group_cols: list[str], flag_col: str) -> pd.DataFrame:
    out = df.copy()
    flags = []
    for _, row in out.iterrows():
        peers = out
        for col in group_cols:
            peers = peers[peers[col] == row[col]]
        dominated = any(_dominates(peer, row) for _, peer in peers.iterrows() if peer.name != row.name)
        flags.append(not dominated)
    out[flag_col] = flags
    return out


def _candidate_policies(
    stage1_profiles: list[tuple[str, dict[str, float]]],
    stage2_profiles: list[tuple[str, dict[str, float]]],
    reserve_fractions: list[float],
) -> list[tuple[str, str, str, float, dict[str, float], dict[str, float]]]:
    policies = []
    for (stage1_name, stage1_weights), (stage2_name, stage2_weights), reserve in product(
        stage1_profiles, stage2_profiles, reserve_fractions
    ):
        policy_id = f"upgrade__{stage1_name}__{stage2_name}__reserve_{int(round(reserve * 100))}pct"
        policies.append((policy_id, stage1_name, stage2_name, float(reserve), stage1_weights, stage2_weights))
    return policies


def _run_seed(
    seed: int,
    routing_dir: Path,
    wavelet_dir: Path,
    validity_run: Path,
    budgets: list[float],
    reserve_fractions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    val_df, test_df, alignment = _load_seed_frames(seed, routing_dir, wavelet_dir, validity_run)
    stage1_evidence = _available_stage1_evidence(val_df)
    stage2_specs = _active_residual_specs(val_df)
    stage1_profiles = _stage1_profiles(stage1_evidence)
    stage2_profiles = _residual_profiles(stage2_specs)
    policies = _candidate_policies(stage1_profiles, stage2_profiles, reserve_fractions)
    val_scaled, test_scaled = _scaled_evidence(val_df, test_df, stage1_evidence)

    val_rows = []
    test_rows = []
    for budget in budgets:
        for policy_id, stage1_name, stage2_name, reserve, stage1_weights, stage2_weights in policies:
            val_stage1_score = _weighted_score(val_scaled, stage1_weights)
            test_stage1_score = _weighted_score(test_scaled, stage1_weights)
            val_routed = _route(val_df, val_stage1_score, stage2_specs, stage2_weights, budget, reserve)
            test_routed = _route(test_df, test_stage1_score, stage2_specs, stage2_weights, budget, reserve)
            val_row = _summarize(seed, policy_id, stage1_name, stage2_name, budget, reserve, val_routed)
            test_row = _summarize(seed, policy_id, stage1_name, stage2_name, budget, reserve, test_routed)
            for name, value in stage1_weights.items():
                val_row[f"stage1_weight_{name}"] = float(value)
                test_row[f"stage1_weight_{name}"] = float(value)
            for name, value in stage2_weights.items():
                val_row[f"stage2_weight_{name}"] = float(value)
                test_row[f"stage2_weight_{name}"] = float(value)
            val_rows.append(val_row)
            test_rows.append(test_row)

    val_summary = _mark_pareto(pd.DataFrame(val_rows), ["seed", "budget"], "validation_pareto")
    pareto_keys = set(
        zip(
            val_summary.loc[val_summary["validation_pareto"], "seed"],
            val_summary.loc[val_summary["validation_pareto"], "budget"],
            val_summary.loc[val_summary["validation_pareto"], "policy_id"],
        )
    )
    test_summary = pd.DataFrame(test_rows)
    test_summary["validation_pareto"] = [
        (row.seed, row.budget, row.policy_id) in pareto_keys for row in test_summary.itertuples(index=False)
    ]
    alignment.update(
        {
            "n_stage1_profiles": len(stage1_profiles),
            "n_stage2_profiles": len(stage2_profiles),
            "n_candidate_policies_per_budget": len(policies),
            "stage1_evidence": json.dumps(stage1_evidence, sort_keys=True),
            "stage2_mechanisms": ",".join(stage2_specs.keys()),
        }
    )
    return val_summary, test_summary, alignment


def _aggregate(test_df: pd.DataFrame, min_pareto_seeds: int) -> pd.DataFrame:
    metrics = [
        "action_rate",
        "stage1_boundary_rate",
        "stage2_residual_rate",
        "all_error_capture",
        "vtvf_capture",
        "auto_error_rate",
        "auto_vtvf_error_rate",
        "single_label_error_rate",
        "single_label_vtvf_error_rate",
        "route_diversity",
    ]
    extra = [
        col
        for col in test_df.columns
        if col.startswith("route_rate_") or col.startswith("target_capture_") or col.startswith("stage1_weight_") or col.startswith("stage2_weight_")
    ]
    pareto = test_df[test_df["validation_pareto"]].copy()
    if pareto.empty:
        return pd.DataFrame()
    summary = _mean_std(
        pareto,
        ["policy_id", "stage1_profile", "stage2_profile", "reserve_fraction", "budget"],
        metrics + extra,
    )
    rename = {
        "vtvf_capture_mean": "vtvf_capture",
        "all_error_capture_mean": "all_error_capture",
        "auto_vtvf_error_rate_mean": "auto_vtvf_error_rate",
        "auto_error_rate_mean": "auto_error_rate",
    }
    scored = summary.rename(columns=rename)
    scored = _mark_pareto(scored, ["budget"], "aggregate_pareto")
    stable = scored[scored["n_seeds"] >= min_pareto_seeds].copy()
    if not stable.empty:
        stable = _mark_pareto(stable, ["budget"], "stable_aggregate_pareto")
        scored = scored.merge(
            stable[["policy_id", "budget", "stable_aggregate_pareto"]],
            on=["policy_id", "budget"],
            how="left",
        )
    else:
        scored["stable_aggregate_pareto"] = False
    scored["stable_aggregate_pareto"] = scored["stable_aggregate_pareto"].where(
        scored["stable_aggregate_pareto"].notna(),
        False,
    )
    scored["stable_aggregate_pareto"] = scored["stable_aggregate_pareto"].astype(bool)
    scored["min_pareto_seeds"] = min_pareto_seeds
    return scored.rename(columns={v: k for k, v in rename.items()})


def _standardize_v5d_baselines(routing_dir: Path) -> pd.DataFrame:
    path = routing_dir / "hierarchical_router_v5d_reserved" / "all_seed_v5d_reserved_policy_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["method"].astype(str).str.startswith("hierarchical_router_v5d_reserve_")].copy()
    return df.rename(
        columns={
            "all_error_addressed": "all_error_capture",
            "vtvf_cross_error_addressed": "vtvf_capture",
            "automatic_unresolved_error_rate": "auto_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate": "auto_vtvf_error_rate",
        }
    )[
        [
            "seed",
            "budget",
            "method",
            "policy_family",
            "reserve_fraction",
            "action_rate",
            "stage1_boundary_rate",
            "stage2_residual_rate",
            "all_error_capture",
            "vtvf_capture",
            "auto_error_rate",
            "auto_vtvf_error_rate",
        ]
    ]


def _paired_against_v5d(stable_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stable_rows.empty or baseline_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = stable_rows.merge(baseline_rows, on=["seed", "budget"], suffixes=("_upgrade", "_baseline"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "seed": int(row["seed"]),
                "budget": float(row["budget"]),
                "upgrade_policy_id": row["policy_id"],
                "upgrade_stage1_profile": row["stage1_profile"],
                "upgrade_stage2_profile": row["stage2_profile"],
                "upgrade_reserve_fraction": float(row["reserve_fraction_upgrade"]),
                "baseline_method": row["method_baseline"],
                "baseline_reserve_fraction": float(row["reserve_fraction_baseline"]),
                "delta_vtvf_capture": row["vtvf_capture_upgrade"] - row["vtvf_capture_baseline"],
                "delta_all_error_capture": row["all_error_capture_upgrade"] - row["all_error_capture_baseline"],
                "delta_auto_vtvf_error_rate_reduction": row["auto_vtvf_error_rate_baseline"] - row["auto_vtvf_error_rate_upgrade"],
                "delta_auto_error_rate_reduction": row["auto_error_rate_baseline"] - row["auto_error_rate_upgrade"],
                "delta_stage1_rate": row["stage1_boundary_rate_upgrade"] - row["stage1_boundary_rate_baseline"],
                "delta_stage2_rate": row["stage2_residual_rate_upgrade"] - row["stage2_residual_rate_baseline"],
            }
        )
    effects = pd.DataFrame(rows)
    summary = _mean_std(
        effects,
        ["budget", "upgrade_policy_id", "baseline_method"],
        [
            "delta_vtvf_capture",
            "delta_all_error_capture",
            "delta_auto_vtvf_error_rate_reduction",
            "delta_auto_error_rate_reduction",
            "delta_stage1_rate",
            "delta_stage2_rate",
        ],
    )
    return effects, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal-Pareto weight upgrade inside the existing V5D stage1/stage2 router.")
    parser.add_argument("--routing-dir", type=Path, default=DEFAULT_ROUTING_DIR)
    parser.add_argument("--wavelet-dir", type=Path, default=None)
    parser.add_argument("--validity-root", type=Path, default=DEFAULT_VALIDITY_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--reserve-fractions", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--min-pareto-seeds", type=int, default=5)
    args = parser.parse_args()

    wavelet_dir = args.wavelet_dir or (args.routing_dir / "wavelet_boundary_routing_audit")
    validity_runs = _discover_validity_runs(args.validity_root)
    args.out.mkdir(parents=True, exist_ok=True)

    all_val = []
    all_test = []
    manifest = []
    for seed in args.seeds:
        if seed not in validity_runs:
            manifest.append({"seed": seed, "status": "missing_validity_run"})
            continue
        val_summary, test_summary, alignment = _run_seed(
            seed,
            args.routing_dir,
            wavelet_dir,
            validity_runs[seed],
            args.budgets,
            args.reserve_fractions,
        )
        seed_out = args.out / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        val_summary.to_csv(seed_out / "validation_v5d_weight_upgrade_candidates.csv", index=False)
        test_summary.to_csv(seed_out / "test_v5d_weight_upgrade_candidates.csv", index=False)
        all_val.append(val_summary)
        all_test.append(test_summary)
        manifest.append({"seed": seed, "status": "completed", **alignment})

    val_df = pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame()
    test_df = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    val_df.to_csv(args.out / "all_seed_validation_v5d_weight_upgrade_candidates.csv", index=False)
    test_df.to_csv(args.out / "all_seed_test_v5d_weight_upgrade_candidates.csv", index=False)
    aggregate = _aggregate(test_df, args.min_pareto_seeds) if not test_df.empty else pd.DataFrame()
    aggregate.to_csv(args.out / "aggregate_validation_pareto_test_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.out / "v5d_weight_upgrade_alignment_manifest.csv", index=False)

    stable_keys = aggregate.loc[aggregate.get("stable_aggregate_pareto", False), ["policy_id", "budget"]] if not aggregate.empty else pd.DataFrame()
    if not stable_keys.empty:
        stable_test = test_df.merge(stable_keys, on=["policy_id", "budget"], how="inner")
    else:
        stable_test = pd.DataFrame()
    stable_test.to_csv(args.out / "stable_v5d_weight_upgrade_seed_metrics.csv", index=False)
    v5d_baselines = _standardize_v5d_baselines(args.routing_dir)
    effects, effect_summary = _paired_against_v5d(stable_test, v5d_baselines)
    effects.to_csv(args.out / "paired_stable_upgrade_vs_original_v5d_seed_level.csv", index=False)
    effect_summary.to_csv(args.out / "paired_stable_upgrade_vs_original_v5d_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "wavelet_dir": str(wavelet_dir),
        "validity_root": str(args.validity_root),
        "out": str(args.out),
        "budgets": args.budgets,
        "reserve_fractions": args.reserve_fractions,
        "seeds": args.seeds,
        "n_validation_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_aggregate_rows": int(len(aggregate)),
        "n_aggregate_pareto": int(aggregate["aggregate_pareto"].sum()) if "aggregate_pareto" in aggregate else 0,
        "min_pareto_seeds": int(args.min_pareto_seeds),
        "n_stable_aggregate_pareto": int(aggregate["stable_aggregate_pareto"].sum()) if "stable_aggregate_pareto" in aggregate else 0,
        "n_stable_seed_rows": int(len(stable_test)),
        "n_paired_v5d_effect_rows": int(len(effects)),
        "intervention": "do(stage1_evidence_weights, stage2_residual_weights, residual_reserve_fraction) inside V5D",
        "comparison_unit": "complete V5D-style stage1/stage2 routing policy",
        "manifest": manifest,
    }
    (args.out / "v5d_causal_pareto_weight_upgrade_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
