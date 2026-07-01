from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .boundary_first_router_v5b import _discover_validity_runs, _load_seed_frames
from .data import build_duplicate_family_groups, extract_regularity_features_batch, load_rhythm_windows, make_splits
from .evaluate_corruption_severity import _corrupt
from .hierarchical_router_v5c import (
    _active_residual_specs,
    _greedy_unique,
    _optimize_residual_weights,
    _residual_candidate_tuples,
)
from .metrics import softmax
from .models import build_model
from .run_mechanism_routing_10seed import _resolve_runs
from .train import predict
from .wavelet_boundary_routing_audit import WAVELET_FEATURE_NAMES, _extract_wavelet_features


DEFAULT_ROUTING_DIR = Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627")
DEFAULT_VALIDITY_ROOT = Path("results/cnn_tcn_validity_20260626")
DEFAULT_OUT_DIR = Path("results/route_ood_stress_20260629")
CLASS_LABELS = {0: "SR", 1: "VT", 2: "VF"}

ECG_STRUCTURE_PRESERVING_CORRUPTIONS = [
    "gaussian_noise",
    "powerline_interference",
    "baseline_wander",
    "amplitude_scaling",
    "clipping_saturation",
    "time_scaling",
]


PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "complete_router_profile_wavelet_boundary_heavy": {
        "vtvf_boundary": 1.0,
        "wavelet_boundary": 4.0,
        "validity_boundary": 1.0,
        "sr_ventricular": 1.0,
        "representation_conflict": 1.0,
        "atypical_signal": 1.0,
        "hidden_confident": 1.0,
    },
    "complete_router_profile_validity_boundary__sr_ventricular": {
        "validity_boundary": 1.0,
        "sr_ventricular": 1.0,
    },
    "complete_router_profile_wavelet_boundary__atypical_signal": {
        "wavelet_boundary": 1.0,
        "atypical_signal": 1.0,
    },
    "entropy_ranked_review": {
        "entropy": 1.0,
    },
}

MECHANISM_SCORES = {
    "vtvf_boundary": {
        "score_col": "vtvf_boundary_mechanism_risk",
        "candidate": "vtvf_candidate",
        "action": "vtvf_boundary_set",
    },
    "wavelet_boundary": {
        "score_col": "shift_wavelet_vtvf_boundary_risk",
        "candidate": "vtvf_candidate",
        "action": "vtvf_boundary_set",
    },
    "validity_boundary": {
        "score_col": "validity_v1_gate_x_boundary",
        "candidate": "vtvf_candidate",
        "action": "vtvf_boundary_set",
    },
    "sr_ventricular": {
        "score_col": "sr_ventricular_mechanism_risk",
        "candidate": "all",
        "action": "sr_ventricular_review",
    },
    "representation_conflict": {
        "score_col": "representation_conflict_mechanism_risk",
        "candidate": "all",
        "action": "representation_review",
    },
    "atypical_signal": {
        "score_col": "atypical_signal_mechanism_risk",
        "candidate": "all",
        "action": "atypical_review",
    },
    "hidden_confident": {
        "score_col": "hidden_confident_mechanism_risk",
        "candidate": "high_confidence",
        "action": "hidden_failure_review",
    },
    "entropy": {
        "score_col": "shift_entropy",
        "candidate": "all",
        "action": "uncertainty_review",
    },
}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _safe_entropy(probs: np.ndarray) -> np.ndarray:
    return (-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / np.log(probs.shape[1])).astype(
        np.float32
    )


def _rank_margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return (ordered[:, -1] - ordered[:, -2]).astype(np.float32)


def _softmax_boundary(probs: np.ndarray) -> np.ndarray:
    ventricular = probs[:, 1] + probs[:, 2]
    return (ventricular * (1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular, 1e-12))).astype(
        np.float32
    )


def _loader(x: np.ndarray, features: np.ndarray | None, batch_size: int) -> DataLoader:
    y = np.zeros(len(x), dtype=np.int64)
    if features is None:
        ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(features.astype(np.float32)), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _features_for_factory(run_dir: Path, model_name: str):
    if model_name not in {"regularity_fusion", "reliability_gated_fusion"}:
        return lambda x: None
    scaler = np.load(run_dir / "feature_scaler.npz", allow_pickle=True)
    mean, std = scaler["mean"], scaler["std"]

    def features_for(x: np.ndarray) -> np.ndarray:
        return ((extract_regularity_features_batch(x) - mean) / std).astype(np.float32)

    return features_for


def _load_teacher_predictions(
    run_dir: Path,
    model_name: str,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = build_model(model_name).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    features_for = _features_for_factory(run_dir, model_name)
    logits, emb, _ = predict(model, _loader(x, features_for(x), batch_size), device)
    return logits.astype(np.float32), emb.astype(np.float32)


class ConstantBinaryModel:
    def __init__(self, p: float) -> None:
        self.p = float(p)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.p, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def _fit_wavelet_boundary_model(val_wavelet: pd.DataFrame, target: np.ndarray) -> Any:
    if len(np.unique(target)) < 2:
        return ConstantBinaryModel(float(np.mean(target)))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"))
    model.fit(val_wavelet[WAVELET_FEATURE_NAMES], target.astype(int))
    return model


def _robust_scale(val_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    val_score = np.asarray(val_score, dtype=float)
    score = np.asarray(score, dtype=float)
    lo = float(np.nanquantile(val_score, 0.05))
    hi = float(np.nanquantile(val_score, 0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(len(score), dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _update_shift_columns(clean_df: pd.DataFrame, logits: np.ndarray, wavelet_risk: np.ndarray) -> pd.DataFrame:
    out = clean_df.copy()
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    top2 = np.sort(np.argsort(probs, axis=1)[:, -2:], axis=1)
    out["y_pred"] = pred.astype(int)
    out["prob_sr"] = probs[:, 0]
    out["prob_vt"] = probs[:, 1]
    out["prob_vf"] = probs[:, 2]
    out["max_prob"] = probs.max(axis=1)
    out["rank_margin"] = _rank_margin(probs)
    out["entropy"] = _safe_entropy(probs)
    out["shift_entropy"] = out["entropy"]
    out["ventricular_prob"] = probs[:, 1] + probs[:, 2]
    out["softmax_vtvf_ambiguity"] = _softmax_boundary(probs)
    out["abs_prob_vtvf_margin"] = np.abs(probs[:, 1] - probs[:, 2])
    out["pred_is_vtvf"] = np.isin(pred, [1, 2]).astype(np.float32)
    out["top2_are_vtvf"] = (top2 == np.asarray([1, 2])).all(axis=1).astype(np.float32)
    out["is_error"] = out["y_true"].to_numpy(int) != pred
    out["is_vtvf_cross_error"] = ((out["y_true"].to_numpy(int) == 1) & (pred == 2)) | (
        (out["y_true"].to_numpy(int) == 2) & (pred == 1)
    )
    out["is_vtvf_truth"] = out["y_true"].isin([1, 2])
    out["is_vtvf_candidate"] = (out["pred_is_vtvf"].astype(bool) | out["top2_are_vtvf"].astype(bool)).astype(
        np.float32
    )
    out["shift_wavelet_vtvf_boundary_risk"] = wavelet_risk.astype(np.float32)
    # Residual mechanism labels are recomputed only where possible from the shifted prediction.
    out["is_sr_ventricular_error"] = out["is_error"] & (
        ((out["y_true"].to_numpy(int) == 0) & np.isin(pred, [1, 2]))
        | (np.isin(out["y_true"].to_numpy(int), [1, 2]) & (pred == 0))
    )
    return out


def _candidate_mask(df: pd.DataFrame, val_df: pd.DataFrame, mechanism: str) -> np.ndarray:
    kind = MECHANISM_SCORES[mechanism]["candidate"]
    if kind == "vtvf_candidate":
        return df["is_vtvf_candidate"].to_numpy(bool)
    if kind == "high_confidence":
        threshold = float(val_df["max_prob"].quantile(0.50)) if "max_prob" in val_df.columns else 0.0
        return df["max_prob"].to_numpy(float) >= threshold
    return np.ones(len(df), dtype=bool)


def _greedy_profile_route(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights: dict[str, float],
    budget: float,
) -> pd.DataFrame:
    n_select = max(1, int(round(len(test_df) * budget)))
    pairs: list[tuple[float, int, str]] = []
    for mechanism, weight in weights.items():
        if weight <= 0 or mechanism not in MECHANISM_SCORES:
            continue
        score_col = MECHANISM_SCORES[mechanism]["score_col"]
        if score_col not in test_df.columns or score_col not in val_df.columns:
            continue
        mask = _candidate_mask(test_df, val_df, mechanism)
        scaled = _robust_scale(val_df[score_col].to_numpy(float), test_df[score_col].to_numpy(float))
        for idx in np.flatnonzero(mask):
            pairs.append((float(scaled[idx] * weight), int(idx), mechanism))
    pairs.sort(key=lambda item: item[0], reverse=True)
    used: set[int] = set()
    selected: list[tuple[int, str, float]] = []
    for score, idx, mechanism in pairs:
        if idx in used:
            continue
        used.add(idx)
        selected.append((idx, mechanism, score))
        if len(selected) >= n_select:
            break
    routed = test_df[["sample_id", "y_true", "y_pred", "is_error", "is_vtvf_cross_error"]].copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    for idx, mechanism, _ in selected:
        actions[idx] = MECHANISM_SCORES[mechanism]["action"]
        routes[idx] = mechanism
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    return routed


def _assign_v5d(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budget: float,
    reserve_fraction: float,
) -> pd.DataFrame:
    n_total_val = max(1, int(round(len(val_df) * budget)))
    n_total_test = max(1, int(round(len(test_df) * budget)))
    reserve_val = int(round(n_total_val * reserve_fraction))
    reserve_test = int(round(n_total_test * reserve_fraction))
    requested_stage1_val = max(0, n_total_val - reserve_val)
    requested_stage1_test = max(0, n_total_test - reserve_test)

    val_soft = _robust_scale(val_df["softmax_vtvf_ambiguity"].to_numpy(float), val_df["softmax_vtvf_ambiguity"].to_numpy(float))
    test_soft = _robust_scale(val_df["softmax_vtvf_ambiguity"].to_numpy(float), test_df["softmax_vtvf_ambiguity"].to_numpy(float))
    val_wave = _robust_scale(val_df["shift_wavelet_vtvf_boundary_risk"].to_numpy(float), val_df["shift_wavelet_vtvf_boundary_risk"].to_numpy(float))
    test_wave = _robust_scale(val_df["shift_wavelet_vtvf_boundary_risk"].to_numpy(float), test_df["shift_wavelet_vtvf_boundary_risk"].to_numpy(float))
    val_valid = _robust_scale(val_df["validity_v1_gate_x_boundary"].to_numpy(float), val_df["validity_v1_gate_x_boundary"].to_numpy(float))
    test_valid = _robust_scale(val_df["validity_v1_gate_x_boundary"].to_numpy(float), test_df["validity_v1_gate_x_boundary"].to_numpy(float))
    val_boundary_score = (val_soft + val_wave + val_valid) / 3.0
    test_boundary_score = (test_soft + test_wave + test_valid) / 3.0

    def top_n(score: np.ndarray, n_select: int, candidate: np.ndarray) -> np.ndarray:
        eligible = np.flatnonzero(candidate)
        n = min(len(eligible), max(0, n_select))
        mask = np.zeros(len(score), dtype=bool)
        if n <= 0:
            return mask
        order = eligible[np.argsort(-score[eligible])]
        mask[order[:n]] = True
        return mask

    val_stage1 = top_n(val_boundary_score, requested_stage1_val, val_df["is_vtvf_candidate"].to_numpy(bool))
    test_stage1 = top_n(test_boundary_score, requested_stage1_test, test_df["is_vtvf_candidate"].to_numpy(bool))
    residual_val_slots = max(0, n_total_val - int(val_stage1.sum()))
    residual_test_slots = max(0, n_total_test - int(test_stage1.sum()))
    specs = _active_residual_specs(val_df)
    weights, _ = _optimize_residual_weights(val_df, val_stage1, specs, residual_val_slots)
    residual_picks = _greedy_unique(_residual_candidate_tuples(test_df, test_stage1, specs, weights), residual_test_slots)

    routed = test_df[["sample_id", "y_true", "y_pred", "is_error", "is_vtvf_cross_error"]].copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    actions[test_stage1] = "vtvf_boundary_set"
    routes[test_stage1] = "boundary_first"
    for idx, name, _ in residual_picks:
        actions[idx] = specs[name]["action"]
        routes[idx] = name
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    return routed


def _summarize_route(
    seed: int,
    policy: str,
    budget: float,
    corruption: str,
    severity: int,
    routed: pd.DataFrame,
) -> dict[str, Any]:
    selected = routed["mechanism_action"].to_numpy(str) != "single_label"
    auto = ~selected
    is_error = routed["is_error"].to_numpy(bool)
    is_vtvf = routed["is_vtvf_cross_error"].to_numpy(bool)
    row: dict[str, Any] = {
        "seed": seed,
        "policy": policy,
        "budget": float(budget),
        "corruption": corruption,
        "severity": int(severity),
        "n": int(len(routed)),
        "action_rate": float(selected.mean()),
        "error_rate": float(is_error.mean()),
        "vtvf_cross_error_rate": float(is_vtvf.mean()),
        "all_error_capture": float((is_error & selected).sum() / max(is_error.sum(), 1)),
        "vtvf_capture": float((is_vtvf & selected).sum() / max(is_vtvf.sum(), 1)),
        "auto_error_rate": float((is_error & auto).mean()) if auto.any() else np.nan,
        "auto_vtvf_error_rate": float((is_vtvf & auto).mean()) if auto.any() else np.nan,
        "route_diversity": int(pd.Series(routed.loc[selected, "mechanism_route"]).nunique()) if selected.any() else 0,
    }
    for route, count in routed.loc[selected, "mechanism_route"].value_counts().items():
        row[f"route_count_{route}"] = int(count)
    return row


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


def _paired_shift_drop(seed_rows: pd.DataFrame) -> pd.DataFrame:
    clean = seed_rows[(seed_rows["corruption"] == "clean") & (seed_rows["severity"] == 0)]
    shifted = seed_rows[~((seed_rows["corruption"] == "clean") & (seed_rows["severity"] == 0))]
    merged = shifted.merge(
        clean,
        on=["seed", "policy", "budget"],
        suffixes=("_shift", "_clean"),
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "seed": int(row["seed"]),
                "policy": row["policy"],
                "budget": float(row["budget"]),
                "corruption": row["corruption_shift"],
                "severity": int(row["severity_shift"]),
                "delta_vtvf_capture_shift_minus_clean": float(row["vtvf_capture_shift"] - row["vtvf_capture_clean"]),
                "delta_all_error_capture_shift_minus_clean": float(row["all_error_capture_shift"] - row["all_error_capture_clean"]),
                "delta_auto_vtvf_error_rate_shift_minus_clean": float(
                    row["auto_vtvf_error_rate_shift"] - row["auto_vtvf_error_rate_clean"]
                ),
                "delta_auto_error_rate_shift_minus_clean": float(
                    row["auto_error_rate_shift"] - row["auto_error_rate_clean"]
                ),
                "delta_error_rate_shift_minus_clean": float(row["error_rate_shift"] - row["error_rate_clean"]),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    routing_dir = Path(args.routing_dir)
    validity_runs = _discover_validity_runs(Path(args.validity_root))
    run_pairs = _resolve_runs()

    dataset = load_rhythm_windows(args.mat)
    groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruptions = args.corruptions or ECG_STRUCTURE_PRESERVING_CORRUPTIONS

    for seed in args.seeds:
        if seed not in run_pairs or seed not in validity_runs:
            manifests.append({"seed": seed, "status": "missing_run_or_validity"})
            continue
        splits = make_splits(dataset.x, dataset.y, groups=groups, seed=seed)
        teacher = _resolve_run_dir(run_pairs[seed]["teacher"])
        val_df, clean_test_df, alignment = _load_seed_frames(
            seed,
            routing_dir,
            routing_dir / "wavelet_boundary_routing_audit",
            validity_runs[seed],
        )
        if len(splits.x_test) != len(clean_test_df) or not np.array_equal(splits.y_test, clean_test_df["y_true"].to_numpy(int)):
            raise ValueError(f"Raw split does not align with mechanism evidence for seed {seed}")
        if len(splits.x_val) != len(val_df) or not np.array_equal(splits.y_val, val_df["y_true"].to_numpy(int)):
            raise ValueError(f"Raw validation split does not align with mechanism evidence for seed {seed}")

        val_wavelet = _extract_wavelet_features(splits.x_val.astype(np.float32), batch_size=args.wavelet_batch_size)
        wavelet_model = _fit_wavelet_boundary_model(val_wavelet, val_df["is_vtvf_cross_error"].to_numpy(int))
        val_df = val_df.copy()
        val_df["shift_wavelet_vtvf_boundary_risk"] = wavelet_model.predict_proba(val_wavelet[WAVELET_FEATURE_NAMES])[:, 1]
        val_df["shift_entropy"] = val_df["entropy"].astype(float)

        conditions = [("clean", 0, splits.x_test.astype(np.float32))]
        rng = np.random.default_rng(seed)
        for corruption in corruptions:
            for severity in args.severities:
                conditions.append((corruption, int(severity), _corrupt(splits.x_test.astype(np.float32), corruption, int(severity), rng).astype(np.float32)))

        for corruption, severity, x_eval in conditions:
            logits, _ = _load_teacher_predictions(
                teacher,
                args.model,
                x_eval,
                args.batch_size,
                device,
            )
            test_wavelet = _extract_wavelet_features(x_eval, batch_size=args.wavelet_batch_size)
            wavelet_risk = wavelet_model.predict_proba(test_wavelet[WAVELET_FEATURE_NAMES])[:, 1]
            test_df = _update_shift_columns(clean_test_df, logits, wavelet_risk)
            for budget in args.budgets:
                for reserve in args.v5d_reserve_fractions:
                    routed = _assign_v5d(val_df, test_df, budget, reserve)
                    all_rows.append(
                        _summarize_route(
                            seed,
                            f"v5d_reserve_{int(round(reserve * 100))}pct",
                            budget,
                            corruption,
                            severity,
                            routed,
                        )
                    )
                for profile, weights in PROFILE_WEIGHTS.items():
                    routed = _greedy_profile_route(val_df, test_df, weights, budget)
                    all_rows.append(_summarize_route(seed, profile, budget, corruption, severity, routed))

        manifests.append(
            {
                "seed": seed,
                "status": "completed",
                "teacher_run": str(teacher),
                "n_val": int(len(val_df)),
                "n_test": int(len(clean_test_df)),
                **alignment,
            }
        )

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "route_ood_stress_seed_level.csv", index=False)
    metric_cols = [
        "action_rate",
        "error_rate",
        "vtvf_cross_error_rate",
        "all_error_capture",
        "vtvf_capture",
        "auto_error_rate",
        "auto_vtvf_error_rate",
        "route_diversity",
    ]
    summary = _mean_std(all_df, ["policy", "budget", "corruption", "severity"], metric_cols)
    summary.to_csv(out_dir / "route_ood_stress_mean_std.csv", index=False)
    drops = _paired_shift_drop(all_df)
    drops.to_csv(out_dir / "route_ood_shift_drop_seed_level.csv", index=False)
    if not drops.empty:
        _mean_std(
            drops,
            ["policy", "budget", "corruption", "severity"],
            [
                "delta_vtvf_capture_shift_minus_clean",
                "delta_all_error_capture_shift_minus_clean",
                "delta_auto_vtvf_error_rate_shift_minus_clean",
                "delta_auto_error_rate_shift_minus_clean",
                "delta_error_rate_shift_minus_clean",
            ],
        ).to_csv(out_dir / "route_ood_shift_drop_mean_std.csv", index=False)

    manifest = {
        "out_dir": str(out_dir),
        "routing_dir": str(routing_dir),
        "validity_root": str(args.validity_root),
        "seeds": args.seeds,
        "budgets": args.budgets,
        "v5d_reserve_fractions": args.v5d_reserve_fractions,
        "corruptions": corruptions,
        "severities": args.severities,
        "ecg_structure_preserving_scope": (
            "Default corruptions preserve ECG time order and morphology better than shuffled/flatline stress. "
            "They are OOD-style internal shifts, not external clinical validation."
        ),
        "policy_design_boundary": (
            "Policies named complete_router_profile_* are full routing policies created by wrapping "
            "evidence-weight profiles with candidate masks and recovery actions. They are not raw evidence heads."
        ),
        "dynamic_evidence": "shifted softmax, shifted ECG wavelet risk, shifted y_pred/error labels",
        "static_or_proxy_evidence": "validity gate and residual mechanism risk heads are reused from clean evidence in this pilot",
        "seed_manifest": manifests,
    }
    (out_dir / "route_ood_stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "n_seed_rows": int(len(all_df)),
        "n_summary_rows": int(len(summary)),
        "n_drop_rows": int(len(drops)),
        "completed_seeds": [m["seed"] for m in manifests if m.get("status") == "completed"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route-level OOD stress test for V5D and complete causal-Pareto routing profiles."
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--routing-dir", type=Path, default=DEFAULT_ROUTING_DIR)
    parser.add_argument("--validity-root", type=Path, default=DEFAULT_VALIDITY_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="reliability_gated_fusion")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--v5d-reserve-fractions", type=float, nargs="+", default=[0.0, 0.20, 0.30])
    parser.add_argument("--corruptions", nargs="+", default=None)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wavelet-batch-size", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
