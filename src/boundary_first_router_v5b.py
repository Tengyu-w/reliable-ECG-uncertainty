from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = list(range(42, 52))
CLASS_LABELS = {0: "SR", 1: "VT", 2: "VF"}


def _seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def _discover_validity_runs(root: Path) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    if not root.exists():
        return runs
    for run_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        if not run_dir.is_dir():
            continue
        seed = _seed_from_name(run_dir)
        if seed is None:
            continue
        if (run_dir / "validity_gate_scores_val.csv").exists() and (run_dir / "validity_gate_scores_test.csv").exists():
            runs[seed] = run_dir
    return runs


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _top_budget_mask(score: np.ndarray, budget: float, candidate: np.ndarray | None = None) -> np.ndarray:
    if candidate is None:
        candidate = np.ones(len(score), dtype=bool)
    eligible = np.flatnonzero(candidate)
    n = min(len(eligible), max(1, int(round(len(score) * budget))))
    mask = np.zeros(len(score), dtype=bool)
    if n <= 0:
        return mask
    order = eligible[np.argsort(-score[eligible])]
    mask[order[:n]] = True
    return mask


def _review_metrics(seed: int, method: str, score: np.ndarray, df: pd.DataFrame, budget: float) -> dict[str, Any]:
    candidate = df["is_vtvf_candidate"].to_numpy(bool) if "is_vtvf_candidate" in df.columns else None
    mask = _top_budget_mask(score, budget, candidate=candidate)
    auto = ~mask
    errors = df["is_error"].to_numpy(bool)
    vtvf = df["is_vtvf_cross_error"].to_numpy(bool)
    y = df["y_true"].to_numpy(int)
    pred = df["y_pred"].to_numpy(int)
    vt_to_vf = (y == 1) & (pred == 2)
    vf_to_vt = (y == 2) & (pred == 1)
    return {
        "seed": seed,
        "method": method,
        "policy_family": "boundary_first_set",
        "budget": budget,
        "action_rate": float(mask.mean()),
        "vtvf_set_rate": float(mask.mean()),
        "all_error_addressed": float((errors & mask).sum() / max(errors.sum(), 1)),
        "vtvf_cross_error_addressed": float((vtvf & mask).sum() / max(vtvf.sum(), 1)),
        "vt_to_vf_addressed": float((vt_to_vf & mask).sum() / max(vt_to_vf.sum(), 1)),
        "vf_to_vt_addressed": float((vf_to_vt & mask).sum() / max(vf_to_vt.sum(), 1)),
        "automatic_unresolved_error_rate": float((errors & auto).mean()) if auto.any() else np.nan,
        "automatic_unresolved_vtvf_cross_error_rate": float((vtvf & auto).mean()) if auto.any() else np.nan,
        "single_label_error_rate_after_routing": float((errors & auto).sum() / max(auto.sum(), 1)),
        "single_label_vtvf_cross_error_rate_after_routing": float((vtvf & auto).sum() / max(auto.sum(), 1)),
    }


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


def _robust_scale_from_val(val: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo = float(np.nanquantile(val, 0.05))
    hi = float(np.nanquantile(val, 0.95))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _add_validity_columns(df: pd.DataFrame, gate: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_validity_gate"] = gate["validity_gate"].to_numpy(float)
    out[f"{prefix}_boundary_score"] = gate["boundary_score"].to_numpy(float)
    out[f"{prefix}_confidence"] = gate["confidence"].to_numpy(float)
    out[f"{prefix}_gate_x_boundary"] = (
        gate["validity_gate"].to_numpy(float) * gate["boundary_score"].to_numpy(float)
    )
    out[f"{prefix}_gate_minus_confidence"] = (
        gate["validity_gate"].to_numpy(float) + (1.0 - gate["confidence"].to_numpy(float))
    )
    return out


def _load_seed_frames(seed: int, routing_dir: Path, wavelet_dir: Path, validity_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    val_path = wavelet_dir / f"seed{seed}" / "evidence_scores_val_with_wavelet.csv"
    test_path = wavelet_dir / f"seed{seed}" / "evidence_scores_test_with_wavelet.csv"
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    val_gate = pd.read_csv(validity_run / "validity_gate_scores_val.csv")
    test_gate = pd.read_csv(validity_run / "validity_gate_scores_test.csv")
    alignment = {
        "seed": seed,
        "validity_run": str(validity_run),
        "val_y_true_aligned": bool(len(val_gate) == len(val_df) and np.array_equal(val_gate["y_true"].to_numpy(int), val_df["y_true"].to_numpy(int))),
        "test_y_true_aligned": bool(len(test_gate) == len(test_df) and np.array_equal(test_gate["y_true"].to_numpy(int), test_df["y_true"].to_numpy(int))),
        "val_y_pred_same_as_teacher": bool(len(val_gate) == len(val_df) and np.array_equal(val_gate["y_pred"].to_numpy(int), val_df["y_pred"].to_numpy(int))),
        "test_y_pred_same_as_teacher": bool(len(test_gate) == len(test_df) and np.array_equal(test_gate["y_pred"].to_numpy(int), test_df["y_pred"].to_numpy(int))),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    if not alignment["val_y_true_aligned"] or not alignment["test_y_true_aligned"]:
        raise ValueError(f"Validity gate split does not align for seed {seed}: {alignment}")
    val_df = _add_validity_columns(val_df, val_gate, "validity_v1")
    test_df = _add_validity_columns(test_df, test_gate, "validity_v1")
    return val_df, test_df, alignment


def _fit_boundary_ensemble(val_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = val_df["is_vtvf_cross_error"].to_numpy(int)
    if len(np.unique(y)) < 2:
        val_score = np.full(len(val_df), y.mean(), dtype=np.float32)
        test_score = np.full(len(test_df), y.mean(), dtype=np.float32)
        return val_score, test_score, {"model": "constant", "features": features}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"))
    model.fit(val_df[features], y)
    val_score = model.predict_proba(val_df[features])[:, 1]
    test_score = model.predict_proba(test_df[features])[:, 1]
    coef = model.named_steps["logisticregression"].coef_[0].tolist()
    return val_score, test_score, {"model": "logistic_regression", "features": features, "coefficients": coef}


def _score_dict(val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    val_scores: dict[str, np.ndarray] = {
        "softmax_vtvf_ambiguity": val_df["softmax_vtvf_ambiguity"].to_numpy(float),
        "validity_gate": val_df["validity_v1_validity_gate"].to_numpy(float),
        "validity_boundary_score": val_df["validity_v1_boundary_score"].to_numpy(float),
        "validity_gate_x_boundary": val_df["validity_v1_gate_x_boundary"].to_numpy(float),
        "wavelet_vtvf_boundary_risk": val_df["wavelet_vtvf_boundary_risk"].to_numpy(float),
    }
    test_scores: dict[str, np.ndarray] = {
        "softmax_vtvf_ambiguity": test_df["softmax_vtvf_ambiguity"].to_numpy(float),
        "validity_gate": test_df["validity_v1_validity_gate"].to_numpy(float),
        "validity_boundary_score": test_df["validity_v1_boundary_score"].to_numpy(float),
        "validity_gate_x_boundary": test_df["validity_v1_gate_x_boundary"].to_numpy(float),
        "wavelet_vtvf_boundary_risk": test_df["wavelet_vtvf_boundary_risk"].to_numpy(float),
    }
    scaled_val = {name: _robust_scale_from_val(score, score) for name, score in val_scores.items()}
    scaled_test = {name: _robust_scale_from_val(val_scores[name], test_scores[name]) for name in test_scores}
    val_scores["mean_softmax_validity_wavelet"] = (
        scaled_val["softmax_vtvf_ambiguity"]
        + scaled_val["validity_gate_x_boundary"]
        + scaled_val["wavelet_vtvf_boundary_risk"]
    ) / 3.0
    test_scores["mean_softmax_validity_wavelet"] = (
        scaled_test["softmax_vtvf_ambiguity"]
        + scaled_test["validity_gate_x_boundary"]
        + scaled_test["wavelet_vtvf_boundary_risk"]
    ) / 3.0
    val_scores["max_softmax_validity_wavelet"] = np.maximum.reduce(
        [
            scaled_val["softmax_vtvf_ambiguity"],
            scaled_val["validity_gate_x_boundary"],
            scaled_val["wavelet_vtvf_boundary_risk"],
        ]
    )
    test_scores["max_softmax_validity_wavelet"] = np.maximum.reduce(
        [
            scaled_test["softmax_vtvf_ambiguity"],
            scaled_test["validity_gate_x_boundary"],
            scaled_test["wavelet_vtvf_boundary_risk"],
        ]
    )
    ensemble_features = [
        "softmax_vtvf_ambiguity",
        "abs_prob_vtvf_margin",
        "rank_margin",
        "validity_v1_validity_gate",
        "validity_v1_boundary_score",
        "validity_v1_gate_x_boundary",
        "wavelet_vtvf_boundary_risk",
    ]
    val_ensemble, test_ensemble, info = _fit_boundary_ensemble(val_df, test_df, ensemble_features)
    val_scores["learned_boundary_ensemble_v5b"] = val_ensemble
    test_scores["learned_boundary_ensemble_v5b"] = test_ensemble
    return val_scores, test_scores, [info]


def _route_assignments(df: pd.DataFrame, score: np.ndarray, budget: float) -> pd.DataFrame:
    routed = df.copy()
    candidate = routed["is_vtvf_candidate"].to_numpy(bool) if "is_vtvf_candidate" in routed.columns else None
    mask = _top_budget_mask(score, budget, candidate=candidate)
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    output = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    actions[mask] = "vtvf_boundary_set"
    routes[mask] = "boundary_first_v5b"
    output[mask] = "{VT,VF}"
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    routed["mechanism_output_set"] = output
    routed["mechanism_strategy"] = "boundary_first_v5b"
    routed["boundary_first_score"] = score
    routed["boundary_first_selected"] = mask
    routed["budget"] = budget
    return routed


def _read_v4(seed: int, routing_dir: Path) -> pd.DataFrame:
    path = routing_dir / f"seed{seed}" / "optimized_mechanism_layered_policy_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "seed": seed,
            "method": "optimized_mechanism_router_v4",
            "policy_family": "mechanism_router",
            "budget": df["budget"].astype(float),
            "action_rate": df["mechanism_action_rate"].astype(float),
            "vtvf_set_rate": df["vtvf_set_rate"].astype(float),
            "all_error_addressed": df["all_error_addressed"].astype(float),
            "vtvf_cross_error_addressed": df["vtvf_cross_error_addressed"].astype(float),
            "automatic_unresolved_error_rate": df["automatic_unresolved_error_rate"].astype(float),
            "automatic_unresolved_vtvf_cross_error_rate": df["automatic_unresolved_vtvf_cross_error_rate"].astype(float),
        }
    )


def _run_seed(
    seed: int,
    routing_dir: Path,
    wavelet_dir: Path,
    validity_run: Path,
    out_dir: Path,
    budgets: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    val_df, test_df, alignment = _load_seed_frames(seed, routing_dir, wavelet_dir, validity_run)
    val_scores, test_scores, ensemble_info = _score_dict(val_df, test_df)
    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(seed_out / "evidence_scores_test_boundary_first.csv", index=False)
    val_df.to_csv(seed_out / "evidence_scores_val_boundary_first.csv", index=False)

    diagnostic_rows = []
    for method, test_score in test_scores.items():
        diagnostic_rows.append(
            {
                "seed": seed,
                "method": method,
                "test_auroc": _safe_auc(test_df["is_vtvf_cross_error"].to_numpy(int), test_score),
                "test_aupr": _safe_aupr(test_df["is_vtvf_cross_error"].to_numpy(int), test_score),
                "val_auroc": _safe_auc(val_df["is_vtvf_cross_error"].to_numpy(int), val_scores[method]),
                "val_aupr": _safe_aupr(val_df["is_vtvf_cross_error"].to_numpy(int), val_scores[method]),
            }
        )

    rows = []
    routed_frames = []
    for method, score in test_scores.items():
        for budget in budgets:
            rows.append(_review_metrics(seed, method, score, test_df, budget))
            if method in {"learned_boundary_ensemble_v5b", "mean_softmax_validity_wavelet"}:
                routed_frames.append(_route_assignments(test_df, score, budget))
                routed_frames[-1]["boundary_first_method"] = method
    v4 = _read_v4(seed, routing_dir)
    if not v4.empty:
        rows.extend(v4.to_dict(orient="records"))
    pd.DataFrame(rows).to_csv(seed_out / "boundary_first_policy_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(seed_out / "boundary_signal_diagnostics.csv", index=False)
    pd.DataFrame(ensemble_info).to_json(seed_out / "boundary_ensemble_model.json", orient="records", indent=2)
    if routed_frames:
        pd.concat(routed_frames, ignore_index=True).to_csv(seed_out / "boundary_first_routing_assignments_test.csv", index=False)
    return rows, diagnostic_rows, ensemble_info, alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-first v5b router with softmax, validity, and wavelet evidence.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--wavelet-dir", type=Path, default=None)
    parser.add_argument("--validity-root", type=Path, default=Path("results/cnn_tcn_validity_20260626"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    wavelet_dir = args.wavelet_dir or (args.routing_dir / "wavelet_boundary_routing_audit")
    out_dir = args.out or (args.routing_dir / "boundary_first_router_v5b")
    out_dir.mkdir(parents=True, exist_ok=True)
    validity_runs = _discover_validity_runs(args.validity_root)

    all_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_ensemble: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for seed in SEEDS:
        if seed not in validity_runs:
            manifest.append({"seed": seed, "status": "missing_validity_run"})
            continue
        rows, diagnostics, ensemble_info, alignment = _run_seed(
            seed, args.routing_dir, wavelet_dir, validity_runs[seed], out_dir, args.budgets
        )
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)
        all_ensemble.extend({"seed": seed, **info} for info in ensemble_info)
        manifest.append({"seed": seed, "status": "completed", **alignment})

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_seed_boundary_first_policy_summary.csv", index=False)
    pd.DataFrame(all_diagnostics).to_csv(out_dir / "all_seed_boundary_signal_diagnostics.csv", index=False)
    pd.DataFrame(all_ensemble).to_json(out_dir / "all_seed_boundary_ensemble_models.json", orient="records", indent=2)
    pd.DataFrame(manifest).to_csv(out_dir / "boundary_first_alignment_manifest.csv", index=False)

    metric_cols = [
        "action_rate",
        "vtvf_set_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "vt_to_vf_addressed",
        "vf_to_vt_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
        "single_label_error_rate_after_routing",
        "single_label_vtvf_cross_error_rate_after_routing",
    ]
    summary = _mean_std(all_df, ["method", "policy_family", "budget"], [c for c in metric_cols if c in all_df.columns])
    summary.to_csv(out_dir / "boundary_first_policy_mean_std.csv", index=False)
    diagnostic_summary = _mean_std(
        pd.DataFrame(all_diagnostics),
        ["method"],
        ["test_auroc", "test_aupr", "val_auroc", "val_aupr"],
    )
    diagnostic_summary.to_csv(out_dir / "boundary_signal_diagnostics_mean_std.csv", index=False)

    for main_method, stem in [
        ("learned_boundary_ensemble_v5b", "paired_v5b_boundary_first_vs_baselines"),
        ("mean_softmax_validity_wavelet", "paired_v5b_recommended_mean_vs_baselines"),
    ]:
        paired_rows = []
        main = all_df[all_df["method"].eq(main_method)]
        for method in sorted(set(all_df["method"]) - {main_method}):
            other = all_df[all_df["method"].eq(method)]
            merged = main.merge(other, on=["seed", "budget"], suffixes=("_v5b", "_baseline"))
            for _, row in merged.iterrows():
                paired_rows.append(
                    {
                        "seed": int(row["seed"]),
                        "budget": float(row["budget"]),
                        "main_method": main_method,
                        "baseline_method": method,
                        "delta_all_error_addressed_v5b_minus_baseline": row["all_error_addressed_v5b"]
                        - row["all_error_addressed_baseline"],
                        "delta_vtvf_cross_error_addressed_v5b_minus_baseline": row["vtvf_cross_error_addressed_v5b"]
                        - row["vtvf_cross_error_addressed_baseline"],
                        "delta_unresolved_vtvf_v5b_minus_baseline": row[
                            "automatic_unresolved_vtvf_cross_error_rate_v5b"
                        ]
                        - row["automatic_unresolved_vtvf_cross_error_rate_baseline"],
                        "delta_action_rate_v5b_minus_baseline": row["action_rate_v5b"] - row["action_rate_baseline"],
                    }
                )
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(out_dir / f"{stem}.csv", index=False)
        if not paired.empty:
            _mean_std(
                paired.rename(columns={"baseline_method": "method"}),
                ["method", "budget"],
                [
                    "delta_all_error_addressed_v5b_minus_baseline",
                    "delta_vtvf_cross_error_addressed_v5b_minus_baseline",
                    "delta_unresolved_vtvf_v5b_minus_baseline",
                    "delta_action_rate_v5b_minus_baseline",
                ],
            ).to_csv(out_dir / f"{stem}_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "wavelet_dir": str(wavelet_dir),
        "validity_root": str(args.validity_root),
        "out_dir": str(out_dir),
        "budgets": args.budgets,
        "n_rows": int(len(all_df)),
        "manifest": manifest,
    }
    (out_dir / "boundary_first_router_v5b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.sort_values(["budget", "vtvf_cross_error_addressed_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
