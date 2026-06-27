from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .top_journal_reliability_directions import (
    RunSpec,
    _atlas_row,
    _bootstrap_ci,
    _paired_delta_summary,
    _summarise_atlas,
)


def _complete_run(run_dir: Path) -> bool:
    required = [
        "metrics.json",
        "embeddings_train.npz",
        "embeddings_test.npz",
        "test_predictions.csv",
        "validity_gate_scores_test.csv",
        "history.csv",
        "best_model.pt",
    ]
    return all((run_dir / name).exists() for name in required)


def _seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def _discover_validity_runs(root: Path) -> list[RunSpec]:
    candidates: dict[int, Path] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not _complete_run(run_dir):
            continue
        seed = _seed_from_name(run_dir)
        if seed is None:
            continue
        previous = candidates.get(seed)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            candidates[seed] = run_dir
    return [
        RunSpec(
            family="cnn_tcn_validity",
            model="CNN-TCN-Validity",
            seed=seed,
            run_dir=run_dir,
            pair_group=f"cnn_tcn_validity_seed{seed}",
        )
        for seed, run_dir in sorted(candidates.items())
    ]


def _safe_binary_metric(y: np.ndarray, score: np.ndarray, metric_fn) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(metric_fn(y[mask], score[mask]))


def _gate_row(run: RunSpec) -> dict:
    gate = pd.read_csv(run.run_dir / "validity_gate_scores_test.csv")
    gate["is_error"] = gate["y_true"] != gate["y_pred"]
    gate["is_vtvf"] = gate["y_true"].isin([1, 2])
    gate["is_vtvf_boundary_error"] = (
        ((gate["y_true"] == 1) & (gate["y_pred"] == 2))
        | ((gate["y_true"] == 2) & (gate["y_pred"] == 1))
    )
    gate["is_confident_error"] = gate["is_error"] & (gate["confidence"] >= 0.90)
    vmask = gate["is_vtvf"].to_numpy()
    return {
        "family": run.family,
        "model": run.model,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "gate_mean": float(gate["validity_gate"].mean()),
        "gate_correct_mean": float(gate.loc[~gate["is_error"], "validity_gate"].mean()),
        "gate_error_mean": float(gate.loc[gate["is_error"], "validity_gate"].mean()),
        "gate_vtvf_correct_mean": float(gate.loc[gate["is_vtvf"] & ~gate["is_vtvf_boundary_error"], "validity_gate"].mean()),
        "gate_vtvf_boundary_error_mean": float(gate.loc[gate["is_vtvf_boundary_error"], "validity_gate"].mean()),
        "boundary_score_mean": float(gate["boundary_score"].mean()),
        "boundary_score_correct_mean": float(gate.loc[~gate["is_error"], "boundary_score"].mean()),
        "boundary_score_error_mean": float(gate.loc[gate["is_error"], "boundary_score"].mean()),
        "gate_any_error_auroc": _safe_binary_metric(gate["is_error"].to_numpy(), gate["validity_gate"].to_numpy(), roc_auc_score),
        "gate_any_error_aupr": _safe_binary_metric(gate["is_error"].to_numpy(), gate["validity_gate"].to_numpy(), average_precision_score),
        "gate_vtvf_boundary_auroc": _safe_binary_metric(
            gate.loc[vmask, "is_vtvf_boundary_error"].to_numpy(),
            gate.loc[vmask, "validity_gate"].to_numpy(),
            roc_auc_score,
        )
        if vmask.any()
        else float("nan"),
        "gate_vtvf_boundary_aupr": _safe_binary_metric(
            gate.loc[vmask, "is_vtvf_boundary_error"].to_numpy(),
            gate.loc[vmask, "validity_gate"].to_numpy(),
            average_precision_score,
        )
        if vmask.any()
        else float("nan"),
        "boundary_score_any_error_auroc": _safe_binary_metric(
            gate["is_error"].to_numpy(),
            gate["boundary_score"].to_numpy(),
            roc_auc_score,
        ),
        "boundary_score_vtvf_boundary_auroc": _safe_binary_metric(
            gate.loc[vmask, "is_vtvf_boundary_error"].to_numpy(),
            gate.loc[vmask, "boundary_score"].to_numpy(),
            roc_auc_score,
        )
        if vmask.any()
        else float("nan"),
        "n_confident_errors": int(gate["is_confident_error"].sum()),
    }


def _model_metric_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "macro_f1",
        "ece",
        "total_errors",
        "vtvf_cross_errors",
        "embedding_silhouette",
        "vt_vf_normalized_separation",
        "softmax_vtvf_boundary_auroc",
        "knn_vtvf_boundary_auroc",
    ]
    rows = []
    for (family, model), sub in atlas.groupby(["family", "model"], sort=False):
        for metric in metrics:
            if metric not in sub:
                continue
            values = sub[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            lo, hi = _bootstrap_ci(values)
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_deltas(atlas: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("cnn_lstm", "CNN", "CNN-TCN-Validity"),
        ("cnn_lstm", "CNN-LSTM", "CNN-TCN-Validity"),
        ("pro_risk", "Teacher", "CNN-TCN-Validity"),
        ("pro_risk", "ProRisk", "CNN-TCN-Validity"),
    ]
    metrics = [
        "accuracy",
        "macro_f1",
        "ece",
        "total_errors",
        "vtvf_cross_errors",
        "embedding_silhouette",
        "vt_vf_normalized_separation",
        "softmax_vtvf_boundary_auroc",
        "knn_vtvf_boundary_auroc",
    ]
    ctv = atlas[atlas["model"].eq("CNN-TCN-Validity")]
    rows = []
    for family, baseline, comparator in pairs:
        base = atlas[atlas["model"].eq(baseline)]
        comp = ctv if comparator == "CNN-TCN-Validity" else atlas[atlas["model"].eq(comparator)]
        merged = base.merge(comp, on="seed", suffixes=("_baseline", "_comparator"))
        for _, row in merged.iterrows():
            out = {"baseline_family": family, "baseline": baseline, "comparator": comparator, "seed": int(row["seed"])}
            for metric in metrics:
                left = f"{metric}_baseline"
                right = f"{metric}_comparator"
                if left in row and right in row:
                    out[f"{metric}_delta"] = row[right] - row[left]
            rows.append(out)
    return pd.DataFrame(rows)


def _delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, baseline, comparator), sub in deltas.groupby(["baseline_family", "baseline", "comparator"], sort=False):
        for col in sub.columns:
            if not col.endswith("_delta"):
                continue
            vals = sub[col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            rows.append(
                {
                    "baseline_family": family,
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": col,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "median": float(np.median(vals)),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_positive": int((vals > 0).sum()),
                    "n_negative": int((vals < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CNN+TCN+Validity 10seed comparison.")
    parser.add_argument("--validity-root", type=Path, default=Path("results/cnn_tcn_validity_20260626"))
    parser.add_argument(
        "--existing-atlas",
        type=Path,
        default=Path("results/top_journal_reliability_directions_20260626/direction1_representation_underspecification_atlas.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/cnn_tcn_validity_20260626/summary"))
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    runs = _discover_validity_runs(args.validity_root)
    if len(runs) < 10:
        raise SystemExit(f"Expected 10 complete validity runs, found {len(runs)}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame([{"seed": r.seed, "run_dir": str(r.run_dir)} for r in runs])
    manifest.to_csv(args.out / "cnn_tcn_validity_complete_manifest.csv", index=False)

    ctv_atlas = pd.DataFrame([_atlas_row(run, k=args.k) for run in runs])
    ctv_atlas.to_csv(args.out / "cnn_tcn_validity_atlas.csv", index=False)
    ctv_summary = _summarise_atlas(ctv_atlas)
    ctv_summary.to_csv(args.out / "cnn_tcn_validity_atlas_summary.csv", index=False)

    gate = pd.DataFrame([_gate_row(run) for run in runs])
    gate.to_csv(args.out / "cnn_tcn_validity_gate_run_level.csv", index=False)
    gate_summary = _model_metric_summary(gate.rename(columns={"gate_any_error_auroc": "accuracy"}))
    # Keep a clearer gate-specific summary alongside the generic helper output.
    gate_metric_rows = []
    for metric in [col for col in gate.columns if col not in {"family", "model", "seed", "run_dir"}]:
        if not pd.api.types.is_numeric_dtype(gate[metric]):
            continue
        vals = gate[metric].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        lo, hi = _bootstrap_ci(vals)
        gate_metric_rows.append(
            {
                "metric": metric,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    pd.DataFrame(gate_metric_rows).to_csv(args.out / "cnn_tcn_validity_gate_summary.csv", index=False)

    existing = pd.read_csv(args.existing_atlas)
    combined = pd.concat([existing, ctv_atlas], ignore_index=True, sort=False)
    combined.to_csv(args.out / "combined_model_atlas.csv", index=False)
    _model_metric_summary(combined).to_csv(args.out / "combined_model_metric_summary.csv", index=False)

    deltas = _pairwise_deltas(combined)
    deltas.to_csv(args.out / "cnn_tcn_validity_paired_deltas.csv", index=False)
    _delta_summary(deltas).to_csv(args.out / "cnn_tcn_validity_paired_delta_summary.csv", index=False)

    report = {
        "n_validity_runs": len(runs),
        "seeds": [r.seed for r in runs],
        "outputs": {
            "manifest": str(args.out / "cnn_tcn_validity_complete_manifest.csv"),
            "ctv_atlas": str(args.out / "cnn_tcn_validity_atlas.csv"),
            "gate_summary": str(args.out / "cnn_tcn_validity_gate_summary.csv"),
            "combined_summary": str(args.out / "combined_model_metric_summary.csv"),
            "paired_delta_summary": str(args.out / "cnn_tcn_validity_paired_delta_summary.csv"),
        },
    }
    (args.out / "cnn_tcn_validity_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
