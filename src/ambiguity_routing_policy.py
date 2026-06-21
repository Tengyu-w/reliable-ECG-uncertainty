from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import softmax


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _softmax_boundary(probs: np.ndarray) -> np.ndarray:
    ventricular_prob = probs[:, 1] + probs[:, 2]
    balance = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_prob, 1e-8)
    return (ventricular_prob * balance).astype(np.float32)


def _load_base(run_dir: Path) -> pd.DataFrame:
    stability_path = run_dir / "stability_scores.csv"
    if stability_path.exists():
        return pd.read_csv(stability_path)

    pred = pd.read_csv(run_dir / "test_predictions.csv")
    prob_cols = [c for c in pred.columns if c.startswith("prob_")]
    probs = pred[prob_cols].to_numpy(float)
    return pd.DataFrame(
        {
            "index": np.arange(len(pred)),
            "y_true": pred["y_true"].to_numpy(int),
            "y_pred": pred["y_pred"].to_numpy(int),
            "is_error": pred["y_true"].to_numpy(int) != pred["y_pred"].to_numpy(int),
            "is_vtvf_cross_error": (
                ((pred["y_true"] == 1) & (pred["y_pred"] == 2))
                | ((pred["y_true"] == 2) & (pred["y_pred"] == 1))
            ).to_numpy(bool),
            "max_softmax_prob": probs.max(axis=1),
            "entropy": -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1),
            "softmax_vtvf_ambiguity": _softmax_boundary(probs),
        }
    )


def _load_conformal_vtvf_set(run_dir: Path, method: str, alpha: float, n: int) -> np.ndarray:
    path = run_dir / "conformal_sets.csv"
    if not path.exists():
        return np.zeros(n, dtype=bool)
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & np.isclose(df["alpha"].astype(float), alpha)]
    if df.empty:
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    for row in df.to_dict(orient="records"):
        idx = int(row["index"])
        label_set = str(row["set"])
        out[idx] = "1" in label_set and "2" in label_set and int(row["set_size"]) > 1
    return out


def _with_default(df: pd.DataFrame, col: str, value: float = 0.0) -> np.ndarray:
    if col in df:
        return df[col].to_numpy(float)
    return np.full(len(df), value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--conformal-method", type=str, default="lac")
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument("--auto-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--boundary-threshold", type=float, default=0.60)
    parser.add_argument("--atypical-threshold", type=float, default=0.70)
    parser.add_argument("--signal-quality-threshold", type=float, default=0.65)
    parser.add_argument("--stability-risk-threshold", type=float, default=0.65)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    df = _load_base(run_dir).copy()
    n = len(df)

    boundary = _normalise(_with_default(df, "softmax_vtvf_ambiguity"))
    vtvf_mixing = _normalise(_with_default(df, "deployable_vtvf_mixing"))
    atypicality = _normalise(_with_default(df, "knn"))
    instability = _normalise(_with_default(df, "deployable_local_instability"))
    stability_risk = _normalise(_with_default(df, "stability_risk"))
    flip = _normalise(_with_default(df, "pred_flip_rate"))
    drift = _normalise(_with_default(df, "embedding_drift"))
    conformal_vtvf = _load_conformal_vtvf_set(run_dir, args.conformal_method, args.conformal_alpha, n)

    predicted_ventricular = df["y_pred"].isin([1, 2]).to_numpy(bool)
    high_conf = df["max_softmax_prob"].to_numpy(float) >= args.auto_confidence_threshold
    boundary_signal = predicted_ventricular & (
        (boundary >= args.boundary_threshold) | (vtvf_mixing >= args.boundary_threshold) | conformal_vtvf
    )
    signal_quality_signal = (
        (instability >= args.signal_quality_threshold)
        | (stability_risk >= args.stability_risk_threshold)
        | (flip >= args.signal_quality_threshold)
        | (drift >= args.signal_quality_threshold)
    )
    forced_expert_signal = high_conf & (atypicality >= args.atypical_threshold)

    route = np.full(n, "automatic_single_label", dtype=object)
    output = np.asarray([str(v) for v in df["y_pred"].to_numpy(int)], dtype=object)

    route[boundary_signal] = "boundary_review"
    output[boundary_signal] = "{VT,VF}"
    route[signal_quality_signal] = "signal_quality_review"
    output[signal_quality_signal] = "review:signal_quality"
    route[forced_expert_signal] = "forced_expert_review"
    output[forced_expert_signal] = "review:forced_expert"

    df["boundary_signal"] = boundary_signal
    df["signal_quality_signal"] = signal_quality_signal
    df["forced_expert_signal"] = forced_expert_signal
    df["conformal_vtvf_set"] = conformal_vtvf
    df["ambiguity_routing_decision"] = route
    df["ambiguity_aware_output"] = output
    df.to_csv(run_dir / "ambiguity_routing_policy.csv", index=False)

    summary_rows = []
    for decision, sub in df.groupby("ambiguity_routing_decision"):
        summary_rows.append(
            {
                "decision": decision,
                "n": int(len(sub)),
                "rate": float(len(sub) / max(n, 1)),
                "error_rate": float(sub["is_error"].mean()) if "is_error" in sub else np.nan,
                "vtvf_cross_error_rate": float(sub["is_vtvf_cross_error"].mean()) if "is_vtvf_cross_error" in sub else np.nan,
                "vtvf_cross_errors": int(sub["is_vtvf_cross_error"].sum()) if "is_vtvf_cross_error" in sub else 0,
            }
        )
    summary = {
        "run_dir": str(run_dir),
        "config": {
            "conformal_method": args.conformal_method,
            "conformal_alpha": args.conformal_alpha,
            "auto_confidence_threshold": args.auto_confidence_threshold,
            "boundary_threshold": args.boundary_threshold,
            "atypical_threshold": args.atypical_threshold,
            "signal_quality_threshold": args.signal_quality_threshold,
            "stability_risk_threshold": args.stability_risk_threshold,
        },
        "summary": summary_rows,
    }
    pd.DataFrame(summary_rows).to_csv(run_dir / "ambiguity_routing_summary.csv", index=False)
    (run_dir / "ambiguity_routing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
