from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _load_base(run_dir: Path) -> pd.DataFrame:
    policy_path = run_dir / "ambiguity_routing_policy.csv"
    if policy_path.exists():
        return pd.read_csv(policy_path)

    stability_path = run_dir / "stability_scores.csv"
    if stability_path.exists():
        return pd.read_csv(stability_path)

    uncertainty_path = run_dir / "uncertainty_scores.csv"
    ambiguity_path = run_dir / "ambiguity_scores.csv"
    if uncertainty_path.exists() and ambiguity_path.exists():
        uncertainty = pd.read_csv(uncertainty_path)
        ambiguity = pd.read_csv(ambiguity_path)
        out = uncertainty.copy()
        out["index"] = np.arange(len(out))
        for col in [
            "softmax_vtvf_ambiguity",
            "prototype_vtvf_ambiguity",
            "knn_vtvf_mix",
            "ventricular_ambiguity_index",
            "is_any_error",
            "is_vtvf_boundary_error",
        ]:
            if col in ambiguity and col not in out:
                out[col] = ambiguity[col].to_numpy()
        prob_cols = [c for c in ambiguity.columns if c.startswith("prob_")]
        if prob_cols:
            probs = ambiguity[prob_cols].to_numpy(float)
            out["max_softmax_prob"] = probs.max(axis=1)
        else:
            out["max_softmax_prob"] = 1.0 - out["msp"].to_numpy(float) if "msp" in out else 0.0
        if "is_error" not in out:
            if "is_any_error" in out:
                out["is_error"] = out["is_any_error"].astype(bool)
            else:
                out["is_error"] = out["y_true"].to_numpy(int) != out["y_pred"].to_numpy(int)
        if "is_vtvf_cross_error" not in out:
            if "is_vtvf_boundary_error" in out:
                out["is_vtvf_cross_error"] = out["is_vtvf_boundary_error"].astype(bool)
            else:
                out["is_vtvf_cross_error"] = (
                    ((out["y_true"] == 1) & (out["y_pred"] == 2))
                    | ((out["y_true"] == 2) & (out["y_pred"] == 1))
                ).to_numpy(bool)

        neigh_path = run_dir / "embedding_neighborhood_k15.csv"
        if neigh_path.exists():
            neigh = pd.read_csv(neigh_path)
            if len(neigh) == len(out):
                if "local_purity" in neigh:
                    out["deployable_local_instability"] = 1.0 - neigh["local_purity"].to_numpy(float)
                if "vtvf_mixing" in neigh:
                    out["deployable_vtvf_mixing"] = neigh["vtvf_mixing"].to_numpy(float)
        return out

    pred_path = run_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Expected ambiguity_routing_policy.csv, stability_scores.csv, or test_predictions.csv in {run_dir}"
        )
    pred = pd.read_csv(pred_path)
    prob_cols = [c for c in pred.columns if c.startswith("prob_")]
    probs = pred[prob_cols].to_numpy(float)
    max_prob = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
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
            "max_softmax_prob": max_prob,
            "entropy": entropy,
        }
    )


def _column(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df:
        return df[name].to_numpy(float)
    return np.full(len(df), default, dtype=np.float32)


def _bool_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df:
        return np.zeros(len(df), dtype=bool)
    return df[name].astype(bool).to_numpy()


def build_supervisor_table(
    df: pd.DataFrame,
    confidence_threshold: float,
    suspect_threshold: float,
    recover_threshold: float,
    human_threshold: float,
) -> pd.DataFrame:
    entropy = _normalise(_column(df, "entropy"))
    boundary = _normalise(_column(df, "softmax_vtvf_ambiguity"))
    vtvf_mixing = _normalise(_column(df, "deployable_vtvf_mixing"))
    local_instability = _normalise(_column(df, "deployable_local_instability"))
    atypicality = _normalise(_column(df, "knn"))
    stability_risk = _normalise(_column(df, "stability_risk"))
    flip = _normalise(_column(df, "pred_flip_rate"))
    drift = _normalise(_column(df, "embedding_drift"))

    if "boundary_signal" in df:
        boundary_signal = _bool_column(df, "boundary_signal")
    else:
        boundary_signal = (boundary >= suspect_threshold) | (vtvf_mixing >= suspect_threshold)
    if "signal_quality_signal" in df:
        quality_signal = _bool_column(df, "signal_quality_signal")
    else:
        quality_signal = (
            (local_instability >= recover_threshold)
            | (stability_risk >= recover_threshold)
            | (flip >= recover_threshold)
            | (drift >= recover_threshold)
        )
    if "forced_expert_signal" in df:
        forced_signal = _bool_column(df, "forced_expert_signal")
    else:
        high_conf = _column(df, "max_softmax_prob", 0.0) >= confidence_threshold
        forced_signal = high_conf & (atypicality >= human_threshold)

    boundary_risk = np.clip(0.45 * boundary + 0.35 * vtvf_mixing + 0.20 * entropy, 0.0, 1.0)
    quality_risk = np.clip(0.35 * local_instability + 0.30 * stability_risk + 0.20 * flip + 0.15 * drift, 0.0, 1.0)
    hidden_failure_risk = np.clip(0.50 * atypicality + 0.30 * (1.0 - entropy) + 0.20 * boundary, 0.0, 1.0)
    supervisor_risk = np.maximum.reduce([boundary_risk, quality_risk, hidden_failure_risk])

    state = np.full(len(df), "NORMAL", dtype=object)
    action = np.full(len(df), "accept_model_prediction", dtype=object)
    reason = np.full(len(df), "low_supervisor_risk", dtype=object)

    suspect = supervisor_risk >= suspect_threshold
    state[suspect] = "SUSPECT"
    action[suspect] = "continue_with_verification"
    reason[suspect] = "elevated_uncertainty_or_boundary_risk"

    recover = quality_signal | (quality_risk >= recover_threshold)
    state[recover] = "RECOVER"
    action[recover] = "reacquire_signal_or_replan"
    reason[recover] = "instability_or_signal_quality_risk"

    human = forced_signal | (boundary_signal & (boundary_risk >= human_threshold)) | (supervisor_risk >= human_threshold)
    state[human] = "HUMAN_REVIEW"
    action[human] = "request_expert_review_or_takeover"
    reason[human] = "boundary_ambiguity_or_hidden_failure_risk"

    out = df.copy()
    out["boundary_risk"] = boundary_risk
    out["quality_risk"] = quality_risk
    out["hidden_failure_risk"] = hidden_failure_risk
    out["supervisor_risk"] = supervisor_risk
    out["supervisor_state"] = state
    out["supervisor_action"] = action
    out["supervisor_reason"] = reason
    out["embodied_ai_analogue"] = out["supervisor_state"].map(
        {
            "NORMAL": "autonomous_execution",
            "SUSPECT": "slow_down_and_verify_action_outcome",
            "RECOVER": "replan_relocalize_or_reacquire_observation",
            "HUMAN_REVIEW": "request_surgeon_or_operator_takeover",
        }
    )
    return out


def summarise_supervisor(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for state, sub in df.groupby("supervisor_state"):
        rows.append(
            {
                "state": state,
                "n": int(len(sub)),
                "rate": float(len(sub) / n),
                "mean_supervisor_risk": float(sub["supervisor_risk"].mean()),
                "error_rate": float(sub["is_error"].mean()) if "is_error" in sub else np.nan,
                "vtvf_cross_error_rate": float(sub["is_vtvf_cross_error"].mean())
                if "is_vtvf_cross_error" in sub
                else np.nan,
                "vtvf_cross_errors": int(sub["is_vtvf_cross_error"].sum())
                if "is_vtvf_cross_error" in sub
                else 0,
            }
        )
    order = {"NORMAL": 0, "SUSPECT": 1, "RECOVER": 2, "HUMAN_REVIEW": 3}
    return pd.DataFrame(rows).sort_values("state", key=lambda s: s.map(order).fillna(99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--suspect-threshold", type=float, default=0.45)
    parser.add_argument("--recover-threshold", type=float, default=0.65)
    parser.add_argument("--human-threshold", type=float, default=0.75)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    base = _load_base(run_dir)
    table = build_supervisor_table(
        base,
        confidence_threshold=args.confidence_threshold,
        suspect_threshold=args.suspect_threshold,
        recover_threshold=args.recover_threshold,
        human_threshold=args.human_threshold,
    )
    summary = summarise_supervisor(table)

    table.to_csv(run_dir / "runtime_supervisor_policy.csv", index=False)
    summary.to_csv(run_dir / "runtime_supervisor_summary.csv", index=False)
    metadata = {
        "run_dir": str(run_dir),
        "state_machine": {
            "NORMAL": "Accept the model prediction under low supervisor risk.",
            "SUSPECT": "Continue only with additional verification.",
            "RECOVER": "Reacquire signal, replan, or recover before accepting autonomy.",
            "HUMAN_REVIEW": "Request expert review or human takeover.",
        },
        "embodied_ai_mapping": {
            "ECG boundary ambiguity": "ambiguous surgical state or action choice",
            "ECG signal-quality risk": "occlusion, camera shift, tool drift, or poor observation",
            "high-confidence atypicality": "confident but unsupported VLA action",
            "expert review": "surgeon/operator takeover or clarification",
        },
        "thresholds": {
            "confidence_threshold": args.confidence_threshold,
            "suspect_threshold": args.suspect_threshold,
            "recover_threshold": args.recover_threshold,
            "human_threshold": args.human_threshold,
        },
    }
    (run_dir / "runtime_supervisor_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
