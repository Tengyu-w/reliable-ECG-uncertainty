from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import expected_calibration_error, softmax
from .uncertainty import fit_temperature


CLASS_LABELS = {0: "SR", 1: "VT", 2: "VF"}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["embeddings"].astype(np.float32), data["logits"].astype(np.float32), data["y"].astype(np.int64)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return (-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(probs.shape[1])).astype(np.float32)


def _centroid_features(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray) -> dict[str, np.ndarray]:
    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)
    d_vt, d_vf = dist[:, 1], dist[:, 2]
    nearest = dist.argmin(axis=1)
    return {
        "d_sr": dist[:, 0],
        "d_vt": d_vt,
        "d_vf": d_vf,
        "min_proto_dist": dist.min(axis=1),
        "nearest_proto": nearest.astype(np.float32),
        "nearest_proto_is_pred": np.zeros(len(emb), dtype=np.float32),
        "proto_vtvf_ambiguity": 1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12),
        "abs_proto_vtvf_margin": np.abs(d_vf - d_vt),
    }


def _knn_features(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray, k: int) -> dict[str, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k).fit(train_emb)
    _, idx = nn.kneighbors(emb)
    neigh_y = train_y[idx]
    counts = np.stack([(neigh_y == c).mean(axis=1) for c in range(3)], axis=1)
    entropy = -np.sum(counts * np.log(np.maximum(counts, 1e-12)), axis=1) / np.log(3)
    vt = counts[:, 1]
    vf = counts[:, 2]
    ventricular = vt + vf
    vtvf_mix = np.zeros(len(emb), dtype=np.float32)
    valid = ventricular > 0
    vtvf_mix[valid] = 1.0 - np.abs(vt[valid] - vf[valid]) / ventricular[valid]
    return {
        "knn_label_entropy": entropy.astype(np.float32),
        "knn_vtvf_mixing": vtvf_mix.astype(np.float32),
        "knn_vt_fraction": vt.astype(np.float32),
        "knn_vf_fraction": vf.astype(np.float32),
    }


def _feature_frame(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    temperature: float,
    k: int,
) -> pd.DataFrame:
    probs = softmax(logits)
    probs_t = softmax(logits, temperature=temperature)
    pred = probs.argmax(axis=1)
    ventricular_prob = probs[:, 1] + probs[:, 2]
    softmax_vtvf_ambiguity = ventricular_prob * (
        1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_prob, 1e-12)
    )
    centroid = _centroid_features(train_emb, train_y, emb)
    knn = _knn_features(train_emb, train_y, emb, k)
    centroid["nearest_proto_is_pred"] = (centroid["nearest_proto"].astype(int) == pred).astype(np.float32)

    df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": pred,
            "prob_sr": probs[:, 0],
            "prob_vt": probs[:, 1],
            "prob_vf": probs[:, 2],
            "temperature_prob_sr": probs_t[:, 0],
            "temperature_prob_vt": probs_t[:, 1],
            "temperature_prob_vf": probs_t[:, 2],
            "max_prob": probs.max(axis=1),
            "temperature_max_prob": probs_t.max(axis=1),
            "msp_uncertainty": 1.0 - probs.max(axis=1),
            "entropy": _entropy(probs),
            "temperature_entropy": _entropy(probs_t),
            "ventricular_prob": ventricular_prob,
            "softmax_vtvf_ambiguity": softmax_vtvf_ambiguity,
            "abs_prob_vtvf_margin": np.abs(probs[:, 1] - probs[:, 2]),
            "abs_logit_vtvf_margin": np.abs(logits[:, 1] - logits[:, 2]),
            "pred_is_vtvf": np.isin(pred, [1, 2]).astype(np.float32),
            "top2_are_vtvf": (
                np.sort(np.argsort(probs, axis=1)[:, -2:], axis=1) == np.asarray([1, 2])
            ).all(axis=1).astype(np.float32),
            **centroid,
            **knn,
        }
    )
    df["is_error"] = df["y_true"] != df["y_pred"]
    df["is_vtvf_cross_error"] = ((df["y_true"] == 1) & (df["y_pred"] == 2)) | (
        (df["y_true"] == 2) & (df["y_pred"] == 1)
    )
    df["is_vtvf_truth"] = df["y_true"].isin([1, 2])
    return df


def _fit_binary_risk_model(x: pd.DataFrame, y: np.ndarray) -> object:
    if len(np.unique(y)) < 2:
        class ConstantModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                p = np.full(len(features), self.value, dtype=np.float32)
                return np.stack([1.0 - p, p], axis=1)

        return ConstantModel(float(y.mean()))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
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


def _review_curve(df: pd.DataFrame, score: np.ndarray, budgets: list[float]) -> pd.DataFrame:
    any_error = df["is_error"].to_numpy(bool)
    vtvf_error = df["is_vtvf_cross_error"].to_numpy(bool)
    order = np.argsort(-score)
    rows = []
    for budget in budgets:
        n_review = max(1, int(round(len(df) * budget)))
        review_idx = order[:n_review]
        auto_idx = order[n_review:]
        rows.append(
            {
                "policy": "learned_risk_review",
                "budget": budget,
                "reviewed_or_set": n_review,
                "action_rate": n_review / len(df),
                "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
                "vtvf_cross_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
                "auto_error_rate_after_action": float(any_error[auto_idx].mean()) if len(auto_idx) else np.nan,
                "auto_vtvf_cross_error_rate_after_action": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else np.nan,
                "review_error_enrichment": float(any_error[review_idx].mean() / max(any_error.mean(), 1e-8)),
            }
        )
    return pd.DataFrame(rows)


def _threshold_from_validation(score: np.ndarray, budget: float) -> float:
    n = max(1, int(round(len(score) * budget)))
    order = np.sort(score)[::-1]
    return float(order[min(n - 1, len(order) - 1)])


def _set_policy_summary(val_df: pd.DataFrame, test_df: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        threshold = _threshold_from_validation(val_df["vtvf_boundary_risk"].to_numpy(float), budget)
        for split, df in [("val", val_df), ("test", test_df)]:
            flag = (
                (df["vtvf_boundary_risk"].to_numpy(float) >= threshold)
                & df["pred_is_vtvf"].to_numpy(bool)
            )
            y = df["y_true"].to_numpy(int)
            pred = df["y_pred"].to_numpy(int)
            singleton = ~flag
            set_contains_true = flag & np.isin(y, [1, 2])
            unresolved_error = singleton & (y != pred)
            unresolved_vtvf_cross_error = singleton & (
                ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
            )
            baseline_error = y != pred
            baseline_vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
            rows.append(
                {
                    "policy": "learned_vtvf_prediction_set",
                    "split": split,
                    "budget": budget,
                    "threshold_from_val": threshold,
                    "reviewed_or_set": int(flag.sum()),
                    "action_rate": float(flag.mean()),
                    "set_contains_true_rate": float(set_contains_true[flag].mean()) if flag.any() else np.nan,
                    "avg_set_size": float(1.0 + flag.mean()),
                    "baseline_error_rate": float(baseline_error.mean()),
                    "effective_error_rate": float(unresolved_error.mean()),
                    "baseline_vtvf_cross_error_rate": float(baseline_vtvf.mean()),
                    "effective_vtvf_cross_error_rate": float(unresolved_vtvf_cross_error.mean()),
                    "all_error_captured": float((baseline_error & flag).sum() / max(baseline_error.sum(), 1)),
                    "vtvf_cross_error_captured": float((baseline_vtvf & flag).sum() / max(baseline_vtvf.sum(), 1)),
                }
            )
    return pd.DataFrame(rows)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"y_true", "y_pred", "is_error", "is_vtvf_cross_error", "is_vtvf_truth"}
    return [c for c in df.columns if c not in excluded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    train_emb, train_logits, train_y = _load_split(run_dir, "train")
    val_emb, val_logits, val_y = _load_split(run_dir, "val")
    test_emb, test_logits, test_y = _load_split(run_dir, "test")
    temperature = fit_temperature(val_logits, val_y)

    val_df = _feature_frame(train_emb, train_y, val_emb, val_logits, val_y, temperature, args.k)
    test_df = _feature_frame(train_emb, train_y, test_emb, test_logits, test_y, temperature, args.k)
    features = _feature_columns(val_df)

    any_error_model = _fit_binary_risk_model(val_df[features], val_df["is_error"].to_numpy(int))
    vtvf_error_model = _fit_binary_risk_model(val_df[features], val_df["is_vtvf_cross_error"].to_numpy(int))

    val_df["any_error_risk"] = any_error_model.predict_proba(val_df[features])[:, 1]
    test_df["any_error_risk"] = any_error_model.predict_proba(test_df[features])[:, 1]
    val_df["vtvf_boundary_risk"] = vtvf_error_model.predict_proba(val_df[features])[:, 1]
    test_df["vtvf_boundary_risk"] = vtvf_error_model.predict_proba(test_df[features])[:, 1]

    test_df.to_csv(run_dir / "vtvf_decision_calibration_scores_test.csv", index=False)
    val_df.to_csv(run_dir / "vtvf_decision_calibration_scores_val.csv", index=False)

    review_curves = []
    for name, score in {
        "learned_vtvf_boundary_risk": test_df["vtvf_boundary_risk"].to_numpy(float),
        "learned_any_error_risk": test_df["any_error_risk"].to_numpy(float),
        "entropy": test_df["entropy"].to_numpy(float),
        "knn_vtvf_mixing": test_df["knn_vtvf_mixing"].to_numpy(float),
        "prototype_vtvf_ambiguity": test_df["proto_vtvf_ambiguity"].to_numpy(float),
        "softmax_vtvf_ambiguity": test_df["softmax_vtvf_ambiguity"].to_numpy(float),
    }.items():
        curve = _review_curve(test_df, score, args.budgets)
        curve["score"] = name
        review_curves.append(curve)
    review_df = pd.concat(review_curves, ignore_index=True)
    review_df.to_csv(run_dir / "vtvf_decision_calibration_review_curves.csv", index=False)

    set_df = _set_policy_summary(val_df, test_df, args.budgets)
    set_df.to_csv(run_dir / "vtvf_prediction_set_policy_summary.csv", index=False)

    pred = test_df["y_pred"].to_numpy(int)
    probs = test_df[["prob_sr", "prob_vt", "prob_vf"]].to_numpy(float)
    probs_t = test_df[["temperature_prob_sr", "temperature_prob_vt", "temperature_prob_vf"]].to_numpy(float)
    metric_rows = [
        {
            "metric": "single_label_accuracy",
            "value": float((pred == test_y).mean()),
        },
        {
            "metric": "single_label_error_rate",
            "value": float((pred != test_y).mean()),
        },
        {
            "metric": "single_label_vtvf_cross_errors",
            "value": int(test_df["is_vtvf_cross_error"].sum()),
        },
        {
            "metric": "ece_before_temperature",
            "value": expected_calibration_error(test_y, probs),
        },
        {
            "metric": "ece_after_temperature",
            "value": expected_calibration_error(test_y, probs_t),
        },
        {
            "metric": "temperature",
            "value": float(temperature),
        },
        {
            "metric": "vtvf_boundary_risk_auroc",
            "value": _safe_auc(test_df["is_vtvf_cross_error"].to_numpy(int), test_df["vtvf_boundary_risk"].to_numpy(float)),
        },
        {
            "metric": "vtvf_boundary_risk_aupr",
            "value": _safe_aupr(test_df["is_vtvf_cross_error"].to_numpy(int), test_df["vtvf_boundary_risk"].to_numpy(float)),
        },
        {
            "metric": "any_error_risk_auroc",
            "value": _safe_auc(test_df["is_error"].to_numpy(int), test_df["any_error_risk"].to_numpy(float)),
        },
        {
            "metric": "any_error_risk_aupr",
            "value": _safe_aupr(test_df["is_error"].to_numpy(int), test_df["any_error_risk"].to_numpy(float)),
        },
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "vtvf_decision_calibration_metrics.csv", index=False)

    plt.figure(figsize=(7, 4.8))
    for score_name, sub in review_df.groupby("score"):
        plt.plot(sub["budget"], sub["vtvf_cross_error_captured"], marker="o", label=score_name)
    plt.xlabel("Action / review budget")
    plt.ylabel("VT/VF cross-error captured")
    plt.ylim(0, 1.02)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(run_dir / "vtvf_decision_calibration_review_curves.png", dpi=180)
    plt.close()

    test_set = set_df[set_df["split"].eq("test")]
    plt.figure(figsize=(7, 4.4))
    plt.plot(test_set["budget"], test_set["baseline_vtvf_cross_error_rate"], marker="o", label="single-label baseline")
    plt.plot(test_set["budget"], test_set["effective_vtvf_cross_error_rate"], marker="o", label="{VT,VF} set policy")
    plt.xlabel("Validation-selected action budget")
    plt.ylabel("Unresolved VT/VF cross-error rate")
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "vtvf_prediction_set_effective_error.png", dpi=180)
    plt.close()

    report = {
        "run_dir": str(run_dir),
        "temperature": float(temperature),
        "features": features,
        "metrics": metrics.to_dict(orient="records"),
        "best_review_rows_10_20_30": review_df[
            review_df["budget"].isin([0.10, 0.20, 0.30])
            & review_df["score"].eq("learned_vtvf_boundary_risk")
        ].to_dict(orient="records"),
        "prediction_set_test_10_20_30": test_set[test_set["budget"].isin([0.10, 0.20, 0.30])].to_dict(orient="records"),
    }
    (run_dir / "vtvf_decision_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metrics)
    print(review_df[review_df["budget"].isin([0.10, 0.20, 0.30])].sort_values(["budget", "vtvf_cross_error_captured"], ascending=[True, False]).head(18))
    print(test_set[test_set["budget"].isin([0.10, 0.20, 0.30])])


if __name__ == "__main__":
    main()
