from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def _curve(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(score)
    rows = []
    for coverage in np.linspace(0.1, 1.0, 19):
        n_keep = max(1, int(round(len(order) * coverage)))
        keep = order[:n_keep]
        rows.append(
            {
                "coverage": float(coverage),
                "risk": float(np.mean(y_true[keep] != y_pred[keep])),
                "accuracy": float(np.mean(y_true[keep] == y_pred[keep])),
                "macro_f1": _macro_f1(y_true[keep], y_pred[keep]),
                "accepted": int(n_keep),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    scores = pd.read_csv(run_dir / "uncertainty_scores.csv")
    y_true = scores["y_true"].to_numpy()
    y_pred = scores["y_pred"].to_numpy()

    score_cols = [c for c in scores.columns if c not in {"y_true", "y_pred"}]
    if "entropy" in scores and "knn" in scores:
        scores["hybrid_entropy_knn"] = args.alpha * _normalise(scores["entropy"].to_numpy()) + (
            1 - args.alpha
        ) * _normalise(scores["knn"].to_numpy())
        score_cols.append("hybrid_entropy_knn")
    if "entropy" in scores and "mahalanobis" in scores:
        scores["hybrid_entropy_mahalanobis"] = args.alpha * _normalise(scores["entropy"].to_numpy()) + (
            1 - args.alpha
        ) * _normalise(scores["mahalanobis"].to_numpy())
        score_cols.append("hybrid_entropy_mahalanobis")

    rows = []
    curves = []
    for col in score_cols:
        curve = _curve(y_true, y_pred, scores[col].to_numpy())
        curve.insert(0, "score", col)
        curves.append(curve)
        for coverage in (0.7, 0.8, 0.9, 1.0):
            row = curve.iloc[np.argmin(np.abs(curve["coverage"] - coverage))].to_dict()
            row["score"] = col
            rows.append(row)

    summary = pd.DataFrame(rows)
    all_curves = pd.concat(curves, ignore_index=True)
    summary.to_csv(run_dir / "selective_summary.csv", index=False)
    all_curves.to_csv(run_dir / "selective_curves.csv", index=False)

    plt.figure(figsize=(6, 4))
    for col in ["msp", "entropy", "knn", "mahalanobis", "hybrid_entropy_knn"]:
        if col not in all_curves["score"].unique():
            continue
        sub = all_curves[all_curves["score"] == col]
        plt.plot(sub["coverage"], sub["risk"], label=col)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "selective_risk_coverage.png", dpi=180)
    plt.close()

    print(summary)


if __name__ == "__main__":
    main()
