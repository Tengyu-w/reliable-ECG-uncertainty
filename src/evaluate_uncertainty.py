from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .metrics import expected_calibration_error, softmax
from .uncertainty import (
    energy,
    entropy,
    error_detection_metrics,
    fit_temperature,
    knn_distance,
    mahalanobis_distance,
    msp,
    prototype_distance,
)


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["logits"], data["embeddings"], data["y"]


def _plot_reliability(y: np.ndarray, probs: np.ndarray, path: Path, n_bins: int = 10) -> None:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.bar(centers, accs, width=0.08, alpha=0.75, label="accuracy")
    plt.scatter(confs, accs, color="black", s=20, label="bins")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_coverage_risk(y: np.ndarray, probs: np.ndarray, score: np.ndarray, path: Path) -> None:
    order = np.argsort(score)
    y_pred = probs.argmax(axis=1)
    coverages, risks = [], []
    for keep in np.linspace(0.1, 1.0, 30):
        n = max(1, int(len(order) * keep))
        idx = order[:n]
        coverages.append(keep)
        risks.append(float((y_pred[idx] != y[idx]).mean()))
    plt.figure(figsize=(5, 4))
    plt.plot(coverages, risks)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_embedding(emb: np.ndarray, y: np.ndarray, score: np.ndarray, path: Path) -> None:
    xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=y, s=10, cmap="tab10", alpha=0.75)
    high = score >= np.quantile(score, 0.9)
    plt.scatter(xy[high, 0], xy[high, 1], facecolors="none", edgecolors="black", s=35, label="top 10% uncertainty")
    plt.legend(*scatter.legend_elements(), title="class", loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)

    train_logits, train_emb, train_y = _load(run_dir, "train")
    val_logits, _, val_y = _load(run_dir, "val")
    test_logits, test_emb, test_y = _load(run_dir, "test")

    temp = fit_temperature(val_logits, val_y)
    probs = softmax(test_logits)
    probs_t = softmax(test_logits, temperature=temp)
    scores = {
        "msp": msp(probs),
        "entropy": entropy(probs),
        "temperature_msp": msp(probs_t),
        "energy": -energy(test_logits),
        "prototype": prototype_distance(train_emb, train_y, test_emb),
        "mahalanobis": mahalanobis_distance(train_emb, train_y, test_emb),
        "knn": knn_distance(train_emb, test_emb, k=args.k),
    }

    metrics = error_detection_metrics(test_y, probs, scores)
    metrics.append({"score": "before_temperature", "error_auroc": np.nan, "error_aupr": expected_calibration_error(test_y, probs)})
    metrics.append({"score": "after_temperature", "error_auroc": temp, "error_aupr": expected_calibration_error(test_y, probs_t)})
    pd.DataFrame(metrics).to_csv(run_dir / "uncertainty_metrics.csv", index=False)

    score_df = pd.DataFrame(scores)
    score_df["y_true"] = test_y
    score_df["y_pred"] = probs.argmax(axis=1)
    score_df.to_csv(run_dir / "uncertainty_scores.csv", index=False)

    _plot_reliability(test_y, probs, run_dir / "reliability_before.png")
    _plot_reliability(test_y, probs_t, run_dir / "reliability_after_temperature.png")
    _plot_coverage_risk(test_y, probs, scores["knn"], run_dir / "coverage_risk.png")
    _plot_embedding(test_emb, test_y, scores["knn"], run_dir / "embedding_pca.png")
    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
