from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .metrics import softmax


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(p * np.log(p + eps), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    train = np.load(run_dir / "embeddings_train.npz")
    test = np.load(run_dir / "embeddings_test.npz")
    train_emb, train_y = train["embeddings"], train["y"]
    emb, y, logits = test["embeddings"], test["y"], test["logits"]
    probs = softmax(logits)
    pred = probs.argmax(axis=1)

    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)
    d_vt, d_vf = dist[:, 1], dist[:, 2]

    prototype_vtvf_ambiguity = 1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12)
    softmax_vtvf_ambiguity = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(
        probs[:, 1] + probs[:, 2], 1e-12
    )

    nn = NearestNeighbors(n_neighbors=args.k)
    nn.fit(train_emb)
    _, idx = nn.kneighbors(emb)
    neigh_y = train_y[idx]
    label_probs = np.stack([(neigh_y == c).mean(axis=1) for c in range(3)], axis=1)
    knn_label_entropy = _entropy(label_probs) / np.log(3)
    knn_vtvf_mix = 1.0 - np.abs(label_probs[:, 1] - label_probs[:, 2]) / np.maximum(
        label_probs[:, 1] + label_probs[:, 2], 1e-12
    )
    knn_vtvf_mix[(label_probs[:, 1] + label_probs[:, 2]) == 0] = 0.0

    vai = (
        _normalise(prototype_vtvf_ambiguity)
        + _normalise(softmax_vtvf_ambiguity)
        + _normalise(knn_label_entropy)
        + _normalise(knn_vtvf_mix)
    ) / 4.0

    out = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": pred,
            "prob_sr": probs[:, 0],
            "prob_vt": probs[:, 1],
            "prob_vf": probs[:, 2],
            "d_sr": dist[:, 0],
            "d_vt": d_vt,
            "d_vf": d_vf,
            "prototype_vtvf_ambiguity": prototype_vtvf_ambiguity,
            "softmax_vtvf_ambiguity": softmax_vtvf_ambiguity,
            "knn_label_entropy": knn_label_entropy,
            "knn_vtvf_mix": knn_vtvf_mix,
            "ventricular_ambiguity_index": vai,
        }
    )
    out["is_vtvf"] = out["y_true"].isin([1, 2])
    out["is_vtvf_boundary_error"] = ((out["y_true"] == 1) & (out["y_pred"] == 2)) | (
        (out["y_true"] == 2) & (out["y_pred"] == 1)
    )
    out["is_any_error"] = out["y_true"] != out["y_pred"]
    out.to_csv(run_dir / "ambiguity_scores.csv", index=False)

    rows = []
    for score in [
        "prototype_vtvf_ambiguity",
        "softmax_vtvf_ambiguity",
        "knn_label_entropy",
        "knn_vtvf_mix",
        "ventricular_ambiguity_index",
    ]:
        vmask = out["is_vtvf"].to_numpy()
        target = out.loc[vmask, "is_vtvf_boundary_error"].astype(int).to_numpy()
        values = out.loc[vmask, score].to_numpy()
        auroc = np.nan if len(np.unique(target)) < 2 else roc_auc_score(target, values)
        aupr = np.nan if len(np.unique(target)) < 2 else average_precision_score(target, values)
        rows.append(
            {
                "score": score,
                "vtvf_boundary_auroc": auroc,
                "vtvf_boundary_aupr": aupr,
                "mean_correct_vtvf": float(out.loc[vmask & ~out["is_vtvf_boundary_error"], score].mean()),
                "mean_boundary_error": float(out.loc[out["is_vtvf_boundary_error"], score].mean()),
                "mean_sr": float(out.loc[out["y_true"] == 0, score].mean()),
                "mean_vt": float(out.loc[out["y_true"] == 1, score].mean()),
                "mean_vf": float(out.loc[out["y_true"] == 2, score].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "ambiguity_summary.csv", index=False)

    pca = PCA(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca[:, 0], pca[:, 1], c=vai, s=9, cmap="magma", alpha=0.75)
    plt.colorbar(sc, label="Ventricular ambiguity index")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(run_dir / "embedding_pca_ambiguity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    for label, mask in {
        "correct VT/VF": out["is_vtvf"] & ~out["is_vtvf_boundary_error"],
        "VT/VF boundary error": out["is_vtvf_boundary_error"],
        "SR": out["y_true"] == 0,
    }.items():
        plt.hist(out.loc[mask, "ventricular_ambiguity_index"], bins=30, alpha=0.55, density=True, label=label)
    plt.xlabel("Ventricular ambiguity index")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "ambiguity_distribution.png", dpi=180)
    plt.close()

    print(summary)


if __name__ == "__main__":
    main()
