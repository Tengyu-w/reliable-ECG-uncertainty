from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .data import CLASS_NAMES


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _norm_dist(emb: np.ndarray, y: np.ndarray, i: int, j: int) -> float:
    centroids = np.stack([emb[y == c].mean(axis=0) for c in range(3)])
    within = [np.linalg.norm(emb[y == c] - centroids[c], axis=1).mean() for c in range(3)]
    return float(np.linalg.norm(centroids[i] - centroids[j]) / ((within[i] + within[j]) / 2.0))


def _neighbor_metrics(emb: np.ndarray, y: np.ndarray, k: int) -> tuple[pd.DataFrame, dict[str, float]]:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, indices = nn.kneighbors(emb)
    neigh = indices[:, 1:]
    neigh_y = y[neigh]
    local_purity = (neigh_y == y[:, None]).mean(axis=1)

    ventricular = np.isin(y, [1, 2])
    opposite = np.where(y == 1, 2, 1)
    ventricular_neighbors = np.isin(neigh_y, [1, 2])
    opposite_vtvf = neigh_y == opposite[:, None]
    vtvf_mixing = np.zeros(len(y), dtype=np.float32)
    denom = ventricular_neighbors.sum(axis=1)
    valid = ventricular & (denom > 0)
    vtvf_mixing[valid] = opposite_vtvf[valid].sum(axis=1) / denom[valid]

    rows = pd.DataFrame(
        {
            "y_true": y,
            "local_purity": local_purity,
            "vtvf_mixing": vtvf_mixing,
            "ventricular_neighbor_fraction": ventricular_neighbors.mean(axis=1),
        }
    )
    summary = {
        f"purity_k{k}_mean": float(local_purity.mean()),
        f"purity_k{k}_sr": float(local_purity[y == 0].mean()),
        f"purity_k{k}_vt": float(local_purity[y == 1].mean()),
        f"purity_k{k}_vf": float(local_purity[y == 2].mean()),
        f"vtvf_mixing_k{k}_vt": float(vtvf_mixing[y == 1].mean()),
        f"vtvf_mixing_k{k}_vf": float(vtvf_mixing[y == 2].mean()),
        f"vtvf_mixing_k{k}_ventricular": float(vtvf_mixing[ventricular].mean()),
    }
    return rows, summary


def _plot_3d(points: np.ndarray, y: np.ndarray, out: Path, title: str, errors: np.ndarray | None = None) -> None:
    colors = np.asarray(["#4C78A8", "#F58518", "#54A24B"])[y]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8, alpha=0.55, depthshade=False)
    if errors is not None and errors.any():
        ax.scatter(points[errors, 0], points[errors, 1], points[errors, 2], c="#D62728", s=20, alpha=0.9, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=7)
        for label, color in zip(CLASS_NAMES, ["#4C78A8", "#F58518", "#54A24B"])
    ]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--tsne", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    data = np.load(run_dir / "embeddings_test.npz")
    emb, y = data["embeddings"], data["y"]

    pred_path = run_dir / "test_predictions.csv"
    errors = None
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        errors = (pred["y_true"].to_numpy() != pred["y_pred"].to_numpy())

    pca3_model = PCA(n_components=3, random_state=42)
    pca3 = pca3_model.fit_transform(emb)
    pd.DataFrame(pca3, columns=["pca1", "pca2", "pca3"]).assign(y_true=y).to_csv(
        run_dir / "embedding_pca3_coordinates.csv", index=False
    )
    _plot_3d(pca3, y, run_dir / "embedding_pca3.png", "3D PCA embedding", errors=errors)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda2 = lda.fit_transform(emb, y)
    pd.DataFrame(lda2, columns=["lda1", "lda2"]).assign(y_true=y).to_csv(
        run_dir / "embedding_lda2_coordinates.csv", index=False
    )
    plt.figure(figsize=(6, 5))
    for c, label in enumerate(CLASS_NAMES):
        mask = y == c
        plt.scatter(lda2[mask, 0], lda2[mask, 1], s=9, alpha=0.55, label=label)
    if errors is not None and errors.any():
        plt.scatter(lda2[errors, 0], lda2[errors, 1], s=20, c="red", alpha=0.8, label="error")
    plt.xlabel("LDA 1")
    plt.ylabel("LDA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "embedding_lda2.png", dpi=180)
    plt.close()

    neigh_rows, neigh_summary = _neighbor_metrics(emb, y, args.k)
    if errors is not None:
        neigh_rows["error"] = errors
    neigh_rows.to_csv(run_dir / f"embedding_neighborhood_k{args.k}.csv", index=False)

    summary = {
        "silhouette_full": float(silhouette_score(emb, y)),
        "silhouette_pca3": float(silhouette_score(pca3, y)),
        "silhouette_lda2": float(silhouette_score(lda2, y)),
        "davies_bouldin_full": float(davies_bouldin_score(emb, y)),
        "davies_bouldin_pca3": float(davies_bouldin_score(pca3, y)),
        "trustworthiness_pca3": float(trustworthiness(emb, pca3, n_neighbors=args.k)),
        "sr_vt_norm_dist": _norm_dist(emb, y, 0, 1),
        "sr_vf_norm_dist": _norm_dist(emb, y, 0, 2),
        "vt_vf_norm_dist": _norm_dist(emb, y, 1, 2),
        **neigh_summary,
    }

    if errors is not None:
        summary["error_mean_local_purity"] = float(neigh_rows.loc[errors, "local_purity"].mean())
        summary["correct_mean_local_purity"] = float(neigh_rows.loc[~errors, "local_purity"].mean())
        vtvf_error = (
            ((pred["y_true"].to_numpy() == 1) & (pred["y_pred"].to_numpy() == 2))
            | ((pred["y_true"].to_numpy() == 2) & (pred["y_pred"].to_numpy() == 1))
        )
        if vtvf_error.any():
            summary["vtvf_error_mean_mixing"] = float(neigh_rows.loc[vtvf_error, "vtvf_mixing"].mean())
        summary["correct_ventricular_mean_mixing"] = float(
            neigh_rows.loc[(~errors) & np.isin(y, [1, 2]), "vtvf_mixing"].mean()
        )

    if args.tsne:
        tsne2 = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=42).fit_transform(emb)
        pd.DataFrame(tsne2, columns=["tsne1", "tsne2"]).assign(y_true=y).to_csv(
            run_dir / "embedding_tsne2_coordinates.csv", index=False
        )
        summary["trustworthiness_tsne2"] = float(trustworthiness(emb, tsne2, n_neighbors=args.k))
        plt.figure(figsize=(6, 5))
        for c, label in enumerate(CLASS_NAMES):
            mask = y == c
            plt.scatter(tsne2[mask, 0], tsne2[mask, 1], s=9, alpha=0.55, label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "embedding_tsne2.png", dpi=180)
        plt.close()

    out = pd.DataFrame([summary])
    out.to_csv(run_dir / "embedding_geometry_summary.csv", index=False)
    print(out.T)


if __name__ == "__main__":
    main()
