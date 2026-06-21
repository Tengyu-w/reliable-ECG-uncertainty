from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _embedding_summary(run_dir: Path) -> dict[str, float]:
    data = np.load(run_dir / "embeddings_test.npz")
    emb, y = data["embeddings"], data["y"]
    centroids = np.stack([emb[y == c].mean(axis=0) for c in range(3)])
    within = [np.linalg.norm(emb[y == c] - centroids[c], axis=1).mean() for c in range(3)]

    def norm_dist(i: int, j: int) -> float:
        return float(np.linalg.norm(centroids[i] - centroids[j]) / ((within[i] + within[j]) / 2))

    return {
        "silhouette_full": float(silhouette_score(emb, y)),
        "silhouette_pca2": float(silhouette_score(PCA(n_components=2, random_state=42).fit_transform(emb), y)),
        "sr_vt_norm_dist": norm_dist(0, 1),
        "sr_vf_norm_dist": norm_dist(0, 2),
        "vt_vf_norm_dist": norm_dist(1, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Format: label=path")
    parser.add_argument("--out-dir", type=Path, default=Path("results/versioned_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    classification_rows = []
    uncertainty_rows = []
    reliability_rows = []
    regularity_rows = []
    severity_rows = []

    for spec in args.runs:
        label, path_text = spec.split("=", 1)
        run_dir = Path(path_text)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        row = {"version": label, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
        row.update(_embedding_summary(run_dir))
        classification_rows.append(row)

        for fname, store, prefix in [
            ("uncertainty_metrics.csv", uncertainty_rows, {}),
            ("reliability_map_summary.csv", reliability_rows, {}),
            ("regularity_summary.csv", regularity_rows, {}),
            ("severity_monotonicity.csv", severity_rows, {}),
        ]:
            df = _safe_read_csv(run_dir / fname)
            if df is not None:
                df.insert(0, "version", label)
                store.append(df)

    pd.DataFrame(classification_rows).to_csv(args.out_dir / "classification_embedding_summary.csv", index=False)
    if uncertainty_rows:
        pd.concat(uncertainty_rows, ignore_index=True).to_csv(args.out_dir / "uncertainty_summary.csv", index=False)
    if reliability_rows:
        pd.concat(reliability_rows, ignore_index=True).to_csv(args.out_dir / "reliability_map_summary.csv", index=False)
    if regularity_rows:
        pd.concat(regularity_rows, ignore_index=True).to_csv(args.out_dir / "regularity_summary.csv", index=False)
    if severity_rows:
        pd.concat(severity_rows, ignore_index=True).to_csv(args.out_dir / "severity_monotonicity_summary.csv", index=False)

    print("Wrote summaries to", args.out_dir)


if __name__ == "__main__":
    main()
