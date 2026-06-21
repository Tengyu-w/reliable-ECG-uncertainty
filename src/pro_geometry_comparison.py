from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


CLASS_NAMES = ["SR", "VT", "VF"]
COLORS = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}


def _load_run_level(path: Path, variants: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[df["variant"].isin(variants)].copy()
    if out.empty:
        raise ValueError(f"No requested variants found in {path}: {variants}")
    return out.sort_values(["seed", "variant"])


def _centroids(emb: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([emb[y == c].mean(axis=0) for c in range(3)])


def _within_scatter(emb: np.ndarray, y: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(emb[y == c] - centroids[c], axis=1).mean() for c in range(3)])


def _norm_dist(centroids: np.ndarray, within: np.ndarray, i: int, j: int) -> float:
    denom = (within[i] + within[j]) / 2.0
    return float(np.linalg.norm(centroids[i] - centroids[j]) / max(denom, 1e-12))


def _vtvf_mixing(emb: np.ndarray, y: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, indices = nn.kneighbors(emb)
    neigh_y = y[indices[:, 1:]]
    ventricular = np.isin(y, [1, 2])
    opposite = np.where(y == 1, 2, 1)
    ventricular_neighbors = np.isin(neigh_y, [1, 2])
    opposite_vtvf = neigh_y == opposite[:, None]
    denom = ventricular_neighbors.sum(axis=1)
    valid = ventricular & (denom > 0)
    mix = np.zeros(len(y), dtype=np.float32)
    mix[valid] = opposite_vtvf[valid].sum(axis=1) / denom[valid]
    return float(mix[valid].mean()) if valid.any() else float("nan")


def _auto_vtvf_errors(run_dir: Path) -> float:
    path = run_dir / "ambiguity_routing_summary.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    row = df[df["decision"] == "automatic_single_label"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["vtvf_cross_errors"])


def _run_geometry(row: pd.Series, k: int) -> dict[str, float | str | int]:
    run_dir = Path(str(row["run_dir"]))
    data = np.load(run_dir / "embeddings_test.npz")
    emb = data["embeddings"].astype(np.float32)
    y = data["y"].astype(int)
    cent = _centroids(emb, y)
    within = _within_scatter(emb, y, cent)
    ventricular = np.isin(y, [1, 2])
    vtvf_silhouette = float(silhouette_score(emb[ventricular], y[ventricular])) if len(np.unique(y[ventricular])) == 2 else float("nan")
    return {
        "variant": row["variant"],
        "seed": int(row["seed"]),
        "run_dir": str(run_dir),
        "vt_vf_centroid_distance": float(np.linalg.norm(cent[1] - cent[2])),
        "sr_vt_centroid_distance": float(np.linalg.norm(cent[0] - cent[1])),
        "sr_vf_centroid_distance": float(np.linalg.norm(cent[0] - cent[2])),
        "vt_within_scatter": float(within[1]),
        "vf_within_scatter": float(within[2]),
        "vt_vf_norm_dist": _norm_dist(cent, within, 1, 2),
        "sr_vt_norm_dist": _norm_dist(cent, within, 0, 1),
        "sr_vf_norm_dist": _norm_dist(cent, within, 0, 2),
        "vtvf_silhouette": vtvf_silhouette,
        f"vtvf_mixing_k{k}": _vtvf_mixing(emb, y, k),
        "test_vtvf_cross_errors": float(row["vtvf_cross_errors"]),
        "auto_vtvf_cross_errors": _auto_vtvf_errors(run_dir),
    }


def _mean_std(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in {"variant", "seed", "run_dir"}]
    rows = []
    for variant, sub in df.groupby("variant"):
        out: dict[str, float | str | int] = {"variant": variant, "n": int(len(sub))}
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            out[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_seed_pca(run_df: pd.DataFrame, seed: int, out: Path, max_per_class: int, random_state: int) -> None:
    sub = run_df[run_df["seed"] == seed]
    if set(sub["variant"]) != {"baseline", "prototype_separation"}:
        return
    loaded = {}
    for _, row in sub.iterrows():
        data = np.load(Path(str(row["run_dir"])) / "embeddings_test.npz")
        loaded[row["variant"]] = (data["embeddings"].astype(np.float32), data["y"].astype(int))
    combined = np.concatenate([loaded["baseline"][0], loaded["prototype_separation"][0]], axis=0)
    pca = PCA(n_components=2, random_state=random_state).fit(combined)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    rng = np.random.default_rng(random_state)
    for ax, variant, title in [
        (axes[0], "baseline", "Baseline"),
        (axes[1], "prototype_separation", "PRO"),
    ]:
        emb, y = loaded[variant]
        pts = pca.transform(emb)
        keep_indices = []
        for c in [0, 1, 2]:
            idx = np.flatnonzero(y == c)
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep_indices.append(idx)
        keep = np.concatenate(keep_indices)
        for c, label in enumerate(CLASS_NAMES):
            idx = keep[y[keep] == c]
            alpha = 0.18 if c == 0 else 0.62
            size = 9 if c == 0 else 14
            ax.scatter(pts[idx, 0], pts[idx, 1], s=size, alpha=alpha, color=COLORS[c], label=label)
        cent = np.stack([pts[y == c].mean(axis=0) for c in range(3)])
        ax.scatter(cent[:, 0], cent[:, 1], s=120, marker="X", color=[COLORS[c] for c in range(3)], edgecolor="black", linewidth=0.8)
        ax.plot([cent[1, 0], cent[2, 0]], [cent[1, 1], cent[2, 1]], color="#D62728", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Shared PCA 1")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Shared PCA 2")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle(f"VT/VF embedding geometry before and after prototype separation (seed {seed})")
    fig.tight_layout()
    fig.savefig(out / f"baseline_vs_pro_embedding_pca_seed{seed}.png", dpi=220)
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, out: Path) -> None:
    order = ["baseline", "prototype_separation"]
    labels = ["Baseline", "PRO"]
    metrics = [
        ("vt_vf_norm_dist", "VT/VF normalized separation\n(distance / within-class scatter)"),
        ("auto_vtvf_cross_errors", "Automatic-route VT/VF errors"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(summary.set_index("variant").loc[v, f"{metric}_mean"]) for v in order]
        errs = [float(summary.set_index("variant").loc[v, f"{metric}_std"]) for v in order]
        ax.bar(labels, values, yerr=errs, color=["#9AA5B1", "#2E74B5"], capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "baseline_vs_pro_geometry_safety_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs PRO embedding geometry and safety metrics.")
    parser.add_argument(
        "--run-level",
        type=Path,
        default=Path("results/mitigation_v3_key_ablation_summary_full_analysis/mitigation_run_level_metrics.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/pro_geometry_comparison_20260606"))
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=650)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_df = _load_run_level(args.run_level, ["baseline", "prototype_separation"])
    run_level = pd.DataFrame([_run_geometry(row, args.k) for _, row in run_df.iterrows()])
    run_level.to_csv(args.out / "pro_geometry_run_level.csv", index=False)
    summary = _mean_std(run_level)
    summary.to_csv(args.out / "pro_geometry_mean_std.csv", index=False)
    _plot_seed_pca(run_df, args.plot_seed, args.out, args.max_per_class, random_state=42)
    _plot_summary(summary, args.out)
    print(summary.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
