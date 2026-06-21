from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASS_NAMES, extract_regularity_features_batch, load_rhythm_windows, make_splits
from .evaluate_corruption_severity import _corrupt
from .metrics import softmax
from .models import build_model
from .train import predict


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _loader(x: np.ndarray, batch_size: int, features: np.ndarray | None = None) -> DataLoader:
    y = np.zeros(len(x), dtype=np.int64)
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size)


def _robust_scale(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)


def _softmax_boundary(probs: np.ndarray) -> np.ndarray:
    ventricular_prob = probs[:, 1] + probs[:, 2]
    balance = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_prob, 1e-8)
    return (ventricular_prob * balance).astype(np.float32)


def _deployable_neighbor_scores(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    pred: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_emb)
    distances, idx = nn.kneighbors(emb)
    neigh_y = train_y[idx]

    predicted_support = (neigh_y == pred[:, None]).mean(axis=1)
    predicted_local_instability = (1.0 - predicted_support).astype(np.float32)

    ventricular_neighbors = np.isin(neigh_y, [1, 2])
    denom = ventricular_neighbors.sum(axis=1)
    predicted_vtvf_mixing = np.zeros(len(pred), dtype=np.float32)
    pred_vt = pred == 1
    pred_vf = pred == 2
    valid_vt = pred_vt & (denom > 0)
    valid_vf = pred_vf & (denom > 0)
    predicted_vtvf_mixing[valid_vt] = (neigh_y[valid_vt] == 2).sum(axis=1) / denom[valid_vt]
    predicted_vtvf_mixing[valid_vf] = (neigh_y[valid_vf] == 1).sum(axis=1) / denom[valid_vf]
    return distances[:, -1].astype(np.float32), predicted_local_instability, predicted_vtvf_mixing


def _review_curve(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    any_error = df["is_error"].to_numpy(bool)
    vtvf_error = df["is_vtvf_cross_error"].to_numpy(bool)
    for score_col in score_cols:
        score = df[score_col].to_numpy(float)
        order = np.argsort(-score)
        for burden in np.linspace(0.01, 0.50, 50):
            n_review = max(1, int(round(len(df) * burden)))
            review_idx = order[:n_review]
            auto_idx = order[n_review:]
            rows.append(
                {
                    "score": score_col,
                    "review_burden": float(burden),
                    "reviewed": int(n_review),
                    "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
                    "vtvf_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
                    "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                    "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _plot_review(curves: pd.DataFrame, out: Path, target: str) -> None:
    plt.figure(figsize=(7.0, 4.6))
    for score in curves["score"].unique():
        sub = curves[curves["score"] == score]
        plt.plot(sub["review_burden"], sub[target], marker="o", linewidth=1.5, markersize=3, label=score)
    plt.xlabel("Review burden")
    plt.ylabel(target.replace("_", " "))
    plt.ylim(0, 1.02)
    plt.grid(True, color="#e5e7eb", linewidth=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_distributions(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=170)
    groups = {
        "Correct": ~df["is_error"],
        "Any error": df["is_error"],
        "VT/VF cross-error": df["is_vtvf_cross_error"],
    }
    metrics = [
        ("pred_flip_rate", "Prediction flip rate"),
        ("embedding_drift_norm", "Embedding drift"),
        ("stability_aware_risk", "Stability-aware risk"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for label, mask in groups.items():
            vals = df.loc[mask, col].to_numpy(float)
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.45, density=True, label=label)
        ax.set_title(title)
        ax.grid(True, color="#e5e7eb", linewidth=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_confident_stable_cases(x_test: np.ndarray, cases: pd.DataFrame, out: Path, max_cases: int = 4) -> None:
    cases = cases.head(max_cases)
    if cases.empty:
        return
    fig, axes = plt.subplots(len(cases), 1, figsize=(9.5, 2.6 * len(cases)), dpi=160)
    if len(cases) == 1:
        axes = [axes]
    t = np.arange(x_test.shape[-1]) / 100.0
    for ax, (_, row) in zip(axes, cases.iterrows()):
        idx = int(row["index"])
        ax.plot(t, x_test[idx, 0], color="#111827", linewidth=1.0)
        true_name = CLASS_NAMES[int(row["y_true"])]
        pred_name = CLASS_NAMES[int(row["y_pred"])]
        ax.set_title(
            f"Confident-stable-error candidate #{idx}: true={true_name}, pred={pred_name}, "
            f"Pmax={row['max_softmax_prob']:.3f}, flip={row['pred_flip_rate']:.3f}, KNN={row['knn_norm']:.3f}",
            fontsize=9,
            loc="left",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG")
        ax.grid(True, color="#e5e7eb", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "tcn",
            "resnet1d",
            "inception_time",
            "bigru",
            "regularity_fusion",
            "reliability_gated_fusion",
        ],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--severity", type=int, default=1)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    x_test = splits.x_test.astype(np.float32)

    feature_scaler = None
    if args.model in {"regularity_fusion", "reliability_gated_fusion"}:
        scaler_data = np.load(run_dir / "feature_scaler.npz", allow_pickle=True)
        feature_scaler = (scaler_data["mean"], scaler_data["std"])

    def features_for(x: np.ndarray) -> np.ndarray | None:
        if feature_scaler is None:
            return None
        mean, std = feature_scaler
        return ((extract_regularity_features_batch(x) - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])

    base = np.load(run_dir / "embeddings_test.npz")
    train = np.load(run_dir / "embeddings_train.npz")
    base_logits = base["logits"]
    base_emb = base["embeddings"]
    train_emb = train["embeddings"]
    train_y = train["y"].astype(int)
    y = base["y"].astype(int)
    probs = softmax(base_logits)
    pred = probs.argmax(axis=1)
    base_conf = probs[np.arange(len(probs)), pred]

    rng = np.random.default_rng(args.seed)
    perturbations = ["gaussian_noise", "baseline_wander", "amplitude_scaling", "time_scaling"]
    flip_counts = np.zeros(len(y), dtype=np.float32)
    conf_drop_sum = np.zeros(len(y), dtype=np.float32)
    drift_sum = np.zeros(len(y), dtype=np.float32)
    total = 0
    for kind in perturbations:
        for _ in range(args.repeats):
            x_pert = _corrupt(x_test, kind, args.severity, rng).astype(np.float32)
            logits, emb, _ = predict(model, _loader(x_pert, args.batch_size, features_for(x_pert)), device)
            pert_probs = softmax(logits)
            pert_pred = pert_probs.argmax(axis=1)
            pert_conf_for_base = pert_probs[np.arange(len(pert_probs)), pred]
            flip_counts += (pert_pred != pred).astype(np.float32)
            conf_drop_sum += np.maximum(0.0, base_conf - pert_conf_for_base).astype(np.float32)
            drift_sum += np.linalg.norm(emb - base_emb, axis=1).astype(np.float32)
            total += 1

    knn_distance, deployable_local_instability, deployable_vtvf_mixing = _deployable_neighbor_scores(
        train_emb, train_y, base_emb, pred, k=15
    )
    risk_head_path = run_dir / "risk_scores_test.csv"
    if not risk_head_path.exists():
        risk_head_path = run_dir.parent / "20260602_002434_embedding_risk_head" / "risk_scores_test.csv"
    risk_head = pd.read_csv(risk_head_path) if risk_head_path.exists() else None

    df = pd.DataFrame(
        {
            "index": np.arange(len(y)),
            "y_true": y,
            "y_pred": pred,
            "is_error": y != pred,
            "is_vtvf_cross_error": ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1)),
            "max_softmax_prob": probs.max(axis=1),
            "msp": 1.0 - probs.max(axis=1),
            "entropy": _entropy(probs),
            "knn": knn_distance,
            "deployable_local_instability": deployable_local_instability,
            "deployable_vtvf_mixing": deployable_vtvf_mixing,
            "softmax_vtvf_ambiguity": _softmax_boundary(probs),
            "pred_flip_rate": flip_counts / max(total, 1),
            "confidence_drop": conf_drop_sum / max(total, 1),
            "embedding_drift": drift_sum / max(total, 1),
        }
    )
    df["entropy_norm"] = _robust_scale(df["entropy"].to_numpy(float))
    df["knn_norm"] = _robust_scale(df["knn"].to_numpy(float))
    df["deployable_local_instability_norm"] = _robust_scale(df["deployable_local_instability"].to_numpy(float))
    df["deployable_vtvf_mixing_norm"] = _robust_scale(df["deployable_vtvf_mixing"].to_numpy(float))
    df["boundary_norm"] = _robust_scale(df["softmax_vtvf_ambiguity"].to_numpy(float))
    df["pred_flip_norm"] = _robust_scale(df["pred_flip_rate"].to_numpy(float))
    df["confidence_drop_norm"] = _robust_scale(df["confidence_drop"].to_numpy(float))
    df["embedding_drift_norm"] = _robust_scale(df["embedding_drift"].to_numpy(float))
    df["stability_risk"] = np.clip(
        0.45 * df["pred_flip_norm"] + 0.40 * df["embedding_drift_norm"] + 0.15 * df["confidence_drop_norm"],
        0.0,
        1.0,
    )
    df["stability_aware_risk"] = np.clip(
        0.18 * df["entropy_norm"]
        + 0.18 * df["deployable_local_instability_norm"]
        + 0.16 * df["deployable_vtvf_mixing_norm"]
        + 0.14 * df["knn_norm"]
        + 0.12 * df["boundary_norm"]
        + 0.12 * df["pred_flip_norm"]
        + 0.10 * df["embedding_drift_norm"],
        0.0,
        1.0,
    )
    if risk_head is not None and len(risk_head) == len(df):
        df["risk_head"] = risk_head["risk_score"].to_numpy(float)
    df["confident_stable_error"] = (
        df["is_error"]
        & (df["max_softmax_prob"] >= 0.70)
        & (df["pred_flip_rate"] <= 0.15)
        & (df["knn_norm"] >= 0.50)
    )

    df.to_csv(run_dir / "stability_scores.csv", index=False)
    score_cols = [
        "entropy_norm",
        "deployable_local_instability_norm",
        "deployable_vtvf_mixing_norm",
        "stability_risk",
        "stability_aware_risk",
    ]
    if "risk_head" in df:
        score_cols.insert(3, "risk_head")
    curves = _review_curve(df, score_cols)
    curves.to_csv(run_dir / "stability_review_curves.csv", index=False)
    _plot_review(curves, run_dir / "stability_review_vtvf_capture.png", "vtvf_error_captured")
    _plot_review(curves, run_dir / "stability_review_all_error_capture.png", "all_error_captured")
    _plot_distributions(df, run_dir / "stability_distributions.png")

    cases = df[df["confident_stable_error"]].sort_values(["knn_norm", "msp"], ascending=False)
    cases.to_csv(run_dir / "confident_stable_error_cases.csv", index=False)
    _plot_confident_stable_cases(x_test, cases, run_dir / "confident_stable_error_cases.png")

    summary_rows = []
    for group_name, mask in {
        "correct": ~df["is_error"],
        "any_error": df["is_error"],
        "vtvf_cross_error": df["is_vtvf_cross_error"],
        "confident_stable_error": df["confident_stable_error"],
    }.items():
        sub = df[mask]
        summary_rows.append(
            {
                "group": group_name,
                "n": int(len(sub)),
                "pred_flip_rate_mean": float(sub["pred_flip_rate"].mean()) if len(sub) else np.nan,
                "embedding_drift_mean": float(sub["embedding_drift"].mean()) if len(sub) else np.nan,
                "stability_aware_risk_mean": float(sub["stability_aware_risk"].mean()) if len(sub) else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(run_dir / "stability_summary.csv", index=False)
    print(pd.DataFrame(summary_rows))
    print(curves[curves["review_burden"].isin([0.1, 0.2, 0.3])].sort_values(["review_burden", "vtvf_error_captured"], ascending=[True, False]))


if __name__ == "__main__":
    main()
