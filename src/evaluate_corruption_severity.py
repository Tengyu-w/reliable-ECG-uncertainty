from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import softmax
from .models import build_model
from .train import predict
from .uncertainty import entropy, knn_distance, mahalanobis_distance, msp, prototype_distance


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


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _corrupt(x: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    scale = severity / 4.0
    n = y.shape[-1]
    if kind == "gaussian_noise":
        return y + rng.normal(0.0, 0.15 + 0.85 * scale, size=y.shape).astype(np.float32)
    if kind == "powerline_interference":
        t = np.arange(n, dtype=np.float32) / 100.0
        phase = rng.uniform(0, 2 * np.pi, size=(y.shape[0], 1, 1)).astype(np.float32)
        freq = 50.0
        amp = 0.05 + 0.45 * scale
        return (y + amp * np.sin(2 * np.pi * freq * t[None, None, :] + phase)).astype(np.float32)
    if kind == "baseline_wander":
        t = np.linspace(0, 2 * np.pi, y.shape[-1], dtype=np.float32)
        phase = rng.uniform(0, 2 * np.pi, size=(y.shape[0], 1, 1)).astype(np.float32)
        return y + (0.2 + 1.2 * scale) * np.sin(t[None, None, :] + phase)
    if kind == "baseline_jump":
        jump = rng.uniform(-1.0, 1.0, size=(y.shape[0], 1, 1)).astype(np.float32) * (0.4 + 1.6 * scale)
        for i in range(y.shape[0]):
            start = rng.integers(n // 5, 4 * n // 5)
            y[i, :, start:] += jump[i]
        return y.astype(np.float32)
    if kind == "random_masking":
        mask_len = max(1, int(y.shape[-1] * (0.05 + 0.35 * scale)))
        for i in range(y.shape[0]):
            start = rng.integers(0, y.shape[-1] - mask_len + 1)
            y[i, :, start : start + mask_len] = 0.0
        return y
    if kind == "flatline_dropout":
        mask_len = max(1, int(n * (0.08 + 0.55 * scale)))
        for i in range(y.shape[0]):
            start = rng.integers(0, n - mask_len + 1)
            y[i, :, start : start + mask_len] = y[i, :, start]
        return y.astype(np.float32)
    if kind == "spike":
        n_spikes = 1 + severity * 2
        amp = 1.0 + 4.0 * scale
        for i in range(y.shape[0]):
            idx = rng.integers(0, y.shape[-1], size=n_spikes)
            y[i, 0, idx] += rng.choice([-amp, amp], size=n_spikes)
        return y
    if kind == "amplitude_scaling":
        factors = rng.uniform(1.0 - 0.75 * scale, 1.0 + 1.75 * scale, size=(y.shape[0], 1, 1))
        return (y * factors.astype(np.float32)).astype(np.float32)
    if kind == "clipping_saturation":
        threshold = np.quantile(np.abs(y), max(0.35, 0.95 - 0.13 * severity), axis=-1, keepdims=True)
        return np.clip(y, -threshold, threshold).astype(np.float32)
    if kind == "time_scaling":
        out = np.empty_like(y)
        old_grid = np.linspace(0, 1, n)
        for i in range(y.shape[0]):
            factor = 1.0 + rng.choice([-1.0, 1.0]) * (0.05 + 0.2 * scale)
            warped = np.clip((old_grid - 0.5) * factor + 0.5, 0, 1)
            out[i, 0] = np.interp(old_grid, warped, y[i, 0])
        return out.astype(np.float32)
    if kind == "mixed_noise_baseline":
        y = _corrupt(y, "gaussian_noise", severity, rng)
        y = _corrupt(y, "baseline_wander", severity, rng)
        return y.astype(np.float32)
    raise ValueError(kind)


def _scores(
    logits: np.ndarray,
    emb: np.ndarray,
    train_emb: np.ndarray,
    train_y: np.ndarray,
) -> dict[str, np.ndarray]:
    probs = softmax(logits)
    ent = entropy(probs)
    knn = knn_distance(train_emb, emb)
    maha = mahalanobis_distance(train_emb, train_y, emb)
    return {
        "msp": msp(probs),
        "entropy": ent,
        "prototype": prototype_distance(train_emb, train_y, emb),
        "mahalanobis": maha,
        "knn": knn,
        "hybrid_entropy_knn": 0.5 * _normalise(ent) + 0.5 * _normalise(knn),
        "hybrid_entropy_mahalanobis": 0.5 * _normalise(ent) + 0.5 * _normalise(maha),
    }


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
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    train_data = np.load(run_dir / "embeddings_train.npz")
    train_emb, train_y = train_data["embeddings"], train_data["y"]

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    x_id = splits.x_test
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
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)["model"])

    id_logits, id_emb, _ = predict(model, _loader(x_id, args.batch_size, features_for(x_id)), device)
    id_scores = _scores(id_logits, id_emb, train_emb, train_y)
    rng = np.random.default_rng(args.seed)

    rows = []
    corruptions = [
        "gaussian_noise",
        "powerline_interference",
        "baseline_wander",
        "baseline_jump",
        "random_masking",
        "flatline_dropout",
        "spike",
        "amplitude_scaling",
        "clipping_saturation",
        "time_scaling",
        "mixed_noise_baseline",
    ]
    for kind in corruptions:
        for severity in range(1, 5):
            x_ood = _corrupt(x_id, kind, severity, rng)
            x_ood = x_ood.astype(np.float32)
            logits, emb, _ = predict(model, _loader(x_ood, args.batch_size, features_for(x_ood)), device)
            ood_scores = _scores(logits, emb, train_emb, train_y)
            for score_name, id_score in id_scores.items():
                ood_score = ood_scores[score_name]
                labels = np.concatenate([np.zeros_like(id_score), np.ones_like(ood_score)])
                values = np.concatenate([id_score, ood_score])
                rows.append(
                    {
                        "corruption": kind,
                        "severity": severity,
                        "score": score_name,
                        "id_mean": float(np.mean(id_score)),
                        "ood_mean": float(np.mean(ood_score)),
                        "mean_shift": float(np.mean(ood_score) - np.mean(id_score)),
                        "auroc": float(roc_auc_score(labels, values)),
                        "aupr": float(average_precision_score(labels, values)),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "corruption_severity_metrics.csv", index=False)

    selected = out[out["score"].isin(["msp", "mahalanobis", "knn", "hybrid_entropy_knn"])]
    for corruption in selected["corruption"].unique():
        plt.figure(figsize=(6, 4))
        sub = selected[selected["corruption"] == corruption]
        for score in sub["score"].unique():
            score_sub = sub[sub["score"] == score]
            plt.plot(score_sub["severity"], score_sub["auroc"], marker="o", label=score)
        plt.title(corruption)
        plt.xlabel("Severity")
        plt.ylabel("OOD AUROC")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"severity_{corruption}.png", dpi=180)
        plt.close()

    print(out)


if __name__ == "__main__":
    main()
