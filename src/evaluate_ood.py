from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import softmax
from .models import build_model
from .uncertainty import energy, entropy, knn_distance, mahalanobis_distance, msp, prototype_distance
from .train import predict


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _make_ood(x: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = x + rng.normal(0.0, 0.75, size=x.shape).astype(np.float32)

    t = np.linspace(-1.0, 1.0, x.shape[-1], dtype=np.float32)
    drift = x + rng.uniform(-1.0, 1.0, size=(x.shape[0], 1, 1)).astype(np.float32) * t

    masked = x.copy()
    mask_len = x.shape[-1] // 5
    for i in range(masked.shape[0]):
        start = rng.integers(0, x.shape[-1] - mask_len + 1)
        masked[i, :, start : start + mask_len] = 0.0

    shuffled = x.copy()
    for i in range(shuffled.shape[0]):
        rng.shuffle(shuffled[i, 0])

    scaled = x * rng.uniform(0.2, 2.5, size=(x.shape[0], 1, 1)).astype(np.float32)
    return {
        "gaussian_noise": noise,
        "baseline_drift": drift,
        "random_masking": masked,
        "shuffled": shuffled,
        "amplitude_scaling": scaled,
    }


def _loader(x: np.ndarray, batch_size: int, features: np.ndarray | None = None) -> DataLoader:
    y = np.zeros(len(x), dtype=np.int64)
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["cnn", "tcn", "resnet1d", "regularity_fusion", "reliability_gated_fusion"],
        default="cnn",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    train_data = np.load(run_dir / "embeddings_train.npz")
    train_emb, train_y = train_data["embeddings"], train_data["y"]

    dataset = load_rhythm_windows(args.mat, max_windows_per_record=args.max_windows_per_record)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids)
    id_x = splits.x_test
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
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device)["model"])

    id_logits, id_emb, _ = predict(model, _loader(id_x, args.batch_size, features_for(id_x)), device)
    id_probs = softmax(id_logits)
    id_scores = {
        "msp": msp(id_probs),
        "entropy": entropy(id_probs),
        "energy": -energy(id_logits),
        "prototype": prototype_distance(train_emb, train_y, id_emb),
        "mahalanobis": mahalanobis_distance(train_emb, train_y, id_emb),
        "knn": knn_distance(train_emb, id_emb),
    }

    rows = []
    for ood_name, ood_x in _make_ood(id_x).items():
        ood_logits, ood_emb, _ = predict(model, _loader(ood_x, args.batch_size, features_for(ood_x)), device)
        ood_probs = softmax(ood_logits)
        ood_scores = {
            "msp": msp(ood_probs),
            "entropy": entropy(ood_probs),
            "energy": -energy(ood_logits),
            "prototype": prototype_distance(train_emb, train_y, ood_emb),
            "mahalanobis": mahalanobis_distance(train_emb, train_y, ood_emb),
            "knn": knn_distance(train_emb, ood_emb),
        }
        for score_name in id_scores:
            auroc, aupr = _ood_metrics(id_scores[score_name], ood_scores[score_name])
            rows.append({"ood_type": ood_name, "score": score_name, "auroc": auroc, "aupr": aupr})

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "ood_metrics.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
