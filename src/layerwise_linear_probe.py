from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASS_NAMES, build_duplicate_family_groups, load_rhythm_windows, make_splits
from .models import CNNLSTMClassifier, ECGCNN, build_model


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)


def _groups(dataset, split_grouping: str) -> np.ndarray:
    if split_grouping == "duplicate_family":
        return build_duplicate_family_groups(dataset.x, dataset.record_ids)
    return dataset.record_ids


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)


def _pool_time(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 3:
        mean = z.mean(dim=-1)
        std = z.std(dim=-1)
        return torch.cat([mean, std], dim=1)
    return z


@torch.no_grad()
def _extract_cnn_layers(model: ECGCNN, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    layers: dict[str, list[np.ndarray]] = {
        "conv1": [],
        "pool1": [],
        "conv2": [],
        "pool2": [],
        "conv3": [],
        "pre_embedding_pool": [],
        "final_embedding": [],
        "classifier_logits": [],
    }
    y_all: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        z = model.encoder[0](xb)
        layers["conv1"].append(_pool_time(z).cpu().numpy())
        z = model.encoder[1](z)
        layers["pool1"].append(_pool_time(z).cpu().numpy())
        z = model.encoder[2](z)
        layers["conv2"].append(_pool_time(z).cpu().numpy())
        z = model.encoder[3](z)
        layers["pool2"].append(_pool_time(z).cpu().numpy())
        z = model.encoder[4](z)
        layers["conv3"].append(_pool_time(z).cpu().numpy())
        pooled = model.encoder[5](z).squeeze(-1)
        layers["pre_embedding_pool"].append(pooled.cpu().numpy())
        emb = torch.relu(model.embedding(pooled))
        logits = model.classifier(emb)
        layers["final_embedding"].append(emb.cpu().numpy())
        layers["classifier_logits"].append(logits.cpu().numpy())
        y_all.append(yb.numpy())
    return {name: np.concatenate(values, axis=0) for name, values in layers.items()}, np.concatenate(y_all)


@torch.no_grad()
def _extract_cnn_lstm_layers(
    model: CNNLSTMClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    layers: dict[str, list[np.ndarray]] = {
        "cnn_conv1": [],
        "cnn_pool1": [],
        "cnn_conv2": [],
        "cnn_pool2": [],
        "cnn_conv3": [],
        "cnn_sequence": [],
        "lstm_last_state": [],
        "final_embedding": [],
        "classifier_logits": [],
    }
    y_all: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        z = model.cnn[0](xb)
        layers["cnn_conv1"].append(_pool_time(z).cpu().numpy())
        z = model.cnn[1](z)
        layers["cnn_pool1"].append(_pool_time(z).cpu().numpy())
        z = model.cnn[2](z)
        layers["cnn_conv2"].append(_pool_time(z).cpu().numpy())
        z = model.cnn[3](z)
        layers["cnn_pool2"].append(_pool_time(z).cpu().numpy())
        z = model.cnn[4](z)
        layers["cnn_conv3"].append(_pool_time(z).cpu().numpy())
        layers["cnn_sequence"].append(_pool_time(z).cpu().numpy())
        sequence = z.transpose(1, 2)
        _, (h, _) = model.lstm(sequence)
        state = torch.cat([h[-2], h[-1]], dim=1)
        layers["lstm_last_state"].append(state.cpu().numpy())
        emb = model.embedding(state)
        logits = model.classifier(emb)
        layers["final_embedding"].append(emb.cpu().numpy())
        layers["classifier_logits"].append(logits.cpu().numpy())
        y_all.append(yb.numpy())
    return {name: np.concatenate(values, axis=0) for name, values in layers.items()}, np.concatenate(y_all)


def _vtvf_cross_errors(y: np.ndarray, pred: np.ndarray) -> int:
    return int(np.sum(((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))))


def _probe_rows(train_layers: dict[str, np.ndarray], train_y: np.ndarray, test_layers: dict[str, np.ndarray], test_y: np.ndarray):
    rows = []
    for layer, x_train in train_layers.items():
        x_test = test_layers[layer]
        multi = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
        multi.fit(x_train, train_y)
        pred = multi.predict(x_test)
        rows.append(
            {
                "layer": layer,
                "probe": "sr_vt_vf_multiclass",
                "dim": int(x_train.shape[1]),
                "accuracy": float(accuracy_score(test_y, pred)),
                "macro_f1": float(f1_score(test_y, pred, average="macro")),
                "vtvf_cross_errors": _vtvf_cross_errors(test_y, pred),
                "auroc": np.nan,
            }
        )

        train_mask = np.isin(train_y, [1, 2])
        test_mask = np.isin(test_y, [1, 2])
        if train_mask.sum() > 10 and test_mask.sum() > 10:
            y_train_bin = (train_y[train_mask] == 2).astype(int)
            y_test_bin = (test_y[test_mask] == 2).astype(int)
            binary = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
            binary.fit(x_train[train_mask], y_train_bin)
            pred_bin = binary.predict(x_test[test_mask])
            score_bin = binary.predict_proba(x_test[test_mask])[:, 1]
            rows.append(
                {
                    "layer": layer,
                    "probe": "vt_vs_vf_binary",
                    "dim": int(x_train.shape[1]),
                    "accuracy": float(accuracy_score(y_test_bin, pred_bin)),
                    "macro_f1": float(f1_score(y_test_bin, pred_bin, average="macro")),
                    "vtvf_cross_errors": np.nan,
                    "auroc": float(roc_auc_score(y_test_bin, score_bin)) if len(np.unique(y_test_bin)) > 1 else np.nan,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true layer-wise linear probes on CNN or CNN-LSTM checkpoints.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["cnn", "cnn_lstm"], required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ckpt = _checkpoint(run_dir)
    ckpt_args = ckpt.get("args", {})
    seed = int(ckpt_args.get("seed", 42))
    split_grouping = str(ckpt_args.get("split_grouping", "record"))

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=_groups(dataset, split_grouping), seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_loader = _loader(splits.x_train, splits.y_train, args.batch_size)
    test_loader = _loader(splits.x_test, splits.y_test, args.batch_size)
    if isinstance(model, ECGCNN):
        train_layers, train_y = _extract_cnn_layers(model, train_loader, device)
        test_layers, test_y = _extract_cnn_layers(model, test_loader, device)
    elif isinstance(model, CNNLSTMClassifier):
        train_layers, train_y = _extract_cnn_lstm_layers(model, train_loader, device)
        test_layers, test_y = _extract_cnn_lstm_layers(model, test_loader, device)
    else:
        raise ValueError("This script currently supports only cnn and cnn_lstm.")

    if not np.array_equal(test_y, splits.y_test):
        raise RuntimeError("Layer extraction split mismatch.")

    out = pd.DataFrame(_probe_rows(train_layers, train_y, test_layers, test_y))
    out.insert(0, "model", "CNN" if args.model == "cnn" else "CNN-LSTM")
    out.insert(0, "seed", seed)
    out.to_csv(run_dir / "layerwise_linear_probe_summary.csv", index=False)

    report = {
        "run_dir": str(run_dir),
        "model": args.model,
        "seed": seed,
        "split_grouping": split_grouping,
        "layers": list(train_layers.keys()),
        "rows": int(len(out)),
    }
    (run_dir / "layerwise_linear_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
