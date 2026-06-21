from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import softmax
from .models import build_model


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _loader(x: np.ndarray, features: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _predict_gates(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, emb_all, gate_all, boundary_all, y_all = [], [], [], [], []
    for xb, fb, yb in loader:
        xb, fb = xb.to(device), fb.to(device)
        logits, emb, gate, boundary = model(xb, fb, return_embedding=True, return_gate=True)
        logits_all.append(logits.cpu().numpy())
        emb_all.append(emb.cpu().numpy())
        gate_all.append(gate.cpu().numpy())
        boundary_all.append(boundary.cpu().numpy())
        y_all.append(yb.numpy())
    return (
        np.concatenate(logits_all),
        np.concatenate(emb_all),
        np.concatenate(gate_all),
        np.concatenate(boundary_all),
        np.concatenate(y_all),
    )


def _case_type(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    case = np.full(len(y), "correct_or_other_error", dtype=object)
    case[(y == pred) & (y == 0)] = "correct_sr"
    case[(y == pred) & (y == 1)] = "correct_vt"
    case[(y == pred) & (y == 2)] = "correct_vf"
    case[(y == 1) & (pred == 2)] = "vt_as_vf"
    case[(y == 2) & (pred == 1)] = "vf_as_vt"
    case[(y != pred) & ~(((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1)))] = "other_error"
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="reliability_gated_fusion")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    if args.seed is None:
        ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        seed = int(ckpt.get("args", {}).get("seed", 42))
    else:
        seed = args.seed
    scaler = np.load(run_dir / "feature_scaler.npz", allow_pickle=True)
    mean, std = scaler["mean"], scaler["std"]

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=seed)
    test_raw = extract_regularity_features_batch(splits.x_test)
    test_features = ((test_raw - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)["model"])

    logits, emb, gate, boundary_logit, y = _predict_gates(
        model,
        _loader(splits.x_test, test_features, splits.y_test, args.batch_size),
        device,
    )
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": pred,
            "prob_sr": probs[:, 0],
            "prob_vt": probs[:, 1],
            "prob_vf": probs[:, 2],
            "gate": gate,
            "boundary_logit": boundary_logit,
            "case_type": _case_type(y, pred),
            "is_error": y != pred,
            "is_vtvf_error": ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1)),
        }
    )
    df.to_csv(run_dir / "gate_scores.csv", index=False)

    summary = (
        df.groupby("case_type")[["gate", "boundary_logit", "is_error", "is_vtvf_error"]]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.to_csv(run_dir / "gate_summary_by_case.csv", index=False)

    order = ["correct_sr", "correct_vt", "correct_vf", "vt_as_vf", "vf_as_vt", "other_error"]
    data = [df.loc[df["case_type"] == c, "gate"].to_numpy() for c in order if (df["case_type"] == c).any()]
    labels = [c for c in order if (df["case_type"] == c).any()]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Regularity gate")
    plt.tight_layout()
    plt.savefig(run_dir / "gate_by_case_type.png", dpi=180)
    plt.close()

    print(summary)


if __name__ == "__main__":
    main()
