from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import REGULARITY_FEATURE_NAMES, extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import softmax
from .models import build_model


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _features_from_scaler(x: np.ndarray, scaler_path: Path) -> np.ndarray:
    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"]
    std = scaler["std"]
    raw = extract_regularity_features_batch(x)
    return ((raw - mean) / std).astype(np.float32)


def _loader(x: np.ndarray, features: np.ndarray, risk: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(features),
        torch.from_numpy(risk.astype(np.float32)),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_risk(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for xb, fb, rb, yb in loader:
        xb = xb.to(device)
        fb = fb.to(device)
        logits, emb, gate, boundary_logit = model(xb, fb, return_embedding=True, return_gate=True)
        probs = softmax(logits.cpu().numpy())
        rows.append(
            pd.DataFrame(
                {
                    "risk_target": rb.numpy(),
                    "risk_score": torch.sigmoid(boundary_logit).cpu().numpy(),
                    "gate": gate.cpu().numpy(),
                    "y_true": yb.numpy(),
                    "y_pred": probs.argmax(axis=1),
                    "confidence": probs.max(axis=1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _risk_metrics(df: pd.DataFrame) -> dict[str, float]:
    is_error = (df["y_true"].to_numpy() != df["y_pred"].to_numpy()).astype(int)
    score = df["risk_score"].to_numpy()
    target = df["risk_target"].to_numpy()
    corr = spearmanr(score, target).correlation
    metrics = {
        "risk_mse": float(np.mean((score - target) ** 2)),
        "risk_mae": float(np.mean(np.abs(score - target))),
        "risk_target_spearman": float(corr) if not np.isnan(corr) else float("nan"),
    }
    if len(np.unique(is_error)) > 1:
        metrics["error_auroc"] = float(roc_auc_score(is_error, score))
        metrics["error_aupr"] = float(average_precision_score(is_error, score))
    vtvf_error = (
        ((df["y_true"].to_numpy() == 1) & (df["y_pred"].to_numpy() == 2))
        | ((df["y_true"].to_numpy() == 2) & (df["y_pred"].to_numpy() == 1))
    ).astype(int)
    if len(np.unique(vtvf_error)) > 1:
        metrics["vtvf_error_auroc"] = float(roc_auc_score(vtvf_error, score))
        metrics["vtvf_error_aupr"] = float(average_precision_score(vtvf_error, score))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--risk-targets", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--run-suffix", type=str, default="risk_head")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    teacher_dir = _resolve_run_dir(args.teacher_run_dir)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_reliability_gated_fusion_{args.run_suffix}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "latest").write_text(str(run_dir), encoding="utf-8")

    for name in [
        "feature_scaler.npz",
        "split_summary.json",
        "metrics.json",
        "test_predictions.csv",
        "embeddings_train.npz",
        "embeddings_val.npz",
        "embeddings_test.npz",
    ]:
        src = teacher_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    train_features = _features_from_scaler(splits.x_train, teacher_dir / "feature_scaler.npz")
    val_features = _features_from_scaler(splits.x_val, teacher_dir / "feature_scaler.npz")
    test_features = _features_from_scaler(splits.x_test, teacher_dir / "feature_scaler.npz")
    risk = np.load(args.risk_targets)
    train_risk = risk["train"].astype(np.float32)
    val_risk = risk["val"].astype(np.float32)
    test_risk = risk["test"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("reliability_gated_fusion", num_classes=3, feature_dim=len(REGULARITY_FEATURE_NAMES)).to(device)
    checkpoint = torch.load(teacher_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.boundary_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.boundary_head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = _loader(splits.x_train, train_features, train_risk, splits.y_train, args.batch_size, shuffle=True)
    val_loader = _loader(splits.x_val, val_features, val_risk, splits.y_val, args.batch_size, shuffle=False)
    test_loader = _loader(splits.x_test, test_features, test_risk, splits.y_test, args.batch_size, shuffle=False)

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.boundary_head.train()
        losses = []
        for xb, fb, rb, _ in tqdm(train_loader, desc=f"risk-head epoch {epoch}", leave=False):
            xb = xb.to(device)
            fb = fb.to(device)
            rb = rb.to(device)
            optimizer.zero_grad()
            _, _, _, boundary_logit = model(xb, fb, return_embedding=True, return_gate=True)
            loss = loss_fn(torch.sigmoid(boundary_logit), rb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_df = _predict_risk(model, val_loader, device)
        val_mse = float(np.mean((val_df["risk_score"].to_numpy() - val_df["risk_target"].to_numpy()) ** 2))
        row = {"epoch": epoch, "train_risk_mse": float(np.mean(losses)), "val_risk_mse": val_mse}
        history.append(row)
        print(row)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "args": {**checkpoint.get("args", {}), "risk_head_finetuned": True}}, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "risk_head_history.csv", index=False)

    summary = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        df = _predict_risk(model, loader, device)
        df.to_csv(run_dir / f"risk_head_scores_{split}.csv", index=False)
        summary[split] = _risk_metrics(df)
    (run_dir / "risk_head_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
