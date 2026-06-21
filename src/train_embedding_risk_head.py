from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingRiskHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load_embeddings(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["embeddings"].astype(np.float32), data["logits"].astype(np.float32), data["y"].astype(np.int64)


def _loader(emb: np.ndarray, risk: np.ndarray, y: np.ndarray, pred: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(emb),
        torch.from_numpy(risk.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
        torch.from_numpy(pred.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for emb, risk, y, pred in loader:
        score = torch.sigmoid(model(emb.to(device))).cpu().numpy()
        rows.append(
            pd.DataFrame(
                {
                    "risk_target": risk.numpy(),
                    "risk_score": score,
                    "y_true": y.numpy(),
                    "y_pred": pred.numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    score = df["risk_score"].to_numpy()
    target = df["risk_target"].to_numpy()
    y = df["y_true"].to_numpy()
    pred = df["y_pred"].to_numpy()
    is_error = (y != pred).astype(int)
    vtvf_error = (((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))).astype(int)
    corr = spearmanr(score, target).correlation
    out = {
        "risk_mse": float(np.mean((score - target) ** 2)),
        "risk_mae": float(np.mean(np.abs(score - target))),
        "risk_target_spearman": float(corr) if not np.isnan(corr) else float("nan"),
    }
    if len(np.unique(is_error)) > 1:
        out["error_auroc"] = float(roc_auc_score(is_error, score))
        out["error_aupr"] = float(average_precision_score(is_error, score))
    if len(np.unique(vtvf_error)) > 1:
        out["vtvf_error_auroc"] = float(roc_auc_score(vtvf_error, score))
        out["vtvf_error_aupr"] = float(average_precision_score(vtvf_error, score))
    return out


def _review_curve(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y_true"].to_numpy()
    pred = df["y_pred"].to_numpy()
    any_error = y != pred
    ventricular_error = np.isin(y, [1, 2]) & any_error
    vtvf_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    order = np.argsort(-df["risk_score"].to_numpy())
    rows = []
    for burden in np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]):
        n_review = max(1, int(round(len(df) * burden)))
        review_idx = order[:n_review]
        auto_idx = order[n_review:]
        rows.append(
            {
                "score": "embedding_risk_head",
                "review_burden": float(burden),
                "reviewed": int(n_review),
                "auto_coverage": float(len(auto_idx) / len(df)),
                "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
                "ventricular_error_captured": float(ventricular_error[review_idx].sum() / max(ventricular_error.sum(), 1)),
                "vtvf_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
                "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                "review_error_enrichment": float(any_error[review_idx].mean() / max(any_error.mean(), 1e-8)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--risk-targets", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--run-suffix", type=str, default="embedding_risk_head")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    teacher_dir = _resolve_run_dir(args.teacher_run_dir)
    risk = np.load(args.risk_targets)
    split_data = {}
    for split in ["train", "val", "test"]:
        emb, logits, y = _load_embeddings(teacher_dir, split)
        pred = logits.argmax(axis=1)
        split_data[split] = (emb, risk[split].astype(np.float32), y, pred)

    run_dir = args.out / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.run_suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "latest").write_text(str(run_dir), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingRiskHead(split_data["train"][0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = _loader(*split_data["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = _loader(*split_data["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = _loader(*split_data["test"], batch_size=args.batch_size, shuffle=False)

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for emb, rb, _, _ in train_loader:
            optimizer.zero_grad()
            pred = torch.sigmoid(model(emb.to(device)))
            loss = loss_fn(pred, rb.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_df = _predict(model, val_loader, device)
        val_mse = float(np.mean((val_df["risk_score"].to_numpy() - val_df["risk_target"].to_numpy()) ** 2))
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val_mse}
        history.append(row)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model": model.state_dict(), "teacher_run_dir": str(teacher_dir)}, run_dir / "embedding_risk_head.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    summary = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        df = _predict(model, loader, device)
        df.to_csv(run_dir / f"risk_scores_{split}.csv", index=False)
        summary[split] = _metrics(df)
    test_curve = _review_curve(pd.read_csv(run_dir / "risk_scores_test.csv"))
    test_curve.to_csv(run_dir / "review_curves.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(test_curve["review_burden"], test_curve["vtvf_error_captured"], marker="o", label="VT/VF errors")
    plt.plot(test_curve["review_burden"], test_curve["all_error_captured"], marker="o", label="all errors")
    plt.xlabel("Review burden")
    plt.ylabel("Error captured")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "review_burden.png", dpi=180)
    plt.close()

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(test_curve[test_curve["review_burden"].isin([0.1, 0.2, 0.3])])


if __name__ == "__main__":
    main()
