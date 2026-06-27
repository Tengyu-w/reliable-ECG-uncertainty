from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_duplicate_family_groups, load_rhythm_windows, make_splits
from .metrics import classification_metrics, expected_calibration_error


class SSLConvEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(128, embedding_dim), nn.ReLU(inplace=True))
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, project: bool = False) -> torch.Tensor:
        emb = self.embedding(self.encoder(x))
        if project:
            return self.projector(emb)
        return emb


def _augment(x: torch.Tensor, noise_std: float = 0.04, mask_prob: float = 0.05) -> torch.Tensor:
    out = x.clone()
    scale = 1.0 + torch.randn((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * 0.08
    out = out * scale.clamp(0.75, 1.25)
    signal_std = x.flatten(1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    out = out + torch.randn_like(out) * signal_std * noise_std
    if mask_prob > 0:
        out = out * (torch.rand_like(out) > mask_prob)
    max_shift = max(1, x.shape[-1] // 40)
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    shifted = []
    for sample, shift in zip(out, shifts):
        shifted.append(torch.roll(sample, int(shift.item()), dims=-1))
    return torch.stack(shifted, dim=0)


def _barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 0.005) -> torch.Tensor:
    z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0).clamp_min(1e-6)
    z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0).clamp_min(1e-6)
    c = z1.T @ z2 / z1.shape[0]
    diag = torch.diagonal(c).add(-1.0).pow(2).sum()
    off_diag = c.flatten()[:-1].view(c.shape[0] - 1, c.shape[1] + 1)[:, 1:].flatten().pow(2).sum()
    return diag + lambd * off_diag


def _train_ssl_encoder(
    x_train: np.ndarray,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    max_train_windows: int | None,
) -> tuple[SSLConvEncoder, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    if max_train_windows is not None and len(x_train) > max_train_windows:
        idx = rng.choice(len(x_train), size=max_train_windows, replace=False)
        x_ssl = x_train[idx]
    else:
        x_ssl = x_train
    loader = DataLoader(TensorDataset(torch.from_numpy(x_ssl)), batch_size=batch_size, shuffle=True, drop_last=True)
    model = SSLConvEncoder(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            z1 = model(_augment(xb), project=True)
            z2 = model(_augment(xb), project=True)
            loss = _barlow_twins_loss(z1, z2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "ssl_loss": float(np.mean(losses)) if losses else float("nan")})
    return model, history


@torch.no_grad()
def _encode(model: SSLConvEncoder, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    parts = []
    for (xb,) in loader:
        parts.append(model(xb.to(device), project=False).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _fit_classifier(train_emb: np.ndarray, train_y: np.ndarray) -> Any:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(train_emb, train_y)
    return clf


def _fit_binary_head(x: np.ndarray, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        return float(y.mean())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"))
    model.fit(x, y)
    return model


def _predict_binary(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(x), model, dtype=np.float32)
    return model.predict_proba(x)[:, 1].astype(np.float32)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return (-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(probs.shape[1])).astype(np.float32)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _review_row(seed: int, method: str, score: np.ndarray, y: np.ndarray, pred: np.ndarray, budget: float) -> dict[str, float | int | str]:
    n = max(1, int(round(len(score) * budget)))
    selected = np.zeros(len(score), dtype=bool)
    selected[np.argsort(-score)[:n]] = True
    auto = ~selected
    is_error = y != pred
    is_vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    return {
        "seed": seed,
        "method": method,
        "budget": budget,
        "action_rate": float(selected.mean()),
        "all_error_addressed": float((is_error & selected).sum() / max(is_error.sum(), 1)),
        "vtvf_cross_error_addressed": float((is_vtvf & selected).sum() / max(is_vtvf.sum(), 1)),
        "automatic_unresolved_error_rate": float((is_error & auto).mean()),
        "automatic_unresolved_vtvf_cross_error_rate": float((is_vtvf & auto).mean()),
        "single_label_error_rate_after_routing": float((is_error & auto).sum() / max(auto.sum(), 1)),
        "single_label_vtvf_cross_error_rate_after_routing": float((is_vtvf & auto).sum() / max(auto.sum(), 1)),
    }


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _run_seed(args: argparse.Namespace, seed: int, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = load_rhythm_windows(args.mat, max_windows_per_record=args.max_windows_per_record)
    groups = None
    if args.split_grouping == "duplicate_family":
        groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    elif args.split_grouping == "record":
        groups = dataset.record_ids
    splits = make_splits(dataset.x, dataset.y, groups=groups, seed=seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, history = _train_ssl_encoder(
        splits.x_train,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        device=device,
        max_train_windows=args.max_train_windows,
    )
    train_emb = _encode(model, splits.x_train, args.batch_size, device)
    val_emb = _encode(model, splits.x_val, args.batch_size, device)
    test_emb = _encode(model, splits.x_test, args.batch_size, device)

    clf = _fit_classifier(train_emb, splits.y_train)
    val_probs = clf.predict_proba(val_emb)
    test_probs = clf.predict_proba(test_emb)
    val_pred = val_probs.argmax(axis=1)
    test_pred = test_probs.argmax(axis=1)
    val_features = np.concatenate([val_emb, val_probs, _entropy(val_probs)[:, None], (1.0 - val_probs.max(axis=1))[:, None]], axis=1)
    test_features = np.concatenate([test_emb, test_probs, _entropy(test_probs)[:, None], (1.0 - test_probs.max(axis=1))[:, None]], axis=1)
    val_any_error = (val_pred != splits.y_val).astype(int)
    val_vtvf_error = (((splits.y_val == 1) & (val_pred == 2)) | ((splits.y_val == 2) & (val_pred == 1))).astype(int)
    test_any_error = (test_pred != splits.y_test).astype(int)
    test_vtvf_error = (((splits.y_test == 1) & (test_pred == 2)) | ((splits.y_test == 2) & (test_pred == 1))).astype(int)

    any_head = _fit_binary_head(val_features, val_any_error)
    boundary_head = _fit_binary_head(val_features, val_vtvf_error)
    any_score = _predict_binary(any_head, test_features)
    boundary_score = _predict_binary(boundary_head, test_features)

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(seed_out / "ssl_history.csv", index=False)
    np.savez(
        seed_out / "frozen_ssl_embeddings.npz",
        train=train_emb,
        val=val_emb,
        test=test_emb,
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
    )
    pred_df = pd.DataFrame(test_probs, columns=["prob_sr", "prob_vt", "prob_vf"])
    pred_df["y_true"] = splits.y_test
    pred_df["y_pred"] = test_pred
    pred_df["ssl_any_error_risk"] = any_score
    pred_df["ssl_vtvf_boundary_risk"] = boundary_score
    pred_df["is_error"] = test_any_error.astype(bool)
    pred_df["is_vtvf_cross_error"] = test_vtvf_error.astype(bool)
    pred_df.to_csv(seed_out / "frozen_ssl_predictions_test.csv", index=False)

    metrics = classification_metrics(splits.y_test, test_probs)
    metrics["ece"] = expected_calibration_error(splits.y_test, test_probs)
    metrics["seed"] = seed
    metrics["method"] = "frozen_self_supervised_encoder"
    metrics["split_grouping"] = args.split_grouping
    metrics["ssl_epochs"] = args.epochs
    metrics["max_train_windows"] = args.max_train_windows
    metrics["any_error_auroc"] = _safe_auc(test_any_error, any_score)
    metrics["any_error_aupr"] = _safe_aupr(test_any_error, any_score)
    metrics["vtvf_error_auroc"] = _safe_auc(test_vtvf_error, boundary_score)
    metrics["vtvf_error_aupr"] = _safe_aupr(test_vtvf_error, boundary_score)
    (seed_out / "frozen_ssl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    policy_rows = []
    for budget in args.budgets:
        policy_rows.append(_review_row(seed, "ssl_any_error_risk", any_score, splits.y_test, test_pred, budget))
        policy_rows.append(_review_row(seed, "ssl_vtvf_boundary_risk", boundary_score, splits.y_test, test_pred, budget))
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(seed_out / "frozen_ssl_review_policy.csv", index=False)
    return metrics, policy


def _write_report(out_dir: Path, metrics_summary: pd.DataFrame, policy_summary: pd.DataFrame) -> None:
    lines = [
        "# Frozen Self-Supervised Encoder Comparison",
        "",
        "## Scope",
        "",
        "This is a lightweight frozen self-supervised ECG encoder comparison. It is not an external ECG foundation-model validation. The goal is to test whether a label-free frozen encoder is worth promoting into the final review-routing pipeline.",
        "",
        "## Classification and risk-head summary",
        "",
    ]
    if not metrics_summary.empty:
        row = metrics_summary.iloc[0]
        lines.extend(
            [
                "| metric | mean | std |",
                "|---|---:|---:|",
                f"| accuracy | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} |",
                f"| macro-F1 | {row['macro_f1_mean']:.4f} | {row['macro_f1_std']:.4f} |",
                f"| ECE | {row['ece_mean']:.4f} | {row['ece_std']:.4f} |",
                f"| any-error AUROC | {row['any_error_auroc_mean']:.4f} | {row['any_error_auroc_std']:.4f} |",
                f"| VT/VF-error AUROC | {row['vtvf_error_auroc_mean']:.4f} | {row['vtvf_error_auroc_std']:.4f} |",
                "",
            ]
        )
    focus = policy_summary[policy_summary["budget"] == 0.20] if not policy_summary.empty else pd.DataFrame()
    if not focus.empty:
        lines.extend(
            [
                "## Review-routing at 20% action budget",
                "",
                "| method | all-error capture | VT/VF capture | unresolved VT/VF rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in focus.iterrows():
            lines.append(
                f"| {row['method']} | {row['all_error_addressed_mean']:.4f} | "
                f"{row['vtvf_cross_error_addressed_mean']:.4f} | "
                f"{row['automatic_unresolved_vtvf_cross_error_rate_mean']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- If the frozen SSL encoder is weaker than v5d, it should be kept as a foundation-readiness baseline rather than replacing the final method.",
            "- If it is competitive, the next upgrade should swap this lightweight encoder for a real external pretrained ECG foundation model and rerun the same frozen-head protocol.",
            "- Claims should remain internal because the encoder is trained on the same ECG source, without external pretraining or external validation.",
            "",
        ]
    )
    (out_dir / "FROZEN_SSL_ENCODER_COMPARISON_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight frozen self-supervised ECG encoder comparison.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--out", type=Path, default=Path("results/frozen_ssl_encoder_comparison_20260627"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-grouping", choices=["record", "duplicate_family", "none"], default="duplicate_family")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_policy = []
    for seed in args.seeds:
        metrics, policy = _run_seed(args, seed, args.out)
        all_metrics.append(metrics)
        all_policy.append(policy)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.out / "frozen_ssl_metrics_seed_level.csv", index=False)
    metric_cols = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "any_error_auroc", "any_error_aupr", "vtvf_error_auroc", "vtvf_error_aupr"]
    metrics_summary = _mean_std(metrics_df, ["method", "split_grouping"], [c for c in metric_cols if c in metrics_df.columns])
    metrics_summary.to_csv(args.out / "frozen_ssl_metrics_mean_std.csv", index=False)

    policy_df = pd.concat(all_policy, ignore_index=True)
    policy_df.to_csv(args.out / "frozen_ssl_review_policy_seed_level.csv", index=False)
    policy_summary = _mean_std(
        policy_df,
        ["method", "budget"],
        [
            "action_rate",
            "all_error_addressed",
            "vtvf_cross_error_addressed",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
            "single_label_error_rate_after_routing",
            "single_label_vtvf_cross_error_rate_after_routing",
        ],
    )
    policy_summary.to_csv(args.out / "frozen_ssl_review_policy_mean_std.csv", index=False)
    (args.out / "frozen_ssl_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    _write_report(args.out, metrics_summary, policy_summary)
    print(f"Wrote frozen SSL encoder comparison to {args.out}")


if __name__ == "__main__":
    main()
