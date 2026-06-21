from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import (
    CLASS_NAMES,
    REGULARITY_FEATURE_NAMES,
    build_duplicate_family_groups,
    extract_regularity_features_batch,
    load_rhythm_windows,
    make_splits,
)
from .metrics import classification_metrics, expected_calibration_error, softmax
from .models import build_model


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    features: np.ndarray | None = None,
    risk_targets: np.ndarray | None = None,
) -> DataLoader:
    if features is None and risk_targets is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    elif risk_targets is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    elif features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(risk_targets), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(risk_targets), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _mild_perturbation(
    x: torch.Tensor,
    noise_std: float,
    scale_std: float,
    mask_prob: float,
) -> torch.Tensor:
    """Generate mild ECG perturbations for stability-consistency training."""
    x_aug = x
    if noise_std > 0:
        signal_std = x.detach().flatten(1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        x_aug = x_aug + torch.randn_like(x_aug) * signal_std * noise_std
    if scale_std > 0:
        scale = 1.0 + torch.randn((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * scale_std
        x_aug = x_aug * scale.clamp(0.5, 1.5)
    if mask_prob > 0:
        keep = torch.rand_like(x_aug) > mask_prob
        x_aug = x_aug * keep
    return x_aug


def _forward_with_embedding(
    model: nn.Module,
    x: torch.Tensor,
    features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features is not None:
        return model(x, features, return_embedding=True)
    return model(x, return_embedding=True)


def _prototype_losses(
    emb: torch.Tensor,
    y: torch.Tensor,
    center_weight: float,
    margin_weight: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    center_loss = torch.zeros((), device=emb.device)
    margin_loss = torch.zeros((), device=emb.device)
    centers: dict[int, torch.Tensor] = {}
    for class_idx in [0, 1, 2]:
        mask = y == class_idx
        if torch.any(mask):
            center = emb[mask].mean(dim=0)
            centers[class_idx] = center
            if center_weight > 0:
                center_loss = center_loss + F.mse_loss(emb[mask], center.expand_as(emb[mask]))
    if center_weight > 0 and centers:
        center_loss = center_loss / len(centers)
    if margin_weight > 0 and 1 in centers and 2 in centers:
        vt_vf_distance = torch.norm(centers[1] - centers[2], p=2)
        margin_loss = F.relu(torch.as_tensor(margin, device=emb.device) - vt_vf_distance).pow(2)
    return center_weight * center_loss, margin_weight * margin_loss


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, emb_all, y_all = [], [], []
    for batch in loader:
        if len(batch) == 4:
            xb, fb, _, yb = batch
            fb = fb.to(device)
        elif len(batch) == 3 and batch[1].ndim == 2:
            xb, fb, yb = batch
            fb = fb.to(device)
        elif len(batch) == 3:
            xb, _, yb = batch
            fb = None
        else:
            xb, yb = batch
            fb = None
        xb = xb.to(device)
        logits, emb = model(xb, fb, return_embedding=True) if fb is not None else model(xb, return_embedding=True)
        logits_all.append(logits.cpu().numpy())
        emb_all.append(emb.cpu().numpy())
        y_all.append(yb.numpy())
    return np.concatenate(logits_all), np.concatenate(emb_all), np.concatenate(y_all)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
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
        default="cnn",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument(
        "--split-grouping",
        choices=["record", "duplicate_family"],
        default="record",
        help="Use source records or exact-duplicate-connected record families as split groups.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--aux-boundary-weight", type=float, default=0.0)
    parser.add_argument("--gate-target-weight", type=float, default=0.0)
    parser.add_argument("--gate-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--gate-sr-target", type=float, default=0.25)
    parser.add_argument("--gate-ventricular-target", type=float, default=0.85)
    parser.add_argument("--risk-targets", type=Path, default=None)
    parser.add_argument("--risk-boundary-weight", type=float, default=0.0)
    parser.add_argument("--risk-gate-weight", type=float, default=0.0)
    parser.add_argument(
        "--boundary-ce-weight",
        type=float,
        default=0.0,
        help="Upweight cross-entropy for high reliability-risk samples; requires --risk-targets.",
    )
    parser.add_argument(
        "--stability-consistency-weight",
        type=float,
        default=0.0,
        help="KL consistency weight between clean and mildly perturbed ECG predictions.",
    )
    parser.add_argument(
        "--embedding-consistency-weight",
        type=float,
        default=0.0,
        help="MSE consistency weight between clean and mildly perturbed embeddings.",
    )
    parser.add_argument("--perturb-noise-std", type=float, default=0.02)
    parser.add_argument("--perturb-scale-std", type=float, default=0.03)
    parser.add_argument("--perturb-mask-prob", type=float, default=0.0)
    parser.add_argument(
        "--prototype-center-weight",
        type=float,
        default=0.0,
        help="Within-class prototype compactness loss on batch embeddings.",
    )
    parser.add_argument(
        "--prototype-margin-weight",
        type=float,
        default=0.0,
        help="VT/VF prototype separation loss on batch embeddings.",
    )
    parser.add_argument("--prototype-vtvf-margin", type=float, default=1.0)
    parser.add_argument(
        "--regularity-aux-weight",
        type=float,
        default=0.0,
        help="Auxiliary MSE loss that predicts regularity features from the fused embedding.",
    )
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_rhythm_windows(args.mat, max_windows_per_record=args.max_windows_per_record)
    split_groups = (
        dataset.record_ids
        if args.split_grouping == "record"
        else build_duplicate_family_groups(dataset.x, dataset.record_ids)
    )
    splits = make_splits(dataset.x, dataset.y, groups=split_groups, seed=args.seed)

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model}{suffix}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest"
    latest.write_text(str(run_dir), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_regularity_features = args.model in {"regularity_fusion", "reliability_gated_fusion"}
    if args.regularity_aux_weight > 0 and not use_regularity_features:
        raise ValueError("--regularity-aux-weight currently requires regularity_fusion or reliability_gated_fusion.")
    train_features = val_features = test_features = None
    train_risk = val_risk = test_risk = None
    feature_scaler = None
    if use_regularity_features:
        train_raw = extract_regularity_features_batch(splits.x_train)
        val_raw = extract_regularity_features_batch(splits.x_val)
        test_raw = extract_regularity_features_batch(splits.x_test)
        mean = train_raw.mean(axis=0)
        std = train_raw.std(axis=0) + 1e-6
        train_features = ((train_raw - mean) / std).astype(np.float32)
        val_features = ((val_raw - mean) / std).astype(np.float32)
        test_features = ((test_raw - mean) / std).astype(np.float32)
        feature_scaler = {"mean": mean.tolist(), "std": std.tolist(), "names": REGULARITY_FEATURE_NAMES}
        np.savez(run_dir / "feature_scaler.npz", mean=mean, std=std, names=np.asarray(REGULARITY_FEATURE_NAMES))
    if args.risk_targets is not None:
        risk = np.load(args.risk_targets)
        train_risk = risk["train"].astype(np.float32)
        val_risk = risk["val"].astype(np.float32)
        test_risk = risk["test"].astype(np.float32)
        if len(train_risk) != len(splits.y_train) or len(val_risk) != len(splits.y_val) or len(test_risk) != len(splits.y_test):
            raise ValueError("Risk target lengths do not match the current data split. Check seed and source run.")
        np.savez(run_dir / "risk_targets_used.npz", train=train_risk, val=val_risk, test=test_risk, source=str(args.risk_targets))
    if args.boundary_ce_weight > 0 and train_risk is None:
        raise ValueError("--boundary-ce-weight requires --risk-targets generated from the same data split.")

    model = build_model(args.model, num_classes=len(CLASS_NAMES), feature_dim=len(REGULARITY_FEATURE_NAMES)).to(device)
    regularity_aux_head = None
    if args.regularity_aux_weight > 0:
        regularity_aux_head = nn.Linear(128, len(REGULARITY_FEATURE_NAMES)).to(device)
    optim_params = list(model.parameters())
    if regularity_aux_head is not None:
        optim_params.extend(regularity_aux_head.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=1e-4)
    weights = None if args.no_class_weights else _class_weights(splits.y_train, len(CLASS_NAMES)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="none")
    use_consistency_training = args.stability_consistency_weight > 0 or args.embedding_consistency_weight > 0
    use_prototype_training = args.prototype_center_weight > 0 or args.prototype_margin_weight > 0
    use_regularity_aux = regularity_aux_head is not None
    is_gated_model = args.model == "reliability_gated_fusion"
    use_auxiliary_losses = is_gated_model and (
        args.aux_boundary_weight > 0
        or args.gate_target_weight > 0
        or args.gate_sparsity_weight > 0
        or args.risk_boundary_weight > 0
        or args.risk_gate_weight > 0
    )
    boundary_pos = float(np.mean(splits.y_train != 0))
    boundary_pos_weight = torch.tensor([(1.0 - boundary_pos) / max(boundary_pos, 1e-6)], device=device)
    boundary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=boundary_pos_weight)
    gate_loss_fn = nn.BCELoss()
    risk_loss_fn = nn.MSELoss()

    train_loader = _loader(
        splits.x_train,
        splits.y_train,
        args.batch_size,
        shuffle=True,
        features=train_features,
        risk_targets=train_risk,
    )
    val_loader = _loader(
        splits.x_val,
        splits.y_val,
        args.batch_size,
        shuffle=False,
        features=val_features,
        risk_targets=val_risk,
    )
    test_loader = _loader(
        splits.x_test,
        splits.y_test,
        args.batch_size,
        shuffle=False,
        features=test_features,
        risk_targets=test_risk,
    )

    history = []
    best_val = -1.0
    best_path = run_dir / "best_model.pt"
    split_summary = {
        "train": np.bincount(splits.y_train, minlength=len(CLASS_NAMES)).tolist(),
        "val": np.bincount(splits.y_val, minlength=len(CLASS_NAMES)).tolist(),
        "test": np.bincount(splits.y_test, minlength=len(CLASS_NAMES)).tolist(),
        "class_names": CLASS_NAMES,
        "split_grouping": args.split_grouping,
        "n_split_groups": int(np.unique(split_groups).size),
        "class_weights": None if weights is None else weights.detach().cpu().numpy().tolist(),
        "regularity_features": feature_scaler,
        "risk_targets": None if args.risk_targets is None else str(args.risk_targets),
        "reliability_guided_mitigation": {
            "boundary_ce_weight": args.boundary_ce_weight,
            "stability_consistency_weight": args.stability_consistency_weight,
            "embedding_consistency_weight": args.embedding_consistency_weight,
            "perturb_noise_std": args.perturb_noise_std,
            "perturb_scale_std": args.perturb_scale_std,
            "perturb_mask_prob": args.perturb_mask_prob,
            "prototype_center_weight": args.prototype_center_weight,
            "prototype_margin_weight": args.prototype_margin_weight,
            "prototype_vtvf_margin": args.prototype_vtvf_margin,
            "regularity_aux_weight": args.regularity_aux_weight,
        },
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(json.dumps(split_summary, indent=2))
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, ce_losses, boundary_losses, gate_losses, sparsity_losses = [], [], [], [], []
        risk_losses, consistency_losses, prototype_losses, regularity_aux_losses = [], [], [], []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            riskb = None
            if len(batch) == 4:
                xb, fb, riskb, yb = batch
                fb = fb.to(device)
                riskb = riskb.to(device)
            elif len(batch) == 3 and batch[1].ndim == 2:
                xb, fb, yb = batch
                fb = fb.to(device)
            elif len(batch) == 3:
                xb, riskb, yb = batch
                fb = None
                riskb = riskb.to(device)
            else:
                xb, yb = batch
                fb = None
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            need_embedding = use_consistency_training or use_prototype_training or use_regularity_aux
            if use_auxiliary_losses and need_embedding:
                logits, emb, gate, boundary_logit = model(xb, fb, return_embedding=True, return_gate=True)
            elif need_embedding:
                logits, emb = _forward_with_embedding(model, xb, fb)
                gate = boundary_logit = None
            elif use_auxiliary_losses:
                logits, gate, boundary_logit = model(xb, fb, return_gate=True)
                emb = None
            else:
                logits = model(xb, fb) if fb is not None else model(xb)
                emb = None
                gate = boundary_logit = None
            ce_per_sample = loss_fn(logits, yb)
            if riskb is not None and args.boundary_ce_weight > 0:
                sample_weight = 1.0 + args.boundary_ce_weight * riskb
                ce_loss = (ce_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            else:
                ce_loss = ce_per_sample.mean()
            loss = ce_loss
            boundary_loss = torch.zeros((), device=device)
            gate_loss = torch.zeros((), device=device)
            sparsity_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            prototype_loss = torch.zeros((), device=device)
            regularity_aux_loss = torch.zeros((), device=device)
            if use_auxiliary_losses:
                ventricular_target = (yb != 0).float()
                gate_target = torch.where(
                    ventricular_target > 0,
                    torch.full_like(ventricular_target, args.gate_ventricular_target),
                    torch.full_like(ventricular_target, args.gate_sr_target),
                )
                if args.aux_boundary_weight > 0:
                    boundary_loss = boundary_loss_fn(boundary_logit, ventricular_target)
                    loss = loss + args.aux_boundary_weight * boundary_loss
                if args.gate_target_weight > 0:
                    gate_loss = gate_loss_fn(gate.clamp(1e-6, 1.0 - 1e-6), gate_target)
                    loss = loss + args.gate_target_weight * gate_loss
                if args.gate_sparsity_weight > 0:
                    sparsity_loss = gate.mean()
                    loss = loss + args.gate_sparsity_weight * sparsity_loss
                if riskb is not None and args.risk_boundary_weight > 0:
                    risk_boundary_loss = risk_loss_fn(torch.sigmoid(boundary_logit), riskb)
                    loss = loss + args.risk_boundary_weight * risk_boundary_loss
                else:
                    risk_boundary_loss = torch.zeros((), device=device)
                if riskb is not None and args.risk_gate_weight > 0:
                    risk_gate_loss = risk_loss_fn(gate, riskb)
                    loss = loss + args.risk_gate_weight * risk_gate_loss
                    risk_boundary_loss = risk_boundary_loss + risk_gate_loss
            if use_consistency_training:
                xb_aug = _mild_perturbation(
                    xb,
                    noise_std=args.perturb_noise_std,
                    scale_std=args.perturb_scale_std,
                    mask_prob=args.perturb_mask_prob,
                )
                aug_logits, aug_emb = _forward_with_embedding(model, xb_aug, fb)
                if args.stability_consistency_weight > 0:
                    clean_prob = F.softmax(logits.detach(), dim=1)
                    pred_consistency = F.kl_div(F.log_softmax(aug_logits, dim=1), clean_prob, reduction="batchmean")
                    consistency_loss = consistency_loss + args.stability_consistency_weight * pred_consistency
                if args.embedding_consistency_weight > 0 and emb is not None:
                    emb_consistency = F.mse_loss(aug_emb, emb.detach())
                    consistency_loss = consistency_loss + args.embedding_consistency_weight * emb_consistency
                loss = loss + consistency_loss
            if use_prototype_training and emb is not None:
                center_loss, margin_loss = _prototype_losses(
                    emb,
                    yb,
                    center_weight=args.prototype_center_weight,
                    margin_weight=args.prototype_margin_weight,
                    margin=args.prototype_vtvf_margin,
                )
                prototype_loss = center_loss + margin_loss
                loss = loss + prototype_loss
            if use_regularity_aux and emb is not None and fb is not None:
                regularity_pred = regularity_aux_head(emb)
                regularity_aux_loss = F.mse_loss(regularity_pred, fb)
                loss = loss + args.regularity_aux_weight * regularity_aux_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            ce_losses.append(ce_loss.item())
            boundary_losses.append(boundary_loss.item())
            gate_losses.append(gate_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            risk_losses.append(risk_boundary_loss.item() if use_auxiliary_losses else 0.0)
            consistency_losses.append(consistency_loss.item())
            prototype_losses.append(prototype_loss.item())
            regularity_aux_losses.append(regularity_aux_loss.item())

        val_logits, _, val_y = predict(model, val_loader, device)
        val_probs = softmax(val_logits)
        val_metrics = classification_metrics(val_y, val_probs)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_ce_loss": float(np.mean(ce_losses)),
            "train_boundary_loss": float(np.mean(boundary_losses)),
            "train_gate_loss": float(np.mean(gate_losses)),
            "train_gate_sparsity": float(np.mean(sparsity_losses)),
            "train_risk_loss": float(np.mean(risk_losses)),
            "train_consistency_loss": float(np.mean(consistency_losses)),
            "train_prototype_loss": float(np.mean(prototype_losses)),
            "train_regularity_aux_loss": float(np.mean(regularity_aux_losses)),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_ece": expected_calibration_error(val_y, val_probs),
        }
        history.append(row)
        print(row)
        if row["val_macro_f1"] > best_val:
            best_val = row["val_macro_f1"]
            safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            payload = {"model": model.state_dict(), "args": safe_args}
            if regularity_aux_head is not None:
                payload["regularity_aux_head"] = regularity_aux_head.state_dict()
            torch.save(payload, best_path)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])

    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        logits, emb, yy = predict(model, loader, device)
        np.savez(run_dir / f"embeddings_{split_name}.npz", logits=logits, embeddings=emb, y=yy)

    test_logits, _, test_y = predict(model, test_loader, device)
    test_probs = softmax(test_logits)
    metrics = classification_metrics(test_y, test_probs)
    metrics["ece"] = expected_calibration_error(test_y, test_probs)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(test_probs, columns=[f"prob_{c}" for c in CLASS_NAMES])
    pred_df["y_true"] = test_y
    pred_df["y_pred"] = test_probs.argmax(axis=1)
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



