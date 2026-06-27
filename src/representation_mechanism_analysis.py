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

from .data import (
    CLASS_NAMES,
    REGULARITY_FEATURE_NAMES,
    build_duplicate_family_groups,
    extract_regularity_features_batch,
    load_rhythm_windows,
    make_splits,
)
from .evaluate_corruption_severity import _corrupt
from .metrics import softmax
from .models import CNNLSTMClassifier, ECGCNN, ReliabilityGatedRegularityFusion, RegularityFusionResNet1D, build_model


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)


def _split_groups(dataset, split_grouping: str) -> np.ndarray:
    if split_grouping == "duplicate_family":
        return build_duplicate_family_groups(dataset.x, dataset.record_ids)
    return dataset.record_ids


def _features_for_run(run_dir: Path, model_name: str, x: np.ndarray) -> np.ndarray | None:
    if model_name not in {"regularity_fusion", "reliability_gated_fusion"}:
        return None
    scaler_path = run_dir / "feature_scaler.npz"
    if not scaler_path.exists():
        return None
    scaler = np.load(scaler_path)
    return ((extract_regularity_features_batch(x) - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _loader(x: np.ndarray, y: np.ndarray, features: np.ndarray | None, batch_size: int) -> DataLoader:
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _pool_time(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 3:
        return torch.cat([z.mean(dim=-1), z.std(dim=-1)], dim=1)
    return z


@torch.no_grad()
def _extract_layers(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    layers: dict[str, list[np.ndarray]] = {}
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    def add(name: str, value: torch.Tensor) -> None:
        layers.setdefault(name, []).append(value.detach().cpu().numpy())

    for batch in loader:
        if len(batch) == 3:
            xb, fb, yb = batch
            xb, fb = xb.to(device), fb.to(device)
        else:
            xb, yb = batch
            xb, fb = xb.to(device), None

        if isinstance(model, ECGCNN):
            z = model.encoder[0](xb)
            add("conv1", _pool_time(z))
            z = model.encoder[1](z)
            add("pool1", _pool_time(z))
            z = model.encoder[2](z)
            add("conv2", _pool_time(z))
            z = model.encoder[3](z)
            add("pool2", _pool_time(z))
            z = model.encoder[4](z)
            add("conv3", _pool_time(z))
            pooled = model.encoder[5](z).squeeze(-1)
            add("pre_embedding_pool", pooled)
            emb = torch.relu(model.embedding(pooled))
            logits = model.classifier(emb)
            add("final_embedding", emb)
            add("classifier_logits", logits)
        elif isinstance(model, CNNLSTMClassifier):
            z = model.cnn[0](xb)
            add("cnn_conv1", _pool_time(z))
            z = model.cnn[1](z)
            add("cnn_pool1", _pool_time(z))
            z = model.cnn[2](z)
            add("cnn_conv2", _pool_time(z))
            z = model.cnn[3](z)
            add("cnn_pool2", _pool_time(z))
            z = model.cnn[4](z)
            add("cnn_conv3", _pool_time(z))
            add("cnn_sequence", _pool_time(z))
            _, (h, _) = model.lstm(z.transpose(1, 2))
            state = torch.cat([h[-2], h[-1]], dim=1)
            add("lstm_last_state", state)
            emb = model.embedding(state)
            logits = model.classifier(emb)
            add("final_embedding", emb)
            add("classifier_logits", logits)
        elif isinstance(model, ReliabilityGatedRegularityFusion):
            if fb is None:
                fb = torch.zeros((xb.shape[0], model.feature_dim), device=device, dtype=xb.dtype)
            _, wave_emb = model.waveform(xb, return_embedding=True)
            feature_emb = model.feature_encoder(fb)
            gate = model.reliability_gate(torch.cat([wave_emb, feature_emb], dim=1))
            fused = model.norm(wave_emb + gate * feature_emb)
            logits = model.classifier(fused)
            add("waveform_embedding", wave_emb)
            add("regularity_feature_embedding", feature_emb)
            add("gate", gate)
            add("fused_embedding", fused)
            add("classifier_logits", logits)
        elif isinstance(model, RegularityFusionResNet1D):
            if fb is None:
                fb = torch.zeros((xb.shape[0], model.feature_dim), device=device, dtype=xb.dtype)
            _, wave_emb = model.waveform(xb, return_embedding=True)
            feature_emb = model.feature_encoder(fb)
            emb = torch.relu(model.fused_embedding(torch.cat([wave_emb, feature_emb], dim=1)))
            logits = model.classifier(emb)
            add("waveform_embedding", wave_emb)
            add("regularity_feature_embedding", feature_emb)
            add("fused_embedding", emb)
            add("classifier_logits", logits)
        else:
            logits, emb = model(xb, return_embedding=True)
            add("final_embedding", emb)
            add("classifier_logits", logits)

        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(yb.numpy())

    return {name: np.concatenate(values, axis=0) for name, values in layers.items()}, np.concatenate(logits_all), np.concatenate(y_all)


def _vtvf_cross(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))


def _probe_rows(train_layers: dict[str, np.ndarray], train_y: np.ndarray, test_layers: dict[str, np.ndarray], test_y: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for layer, x_train in train_layers.items():
        x_test = test_layers[layer]
        if x_train.ndim != 2 or x_train.shape[1] < 2:
            continue
        multi = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
        multi.fit(x_train, train_y)
        pred = multi.predict(x_test)
        rows.append(
            {
                "layer": layer,
                "probe": "sr_vt_vf_multiclass",
                "dim": int(x_train.shape[1]),
                "accuracy": float(accuracy_score(test_y, pred)),
                "macro_f1": float(f1_score(test_y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
                "vtvf_cross_errors": int(_vtvf_cross(test_y, pred).sum()),
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
                    "macro_f1": float(f1_score(y_test_bin, pred_bin, average="macro", zero_division=0)),
                    "vtvf_cross_errors": np.nan,
                    "auroc": float(roc_auc_score(y_test_bin, score_bin)) if len(np.unique(y_test_bin)) > 1 else np.nan,
                }
            )
    return rows


def _stability_rows(clean_layers: dict[str, np.ndarray], pert_layers: dict[str, np.ndarray], y: np.ndarray, pred: np.ndarray, pert_pred: np.ndarray, corruption: str, severity: int) -> list[dict]:
    rows = []
    error = pred != y
    vtvf = np.isin(y, [1, 2])
    cross = _vtvf_cross(y, pred)
    flip = pert_pred != pred
    for layer, clean in clean_layers.items():
        pert = pert_layers[layer]
        if clean.ndim != 2 or clean.shape != pert.shape:
            continue
        shift = np.linalg.norm(pert - clean, axis=1)
        cosine = np.sum(pert * clean, axis=1) / (np.linalg.norm(pert, axis=1) * np.linalg.norm(clean, axis=1) + 1e-12)
        row = {
            "corruption": corruption,
            "severity": severity,
            "layer": layer,
            "embedding_shift_mean": float(shift.mean()),
            "embedding_shift_correct_mean": float(shift[~error].mean()) if (~error).any() else np.nan,
            "embedding_shift_error_mean": float(shift[error].mean()) if error.any() else np.nan,
            "embedding_shift_vtvf_mean": float(shift[vtvf].mean()) if vtvf.any() else np.nan,
            "embedding_shift_vtvf_cross_error_mean": float(shift[cross].mean()) if cross.any() else np.nan,
            "cosine_preservation_mean": float(cosine.mean()),
            "prediction_flip_rate": float(flip.mean()),
            "prediction_flip_error_rate": float(flip[error].mean()) if error.any() else np.nan,
            "prediction_flip_vtvf_cross_error_rate": float(flip[cross].mean()) if cross.any() else np.nan,
        }
        rows.append(row)
    return rows


def _stable_error_summary(
    y: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    pert_preds: list[np.ndarray],
    final_clean: np.ndarray,
    final_pert: list[np.ndarray],
    confidence_threshold: float,
    flip_threshold: float,
) -> pd.DataFrame:
    confidence = probs.max(axis=1)
    error = pred != y
    cross = _vtvf_cross(y, pred)
    if pert_preds:
        flip_rate = np.stack([item != pred for item in pert_preds], axis=1).mean(axis=1)
    else:
        flip_rate = np.zeros(len(y))
    if final_pert:
        shifts = np.stack([np.linalg.norm(item - final_clean, axis=1) for item in final_pert], axis=1)
        mean_shift = shifts.mean(axis=1)
    else:
        mean_shift = np.zeros(len(y))
    stable = flip_rate <= flip_threshold
    confident = confidence >= confidence_threshold
    rows = []
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "any_error": error,
        "vtvf_cross_error": cross,
        "confident_error": error & confident,
        "stable_error": error & stable,
        "confident_stable_error": error & confident & stable,
        "confident_stable_vtvf_cross_error": cross & confident & stable,
    }
    for name, mask in groups.items():
        sub_n = int(mask.sum())
        rows.append(
            {
                "group": name,
                "n": sub_n,
                "fraction": float(sub_n / len(y)),
                "mean_confidence": float(confidence[mask].mean()) if sub_n else np.nan,
                "mean_flip_rate": float(flip_rate[mask].mean()) if sub_n else np.nan,
                "mean_final_embedding_shift": float(mean_shift[mask].mean()) if sub_n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism analysis for layer-wise stability, readability, and stable errors.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["cnn", "cnn_lstm", "regularity_fusion", "reliability_gated_fusion"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--flip-threshold", type=float, default=0.10)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ckpt = _checkpoint(run_dir)
    ckpt_args = ckpt.get("args", {})
    seed = int(ckpt_args.get("seed", 42))
    split_grouping = str(ckpt_args.get("split_grouping", "record"))

    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=_split_groups(dataset, split_grouping), seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(CLASS_NAMES), feature_dim=len(REGULARITY_FEATURE_NAMES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_features = _features_for_run(run_dir, args.model, splits.x_train)
    test_features = _features_for_run(run_dir, args.model, splits.x_test)
    train_layers, _, train_y = _extract_layers(model, _loader(splits.x_train, splits.y_train, train_features, args.batch_size), device)
    clean_layers, clean_logits, test_y = _extract_layers(model, _loader(splits.x_test, splits.y_test, test_features, args.batch_size), device)
    if not np.array_equal(test_y, splits.y_test):
        raise RuntimeError("Test split mismatch during mechanism analysis.")

    probs = softmax(clean_logits)
    pred = probs.argmax(axis=1)
    readability = pd.DataFrame(_probe_rows(train_layers, train_y, clean_layers, test_y))
    readability.to_csv(run_dir / "mechanism_layer_readability.csv", index=False)

    rng = np.random.default_rng(seed)
    stability_rows = []
    pert_preds = []
    final_pert = []
    corruptions = ["gaussian_noise", "baseline_wander", "random_masking", "amplitude_scaling", "mixed_noise_baseline"]
    severities = [1, 2, 3]
    final_layer = "final_embedding" if "final_embedding" in clean_layers else "fused_embedding"
    for corruption in corruptions:
        for severity in severities:
            x_pert = _corrupt(splits.x_test.astype(np.float32), corruption, severity, rng).astype(np.float32)
            features = _features_for_run(run_dir, args.model, x_pert)
            pert_layers, pert_logits, _ = _extract_layers(model, _loader(x_pert, splits.y_test, features, args.batch_size), device)
            pert_pred = softmax(pert_logits).argmax(axis=1)
            pert_preds.append(pert_pred)
            if final_layer in pert_layers:
                final_pert.append(pert_layers[final_layer])
            stability_rows.extend(_stability_rows(clean_layers, pert_layers, test_y, pred, pert_pred, corruption, severity))

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(run_dir / "mechanism_layer_perturbation_stability.csv", index=False)
    stable_errors = _stable_error_summary(
        test_y,
        probs,
        pred,
        pert_preds,
        clean_layers[final_layer],
        final_pert,
        args.confidence_threshold,
        args.flip_threshold,
    )
    stable_errors.to_csv(run_dir / "mechanism_stable_confident_errors.csv", index=False)

    tradeoff = (
        stability.groupby("layer", as_index=False)
        .agg(
            embedding_shift_mean=("embedding_shift_mean", "mean"),
            prediction_flip_rate=("prediction_flip_rate", "mean"),
            cosine_preservation_mean=("cosine_preservation_mean", "mean"),
        )
        .merge(
            readability[readability["probe"].eq("vt_vs_vf_binary")][["layer", "auroc"]].rename(columns={"auroc": "vtvf_probe_auroc"}),
            on="layer",
            how="left",
        )
        .merge(
            readability[readability["probe"].eq("sr_vt_vf_multiclass")][["layer", "macro_f1", "vtvf_cross_errors"]].rename(
                columns={"macro_f1": "multiclass_probe_macro_f1", "vtvf_cross_errors": "probe_vtvf_cross_errors"}
            ),
            on="layer",
            how="left",
        )
    )
    tradeoff.to_csv(run_dir / "mechanism_stability_readability_tradeoff.csv", index=False)

    report = {
        "run_dir": str(run_dir),
        "model": args.model,
        "seed": seed,
        "split_grouping": split_grouping,
        "final_layer_for_stable_errors": final_layer,
        "n_layers": len(clean_layers),
        "n_perturbation_conditions": len(corruptions) * len(severities),
        "outputs": [
            "mechanism_layer_readability.csv",
            "mechanism_layer_perturbation_stability.csv",
            "mechanism_stable_confident_errors.csv",
            "mechanism_stability_readability_tradeoff.csv",
        ],
    }
    (run_dir / "representation_mechanism_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
