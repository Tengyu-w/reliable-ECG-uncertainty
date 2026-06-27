from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    CLASS_NAMES,
    REGULARITY_FEATURE_NAMES,
    build_duplicate_family_groups,
    extract_regularity_features_batch,
    load_rhythm_windows,
    make_splits,
)
from .metrics import softmax
from .models import ReliabilityGatedRegularityFusion, build_model


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _checkpoint_args(run_dir: Path) -> dict:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    return ckpt.get("args", {})


def _load_model(run_dir: Path, model_name: str) -> torch.nn.Module:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = build_model(model_name, num_classes=len(CLASS_NAMES), feature_dim=len(REGULARITY_FEATURE_NAMES))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _features_for_run(run_dir: Path, x: np.ndarray) -> np.ndarray | None:
    scaler_path = run_dir / "feature_scaler.npz"
    if not scaler_path.exists():
        return None
    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    return ((extract_regularity_features_batch(x) - mean) / std).astype(np.float32)


def _split_groups(dataset, split_grouping: str) -> np.ndarray:
    if split_grouping == "duplicate_family":
        return build_duplicate_family_groups(dataset.x, dataset.record_ids)
    return dataset.record_ids


def _loader(x: np.ndarray, features: np.ndarray | None, y: np.ndarray, batch_size: int) -> DataLoader:
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _extract_reliability_gated_layers(
    model: ReliabilityGatedRegularityFusion,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    layers: dict[str, list[np.ndarray]] = {
        "waveform_embedding": [],
        "regularity_feature_embedding": [],
        "fused_embedding": [],
        "classifier_logits": [],
        "gate": [],
    }
    y_all: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            xb, fb, yb = batch
            fb = fb.to(device)
        else:
            xb, yb = batch
            fb = torch.zeros((xb.shape[0], model.feature_dim), dtype=xb.dtype, device=device)
        xb = xb.to(device)
        _, wave_emb = model.waveform(xb, return_embedding=True)
        feature_emb = model.feature_encoder(fb)
        gate = model.reliability_gate(torch.cat([wave_emb, feature_emb], dim=1))
        fused = model.norm(wave_emb + gate * feature_emb)
        logits = model.classifier(fused)
        layers["waveform_embedding"].append(wave_emb.cpu().numpy())
        layers["regularity_feature_embedding"].append(feature_emb.cpu().numpy())
        layers["fused_embedding"].append(fused.cpu().numpy())
        layers["classifier_logits"].append(logits.cpu().numpy())
        layers["gate"].append(gate.cpu().numpy())
        y_all.append(yb.numpy())
    layer_arrays = {name: np.concatenate(values, axis=0) for name, values in layers.items()}
    logits = layer_arrays["classifier_logits"]
    y = np.concatenate(y_all, axis=0)
    return layer_arrays, logits, y


@torch.no_grad()
def _extract_generic_layers(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    emb_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            xb, fb, yb = batch
            xb, fb = xb.to(device), fb.to(device)
            logits, emb = model(xb, fb, return_embedding=True)
        else:
            xb, yb = batch
            xb = xb.to(device)
            logits, emb = model(xb, return_embedding=True)
        emb_all.append(emb.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        y_all.append(yb.numpy())
    logits = np.concatenate(logits_all, axis=0)
    return {"final_embedding": np.concatenate(emb_all, axis=0), "classifier_logits": logits}, logits, np.concatenate(y_all, axis=0)


def _norm_dist(train_rep: np.ndarray, train_y: np.ndarray, i: int, j: int) -> float:
    centroids = np.stack([train_rep[train_y == c].mean(axis=0) for c in range(3)])
    within = [np.linalg.norm(train_rep[train_y == c] - centroids[c], axis=1).mean() for c in range(3)]
    return float(np.linalg.norm(centroids[i] - centroids[j]) / ((within[i] + within[j]) / 2.0))


def _neighbor_summary(train_rep: np.ndarray, train_y: np.ndarray, test_rep: np.ndarray, test_y: np.ndarray, k: int) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=k).fit(train_rep)
    _, idx = nn.kneighbors(test_rep)
    neigh_y = train_y[idx]
    purity = (neigh_y == test_y[:, None]).mean(axis=1)
    ventricular_neighbors = np.isin(neigh_y, [1, 2])
    opposite = np.where(test_y == 1, 2, 1)
    opposite_vtvf = neigh_y == opposite[:, None]
    denom = ventricular_neighbors.sum(axis=1)
    ventricular = np.isin(test_y, [1, 2])
    mixing = np.zeros(len(test_y), dtype=np.float32)
    valid = ventricular & (denom > 0)
    mixing[valid] = opposite_vtvf[valid].sum(axis=1) / denom[valid]
    return {
        "local_purity_mean": float(purity.mean()),
        "vtvf_mixing_ventricular": float(mixing[ventricular].mean()) if ventricular.any() else np.nan,
    }


def _boundary_scores(train_rep: np.ndarray, train_y: np.ndarray, test_rep: np.ndarray, test_y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    centroids = np.stack([train_rep[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(test_rep[:, None, :] - centroids[None, :, :], axis=2)
    d_vt, d_vf = dist[:, 1], dist[:, 2]
    ambiguity = 1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12)
    vtvf_error = ((test_y == 1) & (pred == 2)) | ((test_y == 2) & (pred == 1))
    correct_v = np.isin(test_y, [1, 2]) & (pred == test_y)
    return {
        "prototype_vtvf_ambiguity_correct_vtvf": float(ambiguity[correct_v].mean()) if correct_v.any() else np.nan,
        "prototype_vtvf_ambiguity_vtvf_error": float(ambiguity[vtvf_error].mean()) if vtvf_error.any() else np.nan,
    }


def _safe_silhouette(rep: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(silhouette_score(rep, y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ckpt_args = _checkpoint_args(run_dir)
    model_name = args.model or str(ckpt_args.get("model", "reliability_gated_fusion"))
    seed = int(ckpt_args.get("seed", 42))
    split_grouping = str(ckpt_args.get("split_grouping", "record"))

    dataset = load_rhythm_windows(args.mat)
    groups = _split_groups(dataset, split_grouping)
    splits = make_splits(dataset.x, dataset.y, groups=groups, seed=seed)

    train_features = _features_for_run(run_dir, splits.x_train)
    test_features = _features_for_run(run_dir, splits.x_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run_dir, model_name).to(device)
    train_loader = _loader(splits.x_train, train_features, splits.y_train, args.batch_size)
    test_loader = _loader(splits.x_test, test_features, splits.y_test, args.batch_size)

    if isinstance(model, ReliabilityGatedRegularityFusion):
        train_layers, _, train_y = _extract_reliability_gated_layers(model, train_loader, device)
        test_layers, test_logits, test_y = _extract_reliability_gated_layers(model, test_loader, device)
    else:
        train_layers, _, train_y = _extract_generic_layers(model, train_loader, device)
        test_layers, test_logits, test_y = _extract_generic_layers(model, test_loader, device)

    pred = softmax(test_logits).argmax(axis=1)
    if len(test_y) != len(splits.y_test) or not np.array_equal(test_y, splits.y_test):
        raise RuntimeError("Layerwise extraction split mismatch; check split_summary.json and checkpoint args.")

    rows = []
    for name, test_rep in test_layers.items():
        train_rep = train_layers[name]
        row = {
            "layer": name,
            "dim": int(test_rep.shape[1]) if test_rep.ndim == 2 else int(np.prod(test_rep.shape[1:])),
            "silhouette_test": _safe_silhouette(test_rep, test_y),
            "sr_vt_norm_dist_train": _norm_dist(train_rep, train_y, 0, 1),
            "sr_vf_norm_dist_train": _norm_dist(train_rep, train_y, 0, 2),
            "vt_vf_norm_dist_train": _norm_dist(train_rep, train_y, 1, 2),
            "sr_vt_norm_dist_test": _norm_dist(test_rep, test_y, 0, 1),
            "sr_vf_norm_dist_test": _norm_dist(test_rep, test_y, 0, 2),
            "vt_vf_norm_dist_test": _norm_dist(test_rep, test_y, 1, 2),
            **_neighbor_summary(train_rep, train_y, test_rep, test_y, args.k),
            **_boundary_scores(train_rep, train_y, test_rep, test_y, pred),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "layerwise_representation_summary.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(summary))
    plt.plot(x, summary["sr_vt_norm_dist_train"], marker="o", label="SR-VT")
    plt.plot(x, summary["sr_vf_norm_dist_train"], marker="o", label="SR-VF")
    plt.plot(x, summary["vt_vf_norm_dist_train"], marker="o", label="VT-VF")
    plt.plot(x, summary["vt_vf_norm_dist_test"], marker="x", linestyle="--", label="VT-VF test")
    plt.xticks(x, summary["layer"], rotation=25, ha="right")
    plt.ylabel("Normalised centroid distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "layerwise_centroid_distances.png", dpi=180)
    plt.close()

    report = {
        "run_dir": str(run_dir),
        "model": model_name,
        "seed": seed,
        "split_grouping": split_grouping,
        "k": args.k,
        "layers": summary.to_dict(orient="records"),
    }
    (run_dir / "layerwise_representation_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
