from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
from .evaluate_corruption_severity import _corrupt
from .metrics import softmax
from .models import build_model
from .train import predict


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)


def _checkpoint_args(run_dir: Path) -> dict:
    return _checkpoint(run_dir).get("args", {})


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["embeddings"].astype(np.float32), data["logits"].astype(np.float32), data["y"].astype(np.int64)


def _hash_windows(x: np.ndarray) -> np.ndarray:
    return np.asarray([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x.reshape(len(x), -1)])


def _groups(dataset, split_grouping: str) -> np.ndarray:
    if split_grouping == "duplicate_family":
        return build_duplicate_family_groups(dataset.x, dataset.record_ids)
    return dataset.record_ids


def _split_data(mat: Path, seed: int, split_grouping: str):
    dataset = load_rhythm_windows(mat)
    return make_splits(dataset.x, dataset.y, groups=_groups(dataset, split_grouping), seed=seed)


def _regularity_features(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_raw = extract_regularity_features_batch(train_x)
    test_raw = extract_regularity_features_batch(test_x)
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-6
    return ((train_raw - mean) / std).astype(np.float32), ((test_raw - mean) / std).astype(np.float32)


def _vtvf_cross_errors(y: np.ndarray, pred: np.ndarray) -> int:
    return int(np.sum(((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))))


def _linear_probe_rows(
    train_reps: dict[str, np.ndarray],
    test_reps: dict[str, np.ndarray],
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for name, x_train in train_reps.items():
        x_test = test_reps[name]
        multi = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
        multi.fit(x_train, train_y)
        pred = multi.predict(x_test)
        rows.append(
            {
                "representation": name,
                "probe": "sr_vt_vf_multiclass",
                "accuracy": float(accuracy_score(test_y, pred)),
                "macro_f1": float(f1_score(test_y, pred, average="macro")),
                "vtvf_cross_errors": _vtvf_cross_errors(test_y, pred),
            }
        )

        train_mask = np.isin(train_y, [1, 2])
        test_mask = np.isin(test_y, [1, 2])
        if train_mask.sum() > 5 and test_mask.sum() > 5:
            y_train_bin = (train_y[train_mask] == 2).astype(int)
            y_test_bin = (test_y[test_mask] == 2).astype(int)
            binary = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
            binary.fit(x_train[train_mask], y_train_bin)
            pred_bin = binary.predict(x_test[test_mask])
            score_bin = binary.predict_proba(x_test[test_mask])[:, 1]
            rows.append(
                {
                    "representation": name,
                    "probe": "vt_vs_vf_binary",
                    "accuracy": float(accuracy_score(y_test_bin, pred_bin)),
                    "macro_f1": float(f1_score(y_test_bin, pred_bin, average="macro")),
                    "vtvf_cross_errors": np.nan,
                    "auroc": float(roc_auc_score(y_test_bin, score_bin)) if len(np.unique(y_test_bin)) > 1 else np.nan,
                }
            )
    return rows


def _regularized_cov(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    return cov + np.eye(cov.shape[0]) * max(eps * scale, eps)


def _effective_rank(cov: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 1e-12, None)
    p = vals / vals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _distribution_rows(reps: dict[str, np.ndarray], y: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for rep_name, rep in reps.items():
        means = {c: rep[y == c].mean(axis=0) for c in range(3)}
        covs = {c: _regularized_cov(rep[y == c]) for c in range(3)}
        for c, label in enumerate(CLASS_NAMES):
            vals = np.linalg.eigvalsh(covs[c])
            rows.append(
                {
                    "representation": rep_name,
                    "scope": f"class_{label}",
                    "effective_rank": _effective_rank(covs[c]),
                    "logdet_cov": float(np.linalg.slogdet(covs[c])[1]),
                    "condition_number": float(vals.max() / max(vals.min(), 1e-12)),
                    "mean_radius": float(np.linalg.norm(rep[y == c] - means[c], axis=1).mean()),
                }
            )

        for i, j in [(0, 1), (0, 2), (1, 2)]:
            pooled = (covs[i] + covs[j]) / 2.0
            delta = means[i] - means[j]
            inv = np.linalg.pinv(pooled)
            maha = float(np.sqrt(max(delta @ inv @ delta, 0.0)))
            euclid = float(np.linalg.norm(delta))
            within = float((np.trace(covs[i]) + np.trace(covs[j])) / (2.0 * rep.shape[1]))
            fisher = float((euclid**2) / max(within, 1e-12))
            rows.append(
                {
                    "representation": rep_name,
                    "scope": f"{CLASS_NAMES[i]}_vs_{CLASS_NAMES[j]}",
                    "euclidean_centroid_distance": euclid,
                    "mahalanobis_centroid_distance": maha,
                    "fisher_ratio": fisher,
                    "pooled_logdet_cov": float(np.linalg.slogdet(pooled)[1]),
                }
            )
    return rows


def _nearest_proto(train_rep: np.ndarray, train_y: np.ndarray, rep: np.ndarray) -> np.ndarray:
    centroids = np.stack([train_rep[train_y == c].mean(axis=0) for c in range(3)])
    return np.linalg.norm(rep[:, None, :] - centroids[None, :, :], axis=2).argmin(axis=1)


def _loader(x: np.ndarray, features: np.ndarray | None, batch_size: int) -> DataLoader:
    y = np.zeros(len(x), dtype=np.int64)
    if features is None:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(features), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _features_for_run(run_dir: Path, model_name: str, x: np.ndarray) -> np.ndarray | None:
    if model_name not in {"regularity_fusion", "reliability_gated_fusion"}:
        return None
    scaler_path = run_dir / "feature_scaler.npz"
    if not scaler_path.exists():
        return None
    scaler = np.load(scaler_path)
    return ((extract_regularity_features_batch(x) - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _perturbation_rows(
    mat: Path,
    run_dir: Path,
    model_name: str,
    seed: int,
    split_grouping: str,
    train_emb: np.ndarray,
    train_y: np.ndarray,
    batch_size: int,
) -> list[dict]:
    splits = _split_data(mat, seed, split_grouping)
    ckpt = _checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=len(CLASS_NAMES), feature_dim=len(REGULARITY_FEATURE_NAMES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = splits.x_test.astype(np.float32)
    base_features = _features_for_run(run_dir, model_name, x)
    base_logits, base_emb, _ = predict(model, _loader(x, base_features, batch_size), device)
    base_probs = softmax(base_logits)
    base_pred = base_probs.argmax(axis=1)
    base_proto = _nearest_proto(train_emb, train_y, base_emb)
    base_vtvf_margin = base_logits[:, 1] - base_logits[:, 2]
    nn = NearestNeighbors(n_neighbors=15).fit(train_emb)
    base_neighbors = nn.kneighbors(base_emb, return_distance=False)

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for kind in ["gaussian_noise", "baseline_wander", "random_masking", "amplitude_scaling", "mixed_noise_baseline"]:
        for severity in [1, 2, 3]:
            corrupt_x = _corrupt(x, kind, severity, rng).astype(np.float32)
            features = _features_for_run(run_dir, model_name, corrupt_x)
            logits, emb, _ = predict(model, _loader(corrupt_x, features, batch_size), device)
            pred = softmax(logits).argmax(axis=1)
            proto = _nearest_proto(train_emb, train_y, emb)
            neighbors = nn.kneighbors(emb, return_distance=False)
            inter = np.asarray([len(set(a).intersection(set(b))) for a, b in zip(base_neighbors, neighbors)], dtype=float)
            union = np.asarray([len(set(a).union(set(b))) for a, b in zip(base_neighbors, neighbors)], dtype=float)
            shift = np.linalg.norm(emb - base_emb, axis=1)
            cosine = np.sum(emb * base_emb, axis=1) / (
                np.linalg.norm(emb, axis=1) * np.linalg.norm(base_emb, axis=1) + 1e-12
            )
            rows.append(
                {
                    "corruption": kind,
                    "severity": severity,
                    "embedding_shift_mean": float(shift.mean()),
                    "embedding_shift_vtvf_mean": float(shift[np.isin(splits.y_test, [1, 2])].mean()),
                    "cosine_preservation_mean": float(cosine.mean()),
                    "prediction_flip_rate": float((pred != base_pred).mean()),
                    "prototype_flip_rate": float((proto != base_proto).mean()),
                    "neighbor_jaccard_mean": float((inter / np.maximum(union, 1.0)).mean()),
                    "vtvf_margin_abs_change_mean": float(np.abs((logits[:, 1] - logits[:, 2]) - base_vtvf_margin).mean()),
                }
            )
    return rows


def _concept_rows(
    train_reps: dict[str, np.ndarray],
    test_reps: dict[str, np.ndarray],
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for rep_name, train_rep in train_reps.items():
        test_rep = test_reps[rep_name]
        for idx, feature_name in enumerate(REGULARITY_FEATURE_NAMES):
            target = test_features[:, idx]
            corrs = []
            for dim in range(test_rep.shape[1]):
                if np.std(test_rep[:, dim]) < 1e-8:
                    continue
                corrs.append(abs(pearsonr(test_rep[:, dim], target).statistic))
            ridge = Ridge(alpha=10.0)
            ridge.fit(train_rep, train_features[:, idx])
            pred = ridge.predict(test_rep)
            ss_res = float(np.sum((test_features[:, idx] - pred) ** 2))
            ss_tot = float(np.sum((test_features[:, idx] - test_features[:, idx].mean()) ** 2))
            rows.append(
                {
                    "representation": rep_name,
                    "regularity_feature": feature_name,
                    "max_abs_dim_correlation": float(max(corrs) if corrs else np.nan),
                    "ridge_r2_from_representation": float(1.0 - ss_res / max(ss_tot, 1e-12)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced representation diagnostics for one trained ECG run.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-perturbation", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ckpt_args = _checkpoint_args(run_dir)
    model_name = (args.model or str(ckpt_args.get("model", "cnn"))).lower()
    seed = int(ckpt_args.get("seed", 42))
    split_grouping = str(ckpt_args.get("split_grouping", "record"))

    train_emb, train_logits, train_y = _load_split(run_dir, "train")
    test_emb, test_logits, test_y = _load_split(run_dir, "test")
    splits = _split_data(args.mat, seed, split_grouping)
    if not np.array_equal(test_y, splits.y_test):
        raise RuntimeError("Test labels do not match checkpoint split; aborting advanced diagnostics.")

    train_reg, test_reg = _regularity_features(splits.x_train, splits.x_test)
    train_reps = {
        "final_embedding": train_emb,
        "classifier_logits": train_logits,
        "regularity_features": train_reg,
    }
    test_reps = {
        "final_embedding": test_emb,
        "classifier_logits": test_logits,
        "regularity_features": test_reg,
    }

    linear = pd.DataFrame(_linear_probe_rows(train_reps, test_reps, train_y, test_y))
    distribution = pd.DataFrame(_distribution_rows(test_reps, test_y))
    concept = pd.DataFrame(_concept_rows(train_reps, test_reps, train_reg, test_reg))

    linear.to_csv(run_dir / "advanced_linear_probe_summary.csv", index=False)
    distribution.to_csv(run_dir / "advanced_distribution_geometry.csv", index=False)
    concept.to_csv(run_dir / "advanced_regularity_concept_alignment.csv", index=False)

    perturbation = pd.DataFrame()
    if not args.skip_perturbation:
        perturbation = pd.DataFrame(
            _perturbation_rows(
                args.mat,
                run_dir,
                model_name,
                seed,
                split_grouping,
                train_emb,
                train_y,
                args.batch_size,
            )
        )
        perturbation.to_csv(run_dir / "advanced_perturbation_representation_stability.csv", index=False)

    np.savez_compressed(
        run_dir / "advanced_test_representations.npz",
        sample_hash=_hash_windows(splits.x_test),
        y=test_y,
        final_embedding=test_emb,
        classifier_logits=test_logits,
        regularity_features=test_reg,
    )

    summary = {
        "run_dir": str(run_dir),
        "model": model_name,
        "seed": seed,
        "split_grouping": split_grouping,
        "linear_probe_rows": int(len(linear)),
        "distribution_rows": int(len(distribution)),
        "concept_rows": int(len(concept)),
        "perturbation_rows": int(len(perturbation)),
    }
    (run_dir / "advanced_representation_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
