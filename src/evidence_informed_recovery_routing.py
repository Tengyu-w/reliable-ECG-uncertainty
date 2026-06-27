from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    REGULARITY_FEATURE_NAMES,
    build_duplicate_family_groups,
    extract_regularity_features_batch,
    load_rhythm_windows,
    make_splits,
)
from .metrics import expected_calibration_error, softmax
from .uncertainty import fit_temperature


CLASS_LABELS = {0: "SR", 1: "VT", 2: "VF"}
TARGET_COLUMNS = {
    "sample_id",
    "split",
    "y_true",
    "y_pred",
    "is_error",
    "is_vtvf_cross_error",
    "is_vtvf_truth",
    "is_vtvf_candidate",
    "is_sr_ventricular_error",
    "is_representation_conflict_error",
    "is_atypical_signal_error",
    "is_hidden_confident_error",
    "action",
    "output_set",
    "any_error_risk",
    "vtvf_boundary_risk",
    "mechanism_action",
    "mechanism_route",
    "mechanism_output_set",
    "mechanism_budget",
}
LEAKAGE_OR_ID_COLUMNS = {
    "index",
    "sample_id",
    "y_true",
    "true_label",
    "label",
    "y_pred",
    "prediction",
    "pred",
    "is_error",
    "is_any_error",
    "any_error",
    "contains_true",
    "is_vtvf_cross_error",
    "is_vtvf_boundary_error",
    "vtvf_error",
    "ventricular_error",
    "confident_stable_error",
}
TEXT_COLUMNS_TO_SKIP = {
    "set",
    "supervisor_reason",
    "embodied_ai_analogue",
}

SPLIT_DIAGNOSTIC_FILES = {
    "risk_target": "risk_target_components_{split}.csv",
    "prior_calibration": "vtvf_decision_calibration_scores_{split}.csv",
}

TEST_DIAGNOSTIC_FILES = {
    "ambiguity": "ambiguity_scores.csv",
    "stability": "stability_scores.csv",
    "reliability_map": "reliability_map_scores.csv",
    "lrii": "local_rhythm_instability_scores.csv",
    "regularity_analysis": "regularity_features.csv",
    "uncertainty_analysis": "uncertainty_scores.csv",
    "decision_boundary": "decision_boundary_diagnosis.csv",
    "runtime_supervisor": "runtime_supervisor_policy.csv",
    "ambiguity_routing": "ambiguity_routing_policy.csv",
    "embedding_neighborhood": "embedding_neighborhood_k15.csv",
    "embedding_lda2": "embedding_lda2_coordinates.csv",
    "embedding_pca3": "embedding_pca3_coordinates.csv",
}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return (
        data["embeddings"].astype(np.float32),
        data["logits"].astype(np.float32),
        data["y"].astype(np.int64),
    )


def _load_checkpoint_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "best_model.pt"
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    return dict(args) if isinstance(args, dict) else {}


def _entropy(probs: np.ndarray) -> np.ndarray:
    return (-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(probs.shape[1])).astype(
        np.float32
    )


def _rank_margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return (ordered[:, -1] - ordered[:, -2]).astype(np.float32)


def _centroid_features(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    pred: np.ndarray,
) -> dict[str, np.ndarray]:
    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)
    nearest = dist.argmin(axis=1)
    d_vt, d_vf = dist[:, 1], dist[:, 2]
    d_sorted = np.sort(dist, axis=1)
    return {
        "proto_dist_sr": dist[:, 0],
        "proto_dist_vt": d_vt,
        "proto_dist_vf": d_vf,
        "nearest_proto": nearest.astype(np.float32),
        "nearest_proto_is_pred": (nearest == pred).astype(np.float32),
        "min_proto_dist": dist.min(axis=1),
        "proto_margin": (d_sorted[:, 1] - d_sorted[:, 0]).astype(np.float32),
        "proto_vtvf_ambiguity": (
            1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12)
        ).astype(np.float32),
        "abs_proto_vtvf_margin": np.abs(d_vf - d_vt).astype(np.float32),
        "proto_vtvf_prefers_vf": (d_vf < d_vt).astype(np.float32),
    }


def _knn_features(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    pred: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    k = min(k, len(train_emb))
    nn = NearestNeighbors(n_neighbors=k).fit(train_emb)
    distances, idx = nn.kneighbors(emb)
    neigh_y = train_y[idx]
    counts = np.stack([(neigh_y == c).mean(axis=1) for c in range(3)], axis=1)
    knn_pred = counts.argmax(axis=1)
    entropy = -np.sum(counts * np.log(np.maximum(counts, 1e-12)), axis=1) / np.log(3)
    vt = counts[:, 1]
    vf = counts[:, 2]
    ventricular = vt + vf
    vtvf_mix = np.zeros(len(emb), dtype=np.float32)
    valid = ventricular > 0
    vtvf_mix[valid] = 1.0 - np.abs(vt[valid] - vf[valid]) / ventricular[valid]
    return {
        "knn_mean_distance": distances.mean(axis=1).astype(np.float32),
        "knn_min_distance": distances[:, 0].astype(np.float32),
        "knn_label_entropy": entropy.astype(np.float32),
        "knn_pred": knn_pred.astype(np.float32),
        "knn_pred_is_model_pred": (knn_pred == pred).astype(np.float32),
        "knn_sr_fraction": counts[:, 0].astype(np.float32),
        "knn_vt_fraction": vt.astype(np.float32),
        "knn_vf_fraction": vf.astype(np.float32),
        "knn_vtvf_mixing": vtvf_mix.astype(np.float32),
    }


def _base_feature_frame(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    temperature: float,
    k: int,
    split: str,
) -> pd.DataFrame:
    probs = softmax(logits)
    probs_t = softmax(logits, temperature=temperature)
    pred = probs.argmax(axis=1)
    top2 = np.sort(np.argsort(probs, axis=1)[:, -2:], axis=1)
    ventricular_prob = probs[:, 1] + probs[:, 2]
    softmax_vtvf_ambiguity = ventricular_prob * (
        1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_prob, 1e-12)
    )
    centroid = _centroid_features(train_emb, train_y, emb, pred)
    knn = _knn_features(train_emb, train_y, emb, pred, k)

    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y), dtype=np.int64),
            "split": split,
            "y_true": y,
            "y_pred": pred,
            "prob_sr": probs[:, 0],
            "prob_vt": probs[:, 1],
            "prob_vf": probs[:, 2],
            "temperature_prob_sr": probs_t[:, 0],
            "temperature_prob_vt": probs_t[:, 1],
            "temperature_prob_vf": probs_t[:, 2],
            "max_prob": probs.max(axis=1),
            "rank_margin": _rank_margin(probs),
            "temperature_max_prob": probs_t.max(axis=1),
            "msp_uncertainty": 1.0 - probs.max(axis=1),
            "entropy": _entropy(probs),
            "temperature_entropy": _entropy(probs_t),
            "ventricular_prob": ventricular_prob,
            "softmax_vtvf_ambiguity": softmax_vtvf_ambiguity,
            "abs_prob_vtvf_margin": np.abs(probs[:, 1] - probs[:, 2]),
            "abs_logit_vtvf_margin": np.abs(logits[:, 1] - logits[:, 2]),
            "pred_is_vtvf": np.isin(pred, [1, 2]).astype(np.float32),
            "top2_are_vtvf": (top2 == np.asarray([1, 2])).all(axis=1).astype(np.float32),
            **centroid,
            **knn,
        }
    )
    df["is_error"] = df["y_true"] != df["y_pred"]
    df["is_vtvf_cross_error"] = ((df["y_true"] == 1) & (df["y_pred"] == 2)) | (
        (df["y_true"] == 2) & (df["y_pred"] == 1)
    )
    df["is_vtvf_truth"] = df["y_true"].isin([1, 2])
    df["is_vtvf_candidate"] = (df["pred_is_vtvf"].astype(bool) | df["top2_are_vtvf"].astype(bool)).astype(
        np.float32
    )
    return df


def _add_regularity_features(
    frames: dict[str, pd.DataFrame],
    args: dict[str, Any],
    mat_override: Path | None,
) -> dict[str, pd.DataFrame]:
    mat_path = mat_override or Path(str(args.get("mat", "RHYTHMS.mat")))
    if not mat_path.exists():
        print(f"Regularity features skipped: {mat_path} not found.")
        return frames

    dataset = load_rhythm_windows(
        mat_path,
        max_windows_per_record=args.get("max_windows_per_record"),
    )
    groups = None
    if args.get("split_grouping") == "duplicate_family":
        groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    elif args.get("split_grouping") in {"record", "record_id"}:
        groups = dataset.record_ids
    splits = make_splits(
        dataset.x,
        dataset.y,
        groups=groups,
        seed=int(args.get("seed", 42)),
    )
    raw_by_split = {
        "train": (splits.x_train, splits.y_train),
        "val": (splits.x_val, splits.y_val),
        "test": (splits.x_test, splits.y_test),
    }
    train_reg = extract_regularity_features_batch(raw_by_split["train"][0])
    mean = train_reg.mean(axis=0)
    std = train_reg.std(axis=0) + 1e-6

    for split, frame in frames.items():
        raw_x, raw_y = raw_by_split[split]
        if len(raw_y) != len(frame) or not np.array_equal(raw_y, frame["y_true"].to_numpy(int)):
            print(f"Regularity features skipped for {split}: reconstructed split does not align.")
            continue
        reg = (extract_regularity_features_batch(raw_x) - mean) / std
        for idx, name in enumerate(REGULARITY_FEATURE_NAMES):
            frame[f"regularity_{name}_z"] = reg[:, idx]
    return frames


def _add_second_opinion_features(
    frames: dict[str, pd.DataFrame],
    second_run_dir: Path | None,
    temperature: float,
) -> dict[str, pd.DataFrame]:
    if second_run_dir is None:
        return frames
    second_run_dir = _resolve_run_dir(second_run_dir)
    for split, frame in frames.items():
        _, logits, y = _load_split(second_run_dir, split)
        if len(y) != len(frame) or not np.array_equal(y, frame["y_true"].to_numpy(int)):
            raise ValueError(f"Second-opinion run does not align on {split}: {second_run_dir}")
        probs = softmax(logits)
        probs_t = softmax(logits, temperature=temperature)
        pred = probs.argmax(axis=1)
        primary_pred = frame["y_pred"].to_numpy(int)
        frame["second_prob_sr"] = probs[:, 0]
        frame["second_prob_vt"] = probs[:, 1]
        frame["second_prob_vf"] = probs[:, 2]
        frame["second_entropy"] = _entropy(probs)
        frame["second_temperature_entropy"] = _entropy(probs_t)
        frame["second_max_prob"] = probs.max(axis=1)
        frame["second_pred"] = pred.astype(np.float32)
        frame["model_disagreement"] = (pred != primary_pred).astype(np.float32)
        frame["model_disagreement_both_vtvf"] = (
            (pred != primary_pred) & np.isin(pred, [1, 2]) & np.isin(primary_pred, [1, 2])
        ).astype(np.float32)
        frame["model_disagreement_any_vtvf"] = (
            (pred != primary_pred) & (np.isin(pred, [1, 2]) | np.isin(primary_pred, [1, 2]))
        ).astype(np.float32)
        frame["second_softmax_vtvf_ambiguity"] = (probs[:, 1] + probs[:, 2]) * (
            1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(probs[:, 1] + probs[:, 2], 1e-12)
        )
    return frames


def _cluster_risk(values: np.ndarray, target: np.ndarray, alpha: float = 2.0) -> dict[int, float]:
    global_rate = float(target.mean())
    risks: dict[int, float] = {}
    for cluster in np.unique(values):
        mask = values == cluster
        risks[int(cluster)] = float((target[mask].sum() + alpha * global_rate) / (mask.sum() + alpha))
    return risks


def _add_latent_cluster_features(
    frames: dict[str, pd.DataFrame],
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    test_emb: np.ndarray,
    n_clusters: int,
) -> dict[str, pd.DataFrame]:
    n_clusters = max(2, min(n_clusters, len(train_emb)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    model.fit(train_emb)
    val_cluster = model.predict(val_emb)
    test_cluster = model.predict(test_emb)
    val_frame = frames["val"]
    error_risk = _cluster_risk(val_cluster, val_frame["is_error"].to_numpy(float))
    vtvf_risk = _cluster_risk(val_cluster, val_frame["is_vtvf_cross_error"].to_numpy(float))
    vtvf_density = _cluster_risk(val_cluster, val_frame["is_vtvf_truth"].to_numpy(float))
    distances = {
        "val": model.transform(val_emb).min(axis=1),
        "test": model.transform(test_emb).min(axis=1),
    }
    for split, clusters in {"val": val_cluster, "test": test_cluster}.items():
        frame = frames[split]
        frame["latent_cluster"] = clusters.astype(np.float32)
        frame["latent_cluster_distance"] = distances[split].astype(np.float32)
        frame["latent_cluster_val_error_rate"] = np.asarray([error_risk[int(c)] for c in clusters], dtype=np.float32)
        frame["latent_cluster_val_vtvf_cross_rate"] = np.asarray([vtvf_risk[int(c)] for c in clusters], dtype=np.float32)
        frame["latent_cluster_val_vtvf_truth_rate"] = np.asarray([vtvf_density[int(c)] for c in clusters], dtype=np.float32)
    return frames


def _aligned_diagnostic_frame(
    path: Path,
    frame: pd.DataFrame,
    prefix: str,
    allow_categorical: bool = True,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    manifest: dict[str, Any] = {
        "path": str(path),
        "prefix": prefix,
        "loaded": False,
        "n_added_columns": 0,
        "reason": "",
    }
    if not path.exists():
        manifest["reason"] = "missing"
        return None, manifest
    raw = pd.read_csv(path)
    if len(raw) != len(frame):
        manifest["reason"] = f"row_count_mismatch:{len(raw)}!={len(frame)}"
        return None, manifest
    if "y_true" in raw.columns and not np.array_equal(raw["y_true"].to_numpy(int), frame["y_true"].to_numpy(int)):
        manifest["reason"] = "y_true_mismatch"
        return None, manifest
    if "y_pred" in raw.columns and not np.array_equal(raw["y_pred"].to_numpy(int), frame["y_pred"].to_numpy(int)):
        manifest["reason"] = "y_pred_mismatch"
        return None, manifest

    keep: dict[str, pd.Series] = {}
    for col in raw.columns:
        low = col.lower()
        if low in LEAKAGE_OR_ID_COLUMNS or low in TEXT_COLUMNS_TO_SKIP:
            continue
        if pd.api.types.is_bool_dtype(raw[col]):
            keep[f"{prefix}_{col}"] = raw[col].astype(np.float32)
        elif pd.api.types.is_numeric_dtype(raw[col]):
            keep[f"{prefix}_{col}"] = raw[col].astype(np.float32)
        elif allow_categorical:
            values = raw[col].fillna("missing").astype(str)
            if values.nunique() <= 12:
                dummies = pd.get_dummies(values, prefix=f"{prefix}_{col}", dtype=np.float32)
                for dummy_col in dummies.columns:
                    keep[dummy_col] = dummies[dummy_col]
    if not keep:
        manifest["reason"] = "no_deployable_columns"
        return None, manifest
    manifest["loaded"] = True
    manifest["n_added_columns"] = len(keep)
    manifest["columns"] = list(keep.keys())
    return pd.DataFrame(keep), manifest


def _conformal_diagnostic_frame(path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    manifest: dict[str, Any] = {
        "path": str(path),
        "prefix": "conformal",
        "loaded": False,
        "n_added_columns": 0,
        "reason": "",
    }
    if not path.exists():
        manifest["reason"] = "missing"
        return None, manifest
    raw = pd.read_csv(path)
    if "index" not in raw.columns or "set_size" not in raw.columns:
        manifest["reason"] = "missing_index_or_set_size"
        return None, manifest
    if raw["index"].max() >= len(frame):
        manifest["reason"] = "index_out_of_range"
        return None, manifest
    work = raw.copy()
    if "y_true" in work.columns:
        aligned_y = frame.loc[work["index"].to_numpy(int), "y_true"].to_numpy(int)
        if not np.array_equal(work["y_true"].to_numpy(int), aligned_y):
            manifest["reason"] = "y_true_mismatch"
            return None, manifest
    method = work.get("method", pd.Series(["method"] * len(work))).astype(str)
    alpha = work.get("alpha", pd.Series(["alpha"] * len(work))).astype(str).str.replace(".", "p", regex=False)
    work["policy"] = method + "_alpha_" + alpha
    pivot = work.pivot_table(index="index", columns="policy", values="set_size", aggfunc="mean")
    pivot = pivot.reindex(np.arange(len(frame))).fillna(0.0)
    pivot.columns = [f"conformal_set_size_{col}" for col in pivot.columns]
    out = pivot.reset_index(drop=True).astype(np.float32)
    manifest["loaded"] = True
    manifest["n_added_columns"] = len(out.columns)
    manifest["columns"] = list(out.columns)
    return out, manifest


def _add_historical_diagnostics(
    frames: dict[str, pd.DataFrame],
    run_dir: Path,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    for split, frame in frames.items():
        for prefix, template in SPLIT_DIAGNOSTIC_FILES.items():
            diagnostic, info = _aligned_diagnostic_frame(run_dir / template.format(split=split), frame, prefix)
            info["split"] = split
            info["used_for_risk_training"] = diagnostic is not None
            manifest.append(info)
            if diagnostic is not None:
                frames[split] = pd.concat([frames[split].reset_index(drop=True), diagnostic], axis=1)

    test_frame = frames["test"]
    for prefix, filename in TEST_DIAGNOSTIC_FILES.items():
        diagnostic, info = _aligned_diagnostic_frame(run_dir / filename, test_frame, prefix)
        info["split"] = "test"
        info["used_for_risk_training"] = False
        manifest.append(info)
        if diagnostic is not None:
            frames["test"] = pd.concat([frames["test"].reset_index(drop=True), diagnostic], axis=1)

    conformal, info = _conformal_diagnostic_frame(run_dir / "conformal_sets.csv", frames["test"])
    info["split"] = "test"
    info["used_for_risk_training"] = False
    manifest.append(info)
    if conformal is not None:
        frames["test"] = pd.concat([frames["test"].reset_index(drop=True), conformal], axis=1)
    return frames, manifest


class ConstantRiskModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        p = np.full(len(features), self.value, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def _fit_binary_risk_model(x: pd.DataFrame, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        return ConstantRiskModel(float(y.mean()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(x, y)
    return model


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...] | None = None) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c not in TARGET_COLUMNS
        and not c.endswith("_mechanism_risk")
        and not c.endswith("_threshold_from_val")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes is None:
        return cols
    return [c for c in cols if c.startswith(prefixes)]


def _feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "softmax_only": _feature_columns(
            df,
            (
                "prob_",
                "temperature_prob_",
                "max_prob",
                "rank_margin",
                "temperature_max_prob",
                "msp_",
                "entropy",
                "temperature_entropy",
                "ventricular_",
                "softmax_",
                "abs_prob_",
                "abs_logit_",
                "pred_is_",
                "top2_",
            ),
        ),
        "representation_only": _feature_columns(df, ("proto_", "nearest_", "min_proto", "knn_", "abs_proto")),
        "regularity_only": _feature_columns(df, ("regularity_",)),
        "latent_cluster_only": _feature_columns(df, ("latent_cluster",)),
        "model_disagreement_only": _feature_columns(df, ("second_", "model_disagreement")),
        "historical_diagnostics": _feature_columns(df, ("risk_target_", "prior_calibration_")),
        "all_evidence": _feature_columns(df),
    }
    return {name: cols for name, cols in groups.items() if cols}


def _predict_risk(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, Any]:
    model = _fit_binary_risk_model(val_df[features], val_df[target].to_numpy(int))
    val_score = model.predict_proba(val_df[features])[:, 1]
    test_score = model.predict_proba(test_df[features])[:, 1]
    return val_score, test_score, model


def _columns_by_prefix(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in _feature_columns(df) if c.startswith(prefixes)]


def _unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _add_error_mechanism_targets(val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = {
        "high_conf": float(val_df["max_prob"].quantile(0.75)),
        "low_entropy": float(val_df["entropy"].quantile(0.35)),
        "low_knn_entropy": float(val_df["knn_label_entropy"].quantile(0.50)),
        "high_cluster_distance": float(val_df["latent_cluster_distance"].quantile(0.75)),
        "high_knn_distance": float(val_df["knn_mean_distance"].quantile(0.75)),
        "high_proto_distance": float(val_df["min_proto_dist"].quantile(0.75)),
    }
    for df in [val_df, test_df]:
        y = df["y_true"].to_numpy(int)
        pred = df["y_pred"].to_numpy(int)
        is_error = df["is_error"].to_numpy(bool)
        sr_ventricular = ((y == 0) & np.isin(pred, [1, 2])) | (np.isin(y, [1, 2]) & (pred == 0))
        representation_conflict = (
            df["nearest_proto_is_pred"].to_numpy(float) < 0.5
        ) | (df["knn_pred_is_model_pred"].to_numpy(float) < 0.5)
        atypical = (
            (df["latent_cluster_distance"].to_numpy(float) >= thresholds["high_cluster_distance"])
            | (df["knn_mean_distance"].to_numpy(float) >= thresholds["high_knn_distance"])
            | (df["min_proto_dist"].to_numpy(float) >= thresholds["high_proto_distance"])
        )
        if "model_disagreement" in df.columns:
            model_agrees = df["model_disagreement"].to_numpy(float) < 0.5
        else:
            model_agrees = np.ones(len(df), dtype=bool)
        hidden_confident = (
            (df["max_prob"].to_numpy(float) >= thresholds["high_conf"])
            & (df["entropy"].to_numpy(float) <= thresholds["low_entropy"])
            & (df["knn_label_entropy"].to_numpy(float) <= thresholds["low_knn_entropy"])
            & model_agrees
        )
        df["is_sr_ventricular_error"] = sr_ventricular
        df["is_representation_conflict_error"] = is_error & representation_conflict
        df["is_atypical_signal_error"] = is_error & atypical
        df["is_hidden_confident_error"] = is_error & hidden_confident
    return val_df, test_df


def _mechanism_feature_specs(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    softmax = _columns_by_prefix(
        df,
        (
            "prob_",
            "temperature_prob_",
            "max_prob",
            "rank_margin",
            "temperature_max_prob",
            "msp_",
            "entropy",
            "temperature_entropy",
            "ventricular_",
            "softmax_",
            "abs_prob_",
            "abs_logit_",
            "pred_is_",
            "top2_",
        ),
    )
    representation = _columns_by_prefix(df, ("proto_", "nearest_", "min_proto", "knn_", "abs_proto"))
    regularity = _columns_by_prefix(df, ("regularity_",))
    cluster = _columns_by_prefix(df, ("latent_cluster",))
    disagreement = _columns_by_prefix(df, ("second_", "model_disagreement"))
    historical = _columns_by_prefix(df, ("risk_target_", "prior_calibration_"))
    boundary = [
        c
        for c in softmax + representation + regularity + disagreement + historical
        if any(token in c for token in ["vt", "vf", "boundary", "mixing", "ambiguity", "margin", "entropy", "risk_target"])
    ]
    sr_ventricular = [
        c
        for c in softmax + representation + regularity + cluster + historical
        if any(token in c for token in ["sr", "ventricular", "regularity", "cluster", "proto", "knn", "entropy", "risk_target"])
    ]
    atypical = [
        c
        for c in representation + regularity + cluster + historical + softmax
        if any(token in c for token in ["distance", "atyp", "mahalanobis", "regularity", "cluster", "knn", "proto", "entropy", "risk_target"])
    ]
    hidden = [
        c
        for c in softmax + representation + disagreement + historical + regularity
        if any(token in c for token in ["max_prob", "entropy", "margin", "nearest", "knn", "proto", "disagreement", "risk", "regularity"])
    ]
    return {
        "vtvf_boundary": {
            "target": "is_vtvf_cross_error",
            "features": _unique_columns(boundary),
            "action": "vtvf_boundary_set",
        },
        "sr_ventricular": {
            "target": "is_sr_ventricular_error",
            "features": _unique_columns(sr_ventricular),
            "action": "sr_ventricular_review",
        },
        "representation_conflict": {
            "target": "is_representation_conflict_error",
            "features": _unique_columns(representation + historical),
            "action": "representation_review",
        },
        "atypical_signal": {
            "target": "is_atypical_signal_error",
            "features": _unique_columns(atypical),
            "action": "atypical_review",
        },
        "hidden_confident": {
            "target": "is_hidden_confident_error",
            "features": _unique_columns(hidden),
            "action": "hidden_failure_review",
        },
    }


def _fit_mechanism_risk_heads(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    specs = _mechanism_feature_specs(val_df)
    rows = []
    fitted_specs: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        target = spec["target"]
        features = spec["features"]
        if not features:
            continue
        val_score, test_score, _ = _predict_risk(val_df, test_df, features, target)
        score_col = f"{name}_mechanism_risk"
        val_df[score_col] = val_score
        test_df[score_col] = test_score
        fitted_specs[name] = {**spec, "score_col": score_col, "n_features": len(features)}
        val_positive = int(val_df[target].sum())
        test_positive = int(test_df[target].sum())
        fitted_specs[name]["val_positive"] = val_positive
        fitted_specs[name]["test_positive"] = test_positive
        fitted_specs[name]["enabled_for_routing"] = val_positive >= 5
        rows.append(
            {
                "mechanism": name,
                "target": target,
                "n_features": len(features),
                "val_positive": val_positive,
                "test_positive": test_positive,
                "enabled_for_routing": val_positive >= 5,
                "test_auroc": _safe_auc(test_df[target].to_numpy(int), test_score),
                "test_aupr": _safe_aupr(test_df[target].to_numpy(int), test_score),
            }
        )
    return val_df, test_df, pd.DataFrame(rows), fitted_specs


def _top_budget_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = max(1, int(round(len(score) * budget)))
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n]] = True
    return mask


def _review_metrics(df: pd.DataFrame, score: np.ndarray, budget: float) -> dict[str, float | int]:
    mask = _top_budget_mask(score, budget)
    auto = ~mask
    errors = df["is_error"].to_numpy(bool)
    vtvf = df["is_vtvf_cross_error"].to_numpy(bool)
    return {
        "reviewed": int(mask.sum()),
        "review_rate": float(mask.mean()),
        "all_error_captured": float((errors & mask).sum() / max(errors.sum(), 1)),
        "vtvf_cross_error_captured": float((vtvf & mask).sum() / max(vtvf.sum(), 1)),
        "review_error_enrichment": float(errors[mask].mean() / max(errors.mean(), 1e-8)),
        "auto_error_rate": float(errors[auto].mean()) if auto.any() else float("nan"),
        "auto_vtvf_cross_error_rate": float(vtvf[auto].mean()) if auto.any() else float("nan"),
    }


def _ablation_summary(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budgets: list[float],
) -> pd.DataFrame:
    rows = []
    for group, features in _feature_groups(val_df).items():
        for target, score_name in [
            ("is_error", "any_error"),
            ("is_vtvf_cross_error", "vtvf_cross_error"),
        ]:
            _, test_score, _ = _predict_risk(val_df, test_df, features, target)
            for budget in budgets:
                row = {
                    "feature_group": group,
                    "target": score_name,
                    "budget": budget,
                    "n_features": len(features),
                    "auroc": _safe_auc(test_df[target].to_numpy(int), test_score),
                    "aupr": _safe_aupr(test_df[target].to_numpy(int), test_score),
                }
                row.update(_review_metrics(test_df, test_score, budget))
                rows.append(row)
    return pd.DataFrame(rows)


def _threshold_from_validation(score: np.ndarray, budget: float) -> float:
    n = max(1, int(round(len(score) * budget)))
    ordered = np.sort(score)[::-1]
    return float(ordered[min(n - 1, len(ordered) - 1)])


def _assign_layered_actions(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    review_budget: float,
    set_budget: float,
) -> pd.DataFrame:
    review_threshold = _threshold_from_validation(val_df["any_error_risk"].to_numpy(float), review_budget)
    candidates = val_df["is_vtvf_candidate"].to_numpy(bool)
    candidate_scores = val_df.loc[candidates, "vtvf_boundary_risk"].to_numpy(float)
    set_threshold = _threshold_from_validation(candidate_scores, set_budget) if len(candidate_scores) else 1.0

    routed = test_df.copy()
    review = routed["any_error_risk"].to_numpy(float) >= review_threshold
    set_vtvf = (
        ~review
        & routed["is_vtvf_candidate"].to_numpy(bool)
        & (routed["vtvf_boundary_risk"].to_numpy(float) >= set_threshold)
    )
    actions = np.full(len(routed), "single_label", dtype=object)
    actions[set_vtvf] = "vtvf_set"
    actions[review] = "review"
    routed["action"] = actions
    output_set = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    output_set[set_vtvf] = "{VT,VF}"
    output_set[review] = "review"
    routed["output_set"] = output_set
    routed["review_threshold_from_val"] = review_threshold
    routed["set_threshold_from_val"] = set_threshold
    return routed


def _threshold_on_candidates(score: np.ndarray, candidate: np.ndarray, budget: float) -> float:
    values = score[candidate]
    if len(values) == 0:
        return float("inf")
    return _threshold_from_validation(values, budget)


def _threshold_top_n_on_candidates(score: np.ndarray, candidate: np.ndarray, n_select: int) -> float:
    values = score[candidate]
    if len(values) == 0 or n_select <= 0:
        return float("inf")
    ordered = np.sort(values)[::-1]
    return float(ordered[min(n_select - 1, len(ordered) - 1)])


def _assign_mechanism_actions(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
) -> pd.DataFrame:
    thresholds: dict[str, float] = {}
    budget_weights = {
        "vtvf_boundary": 0.35,
        "sr_ventricular": 0.20,
        "representation_conflict": 0.15,
        "atypical_signal": 0.20,
        "hidden_confident": 0.10,
    }
    enabled = [name for name, spec in specs.items() if spec.get("enabled_for_routing", True)]
    weight_sum = sum(budget_weights.get(name, 0.0) for name in enabled)
    total_slots = max(1, int(round(len(val_df) * budget)))
    for name, spec in specs.items():
        score_col = spec["score_col"]
        if not spec.get("enabled_for_routing", True):
            thresholds[name] = float("inf")
            continue
        if name == "vtvf_boundary":
            candidate = val_df["is_vtvf_candidate"].to_numpy(bool)
        elif name == "hidden_confident":
            candidate = val_df["max_prob"].to_numpy(float) >= val_df["max_prob"].quantile(0.50)
        else:
            candidate = np.ones(len(val_df), dtype=bool)
        allocation = budget_weights.get(name, 0.0) / max(weight_sum, 1e-12)
        n_select = max(1, int(round(total_slots * allocation)))
        thresholds[name] = _threshold_top_n_on_candidates(val_df[score_col].to_numpy(float), candidate, n_select)

    routed = test_df.copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    mechanism = np.full(len(routed), "single_label", dtype=object)

    priority = [
        "hidden_confident",
        "vtvf_boundary",
        "sr_ventricular",
        "atypical_signal",
        "representation_conflict",
    ]
    for name in priority:
        if name not in specs:
            continue
        if not specs[name].get("enabled_for_routing", True):
            continue
        score_col = specs[name]["score_col"]
        if name == "vtvf_boundary":
            candidate = routed["is_vtvf_candidate"].to_numpy(bool)
        elif name == "hidden_confident":
            candidate = routed["max_prob"].to_numpy(float) >= val_df["max_prob"].quantile(0.50)
        else:
            candidate = np.ones(len(routed), dtype=bool)
        flag = (actions == "single_label") & candidate & (routed[score_col].to_numpy(float) >= thresholds[name])
        actions[flag] = specs[name]["action"]
        mechanism[flag] = name

    output_set = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    review_mask = actions != "single_label"
    vtvf_set = actions == "vtvf_boundary_set"
    output_set[review_mask] = "review"
    output_set[vtvf_set] = "{VT,VF}"
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = mechanism
    routed["mechanism_output_set"] = output_set
    routed["mechanism_budget"] = budget
    routed["mechanism_strategy"] = "fixed_weight"
    for name, threshold in thresholds.items():
        routed[f"{name}_threshold_from_val"] = threshold
    return routed


def _mechanism_candidate_mask(
    df: pd.DataFrame,
    name: str,
    val_df: pd.DataFrame,
) -> np.ndarray:
    if name == "vtvf_boundary":
        return df["is_vtvf_candidate"].to_numpy(bool)
    if name == "hidden_confident":
        return df["max_prob"].to_numpy(float) >= val_df["max_prob"].quantile(0.50)
    return np.ones(len(df), dtype=bool)


def _mechanism_pair_utility(df: pd.DataFrame, sample_idx: int, mechanism: str, specs: dict[str, dict[str, Any]]) -> float:
    is_error = float(bool(df.iloc[sample_idx]["is_error"]))
    is_vtvf = float(bool(df.iloc[sample_idx]["is_vtvf_cross_error"]))
    target = specs[mechanism]["target"]
    mechanism_hit = float(bool(df.iloc[sample_idx][target])) if target in df.columns else 0.0
    return is_error + is_vtvf + mechanism_hit


def _candidate_tuples(
    df: pd.DataFrame,
    val_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> list[tuple[float, int, str]]:
    candidates: list[tuple[float, int, str]] = []
    for name, spec in specs.items():
        if not spec.get("enabled_for_routing", True):
            continue
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        mask = _mechanism_candidate_mask(df, name, val_df)
        scores = df[spec["score_col"]].to_numpy(float) * weight
        for idx in np.flatnonzero(mask):
            candidates.append((float(scores[idx]), int(idx), name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _greedy_unique_selection(
    candidates: list[tuple[float, int, str]],
    n_select: int,
) -> list[tuple[int, str, float]]:
    selected: list[tuple[int, str, float]] = []
    used: set[int] = set()
    for score, idx, name in candidates:
        if idx in used:
            continue
        used.add(idx)
        selected.append((idx, name, score))
        if len(selected) >= n_select:
            break
    return selected


def _select_weighted_mechanism_candidates(
    df: pd.DataFrame,
    val_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_select = max(1, int(round(len(df) * budget)))
    candidates = _candidate_tuples(df, val_df, specs, weights)
    selected = _greedy_unique_selection(candidates, n_select)

    routed = df.copy()
    actions = np.full(len(routed), "single_label", dtype=object)
    routes = np.full(len(routed), "single_label", dtype=object)
    weighted_scores = np.zeros(len(routed), dtype=np.float32)
    for idx, name, score in selected:
        actions[idx] = specs[name]["action"]
        routes[idx] = name
        weighted_scores[idx] = float(score)

    output_set = np.asarray([CLASS_LABELS[int(pred)] for pred in routed["y_pred"].to_numpy(int)], dtype=object)
    review_mask = actions != "single_label"
    vtvf_set = actions == "vtvf_boundary_set"
    output_set[review_mask] = "review"
    output_set[vtvf_set] = "{VT,VF}"
    routed["mechanism_action"] = actions
    routed["mechanism_route"] = routes
    routed["mechanism_output_set"] = output_set
    routed["mechanism_budget"] = budget
    routed["mechanism_strategy"] = "validation_optimized"
    routed["mechanism_weighted_score"] = weighted_scores
    for name, weight in weights.items():
        routed[f"{name}_optimized_weight"] = float(weight)

    diagnostics = {
        "budget": budget,
        "requested_slots": n_select,
        "selected_slots": len(selected),
        "selected_rate": len(selected) / len(df),
        "weights": weights,
        "route_counts": pd.Series([name for _, name, _ in selected]).value_counts().to_dict() if selected else {},
    }
    return routed, diagnostics


def _validation_utility_for_weights(
    val_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
    weights: dict[str, float],
) -> float:
    n_select = max(1, int(round(len(val_df) * budget)))
    selected = _greedy_unique_selection(_candidate_tuples(val_df, val_df, specs, weights), n_select)
    utility = 0.0
    for idx, route, _ in selected:
        utility += _mechanism_pair_utility(val_df, int(idx), route, specs)
    return float(utility)


def _optimize_mechanism_weights(
    val_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    enabled = [name for name, spec in specs.items() if spec.get("enabled_for_routing", True)]
    template_profiles = [
        ("equal", {"vtvf_boundary": 1.0, "sr_ventricular": 1.0, "representation_conflict": 1.0, "atypical_signal": 1.0}),
        ("vtvf_boundary_heavy", {"vtvf_boundary": 4.0, "sr_ventricular": 1.0, "representation_conflict": 1.0, "atypical_signal": 1.0}),
        ("sr_ventricular_heavy", {"vtvf_boundary": 1.0, "sr_ventricular": 4.0, "representation_conflict": 1.0, "atypical_signal": 1.0}),
        ("representation_heavy", {"vtvf_boundary": 1.0, "sr_ventricular": 1.0, "representation_conflict": 4.0, "atypical_signal": 1.0}),
        ("atypical_heavy", {"vtvf_boundary": 1.0, "sr_ventricular": 1.0, "representation_conflict": 1.0, "atypical_signal": 4.0}),
        ("boundary_atypical", {"vtvf_boundary": 3.0, "sr_ventricular": 1.0, "representation_conflict": 1.0, "atypical_signal": 3.0}),
        ("representation_atypical", {"vtvf_boundary": 1.0, "sr_ventricular": 1.0, "representation_conflict": 3.0, "atypical_signal": 3.0}),
        ("vtvf_only", {"vtvf_boundary": 1.0}),
        ("sr_only", {"sr_ventricular": 1.0}),
        ("representation_only", {"representation_conflict": 1.0}),
        ("atypical_only", {"atypical_signal": 1.0}),
    ]
    best_score = -1.0
    best_weights = {name: 0.0 for name in specs}
    best_profile = ""
    for profile_name, template in template_profiles:
        weights = {name: 0.0 for name in specs}
        for name in enabled:
            weights[name] = float(template.get(name, 0.0))
        if not any(value > 0 for value in weights.values()):
            continue
        score = _validation_utility_for_weights(val_df, specs, budget, weights)
        active = sum(value > 0 for value in weights.values())
        best_active = sum(value > 0 for value in best_weights.values())
        if score > best_score or (np.isclose(score, best_score) and active < best_active):
            best_score = score
            best_weights = weights
            best_profile = profile_name
    diagnostics = {
        "budget": budget,
        "validation_utility": best_score,
        "selected_profile": best_profile,
        "weights": best_weights,
    }
    return best_weights, diagnostics


def _assign_optimized_mechanism_actions(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weights, optimization = _optimize_mechanism_weights(val_df, specs, budget)
    routed, selection = _select_weighted_mechanism_candidates(test_df, val_df, specs, budget, weights)
    diagnostics = {
        "budget": budget,
        "optimization": optimization,
        "test_selection": selection,
    }
    return routed, diagnostics


def _summarize_mechanism_routing(
    routed: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budget: float,
    strategy: str,
) -> dict[str, Any]:
    y = test_df["y_true"].to_numpy(int)
    pred = test_df["y_pred"].to_numpy(int)
    baseline_error = test_df["is_error"].to_numpy(bool)
    baseline_vtvf = test_df["is_vtvf_cross_error"].to_numpy(bool)
    action = routed["mechanism_action"].to_numpy(str)
    single = action == "single_label"
    addressed = ~single
    vtvf_set = action == "vtvf_boundary_set"
    unresolved_error = single & (y != pred)
    unresolved_vtvf = single & baseline_vtvf
    row: dict[str, Any] = {
        "strategy": strategy,
        "budget": budget,
        "baseline_error_rate": float(baseline_error.mean()),
        "baseline_vtvf_cross_error_rate": float(baseline_vtvf.mean()),
        "single_label_rate": float(single.mean()),
        "mechanism_action_rate": float(addressed.mean()),
        "vtvf_set_rate": float(vtvf_set.mean()),
        "all_error_addressed": float((baseline_error & addressed).sum() / max(baseline_error.sum(), 1)),
        "vtvf_cross_error_addressed": float((baseline_vtvf & addressed).sum() / max(baseline_vtvf.sum(), 1)),
        "single_label_error_rate_after_mechanism_routing": float(unresolved_error.sum() / max(single.sum(), 1)),
        "single_label_vtvf_cross_error_rate_after_mechanism_routing": float(
            unresolved_vtvf.sum() / max(single.sum(), 1)
        ),
        "automatic_unresolved_error_rate": float(unresolved_error.mean()),
        "automatic_unresolved_vtvf_cross_error_rate": float(unresolved_vtvf.mean()),
    }
    mechanism_targets = [spec["target"] for spec in specs.values()]
    for name in specs:
        route_mask = routed["mechanism_route"].eq(name).to_numpy(bool)
        row[f"{name}_route_rate"] = float(route_mask.mean())
        target = specs[name]["target"]
        if target in routed.columns:
            target_mask = routed[target].to_numpy(bool)
            row[f"{name}_target_captured"] = float((target_mask & route_mask).sum() / max(target_mask.sum(), 1))
    for target in mechanism_targets:
        if target in routed.columns:
            target_mask = routed[target].to_numpy(bool)
            row[f"{target}_captured_by_any_mechanism"] = float(
                (target_mask & addressed).sum() / max(target_mask.sum(), 1)
            )
    return row


def _mechanism_policy_summary(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budgets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    routed_frames = []
    for budget in budgets:
        routed = _assign_mechanism_actions(val_df, test_df, specs, budget)
        rows.append(_summarize_mechanism_routing(routed, test_df, specs, budget, "fixed_weight"))
        routed_frames.append(routed)
    return pd.DataFrame(rows), pd.concat(routed_frames, ignore_index=True)


def _optimized_mechanism_policy_summary(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    budgets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    routed_frames = []
    diagnostics = []
    for budget in budgets:
        routed, info = _assign_optimized_mechanism_actions(val_df, test_df, specs, budget)
        rows.append(_summarize_mechanism_routing(routed, test_df, specs, budget, "validation_optimized"))
        diagnostics.append(info)
        routed_frames.append(routed)
    return pd.DataFrame(rows), pd.concat(routed_frames, ignore_index=True), pd.DataFrame(diagnostics)


def _mechanism_route_audit(routed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["mechanism_budget", "mechanism_route", "mechanism_action"]
    score_cols = [c for c in routed.columns if c.endswith("_mechanism_risk")]
    for keys, sub in routed.groupby(group_cols, sort=True):
        budget, route, action = keys
        row: dict[str, Any] = {
            "budget": budget,
            "mechanism_route": route,
            "mechanism_action": action,
            "n": int(len(sub)),
            "error_rate": float(sub["is_error"].mean()),
            "vtvf_cross_error_rate": float(sub["is_vtvf_cross_error"].mean()),
            "sr_ventricular_error_rate": float(sub["is_sr_ventricular_error"].mean()),
            "representation_conflict_error_rate": float(sub["is_representation_conflict_error"].mean()),
            "atypical_signal_error_rate": float(sub["is_atypical_signal_error"].mean()),
            "hidden_confident_error_rate": float(sub["is_hidden_confident_error"].mean()),
            "mean_entropy": float(sub["entropy"].mean()),
            "mean_knn_vtvf_mixing": float(sub["knn_vtvf_mixing"].mean()),
            "mean_nearest_proto_is_pred": float(sub["nearest_proto_is_pred"].mean()),
        }
        for col in score_cols:
            row[f"mean_{col}"] = float(sub[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _layered_policy_summary(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budgets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    routed_frames = []
    baseline_error = test_df["is_error"].to_numpy(bool)
    baseline_vtvf = test_df["is_vtvf_cross_error"].to_numpy(bool)
    y = test_df["y_true"].to_numpy(int)
    pred = test_df["y_pred"].to_numpy(int)
    for budget in budgets:
        routed = _assign_layered_actions(val_df, test_df, review_budget=budget, set_budget=budget)
        action = routed["action"].to_numpy(str)
        review = action == "review"
        set_vtvf = action == "vtvf_set"
        single = action == "single_label"
        set_contains_true = set_vtvf & np.isin(y, [1, 2])
        unresolved_error = single & (y != pred)
        unresolved_vtvf = single & baseline_vtvf
        addressed = review | set_vtvf
        summaries.append(
            {
                "budget": budget,
                "baseline_error_rate": float(baseline_error.mean()),
                "baseline_vtvf_cross_error_rate": float(baseline_vtvf.mean()),
                "review_rate": float(review.mean()),
                "vtvf_set_rate": float(set_vtvf.mean()),
                "single_label_rate": float(single.mean()),
                "set_contains_true_rate": float(set_contains_true[set_vtvf].mean()) if set_vtvf.any() else np.nan,
                "all_error_addressed_by_review_or_set": float((baseline_error & addressed).sum() / max(baseline_error.sum(), 1)),
                "vtvf_cross_error_addressed_by_review_or_set": float(
                    (baseline_vtvf & addressed).sum() / max(baseline_vtvf.sum(), 1)
                ),
                "single_label_error_rate_after_routing": float(unresolved_error.sum() / max(single.sum(), 1)),
                "single_label_vtvf_cross_error_rate_after_routing": float(
                    unresolved_vtvf.sum() / max(single.sum(), 1)
                ),
                "automatic_unresolved_error_rate": float(unresolved_error.mean()),
                "automatic_unresolved_vtvf_cross_error_rate": float(unresolved_vtvf.mean()),
            }
        )
        routed["budget"] = budget
        routed_frames.append(routed)
    return pd.DataFrame(summaries), pd.concat(routed_frames, ignore_index=True)


def _post_routing_audit(routed: pd.DataFrame) -> pd.DataFrame:
    audit_features = [
        "entropy",
        "softmax_vtvf_ambiguity",
        "proto_vtvf_ambiguity",
        "knn_vtvf_mixing",
        "knn_label_entropy",
        "nearest_proto_is_pred",
        "latent_cluster_val_error_rate",
        "latent_cluster_val_vtvf_cross_rate",
        "model_disagreement",
        "any_error_risk",
        "vtvf_boundary_risk",
        "risk_target_risk_target",
        "risk_target_local_instability",
        "risk_target_vtvf_mixing",
        "risk_target_prototype",
        "stability_stability_risk",
        "stability_stability_aware_risk",
        "stability_pred_flip_rate",
        "stability_embedding_drift",
        "runtime_supervisor_supervisor_risk",
        "runtime_supervisor_boundary_risk",
        "runtime_supervisor_quality_risk",
        "lrii_lrii",
        "lrii_boundary_lrii",
        "lrii_atypicality_lrii",
        "reliability_map_atypicality_score",
        "reliability_map_boundary_ambiguity_score",
        "regularity_analysis_sample_entropy",
        "regularity_analysis_mahalanobis_atypicality",
        "uncertainty_analysis_energy",
        "uncertainty_analysis_mahalanobis",
        "embedding_neighborhood_local_purity",
        "embedding_neighborhood_vtvf_mixing",
        "decision_boundary_classifier_proto_disagree",
    ]
    present = [c for c in audit_features if c in routed.columns]
    rows = []
    for (budget, action), sub in routed.groupby(["budget", "action"], sort=True):
        row: dict[str, Any] = {
            "budget": budget,
            "action": action,
            "n": int(len(sub)),
            "rate_within_budget": float(len(sub) / max((routed["budget"] == budget).sum(), 1)),
            "error_rate": float(sub["is_error"].mean()),
            "vtvf_cross_error_rate": float(sub["is_vtvf_cross_error"].mean()),
            "vtvf_truth_rate": float(sub["is_vtvf_truth"].mean()),
        }
        for col in present:
            row[f"mean_{col}"] = float(sub[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--second-opinion-run-dir", type=Path, default=None)
    parser.add_argument("--mat", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = args.out or (run_dir / "evidence_informed_recovery_routing")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_emb, train_logits, train_y = _load_split(run_dir, "train")
    val_emb, val_logits, val_y = _load_split(run_dir, "val")
    test_emb, test_logits, test_y = _load_split(run_dir, "test")
    temperature = fit_temperature(val_logits, val_y)

    frames = {
        "val": _base_feature_frame(train_emb, train_y, val_emb, val_logits, val_y, temperature, args.k, "val"),
        "test": _base_feature_frame(train_emb, train_y, test_emb, test_logits, test_y, temperature, args.k, "test"),
    }
    checkpoint_args = _load_checkpoint_args(run_dir)
    frames = _add_regularity_features(frames, checkpoint_args, args.mat)
    second_temperature = temperature
    if args.second_opinion_run_dir is not None:
        _, second_val_logits, second_val_y = _load_split(_resolve_run_dir(args.second_opinion_run_dir), "val")
        if np.array_equal(second_val_y, val_y):
            second_temperature = fit_temperature(second_val_logits, second_val_y)
    frames = _add_second_opinion_features(frames, args.second_opinion_run_dir, second_temperature)
    frames = _add_latent_cluster_features(frames, train_emb, val_emb, test_emb, args.n_clusters)
    frames, diagnostic_manifest = _add_historical_diagnostics(frames, run_dir)

    val_df = frames["val"]
    test_df = frames["test"]
    val_df, test_df = _add_error_mechanism_targets(val_df, test_df)
    features = _feature_columns(val_df)
    val_df["any_error_risk"], test_df["any_error_risk"], _ = _predict_risk(
        val_df, test_df, features, "is_error"
    )
    val_df["vtvf_boundary_risk"], test_df["vtvf_boundary_risk"], _ = _predict_risk(
        val_df, test_df, features, "is_vtvf_cross_error"
    )
    val_df, test_df, mechanism_head_df, mechanism_specs = _fit_mechanism_risk_heads(val_df, test_df)

    val_df.to_csv(out_dir / "evidence_scores_val.csv", index=False)
    test_df.to_csv(out_dir / "evidence_scores_test.csv", index=False)

    ablation_df = _ablation_summary(val_df, test_df, args.budgets)
    ablation_df.to_csv(out_dir / "evidence_ablation_summary.csv", index=False)

    policy_df, routed_df = _layered_policy_summary(val_df, test_df, args.budgets)
    policy_df.to_csv(out_dir / "layered_policy_summary.csv", index=False)
    routed_df.to_csv(out_dir / "layered_routing_assignments_test.csv", index=False)

    audit_df = _post_routing_audit(routed_df)
    audit_df.to_csv(out_dir / "post_routing_audit.csv", index=False)

    mechanism_head_df.to_csv(out_dir / "mechanism_risk_head_summary.csv", index=False)
    mechanism_policy_df, mechanism_routed_df = _mechanism_policy_summary(
        val_df, test_df, mechanism_specs, args.budgets
    )
    mechanism_policy_df.to_csv(out_dir / "mechanism_layered_policy_summary.csv", index=False)
    mechanism_routed_df.to_csv(out_dir / "mechanism_routing_assignments_test.csv", index=False)
    mechanism_route_audit_df = _mechanism_route_audit(mechanism_routed_df)
    mechanism_route_audit_df.to_csv(out_dir / "mechanism_route_audit.csv", index=False)
    optimized_policy_df, optimized_routed_df, optimized_diagnostics_df = _optimized_mechanism_policy_summary(
        val_df, test_df, mechanism_specs, args.budgets
    )
    optimized_policy_df.to_csv(out_dir / "optimized_mechanism_layered_policy_summary.csv", index=False)
    optimized_routed_df.to_csv(out_dir / "optimized_mechanism_routing_assignments_test.csv", index=False)
    optimized_route_audit_df = _mechanism_route_audit(optimized_routed_df)
    optimized_route_audit_df.to_csv(out_dir / "optimized_mechanism_route_audit.csv", index=False)
    optimized_diagnostics_df.to_csv(out_dir / "optimized_mechanism_budget_diagnostics.csv", index=False)

    probs = test_df[["prob_sr", "prob_vt", "prob_vf"]].to_numpy(float)
    probs_t = test_df[["temperature_prob_sr", "temperature_prob_vt", "temperature_prob_vf"]].to_numpy(float)
    metrics = {
        "run_dir": str(run_dir),
        "second_opinion_run_dir": str(_resolve_run_dir(args.second_opinion_run_dir))
        if args.second_opinion_run_dir
        else None,
        "out_dir": str(out_dir),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "temperature": float(temperature),
        "second_temperature": float(second_temperature),
        "baseline_accuracy": float((test_df["y_true"] == test_df["y_pred"]).mean()),
        "baseline_error_rate": float(test_df["is_error"].mean()),
        "baseline_vtvf_cross_errors": int(test_df["is_vtvf_cross_error"].sum()),
        "baseline_vtvf_cross_error_rate": float(test_df["is_vtvf_cross_error"].mean()),
        "ece_before_temperature": expected_calibration_error(test_y, probs),
        "ece_after_temperature": expected_calibration_error(test_y, probs_t),
        "any_error_risk_auroc": _safe_auc(test_df["is_error"].to_numpy(int), test_df["any_error_risk"].to_numpy(float)),
        "any_error_risk_aupr": _safe_aupr(test_df["is_error"].to_numpy(int), test_df["any_error_risk"].to_numpy(float)),
        "vtvf_boundary_risk_auroc": _safe_auc(
            test_df["is_vtvf_cross_error"].to_numpy(int), test_df["vtvf_boundary_risk"].to_numpy(float)
        ),
        "vtvf_boundary_risk_aupr": _safe_aupr(
            test_df["is_vtvf_cross_error"].to_numpy(int), test_df["vtvf_boundary_risk"].to_numpy(float)
        ),
        "mechanism_heads": mechanism_head_df.to_dict(orient="records"),
        "optimized_mechanism_budget_diagnostics": optimized_diagnostics_df.to_dict(orient="records"),
        "mechanism_specs": {
            name: {
                "target": spec["target"],
                "action": spec["action"],
                "score_col": spec["score_col"],
                "n_features": spec["n_features"],
                "val_positive": spec.get("val_positive"),
                "test_positive": spec.get("test_positive"),
                "enabled_for_routing": spec.get("enabled_for_routing", True),
            }
            for name, spec in mechanism_specs.items()
        },
        "feature_groups": {name: len(cols) for name, cols in _feature_groups(val_df).items()},
        "feature_columns": features,
        "historical_diagnostics_manifest": diagnostic_manifest,
    }
    _write_json(out_dir / "layered_decision_system_report.json", metrics)

    print(pd.DataFrame([metrics]).drop(columns=["feature_columns", "feature_groups"]))
    print(policy_df)
    print(mechanism_head_df)
    print(mechanism_policy_df)
    print(optimized_policy_df)
    print(
        ablation_df[ablation_df["budget"].isin([0.10, 0.20, 0.30])]
        .sort_values(["target", "budget", "vtvf_cross_error_captured"], ascending=[True, True, False])
        .head(24)
    )


if __name__ == "__main__":
    main()
