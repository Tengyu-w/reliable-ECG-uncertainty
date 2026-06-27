from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import softmax


CLASS_NAMES = {0: "SR", 1: "VT", 2: "VF"}
VT, VF = 1, 2


@dataclass(frozen=True)
class RunSpec:
    family: str
    model: str
    seed: int
    run_dir: Path
    pair_group: str


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=1)


def _binary_metric(y: np.ndarray, score: np.ndarray, fn) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(fn(y[mask], score[mask]))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 20260626) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(sample, [2.5, 97.5])
    return float(lo), float(hi)


def _resolve_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _read_embeddings(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    logits = data["logits"]
    emb = data["embeddings"]
    y = data["y"].astype(int)
    return emb, y, logits


def _probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return softmax(logits)


def _prediction_frame(run_dir: Path, split: str = "test") -> pd.DataFrame:
    emb, y, logits = _read_embeddings(run_dir, split)
    probs = _probs_from_logits(logits)
    pred = probs.argmax(axis=1)
    out = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": pred,
            "prob_SR": probs[:, 0],
            "prob_VT": probs[:, 1],
            "prob_VF": probs[:, 2],
        }
    )
    out["confidence"] = probs.max(axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    out["margin"] = top2[:, 1] - top2[:, 0]
    out["entropy"] = _entropy(probs) / np.log(probs.shape[1])
    out["is_error"] = out["y_true"] != out["y_pred"]
    out["is_vtvf"] = out["y_true"].isin([VT, VF])
    out["is_vtvf_boundary_error"] = (
        ((out["y_true"] == VT) & (out["y_pred"] == VF))
        | ((out["y_true"] == VF) & (out["y_pred"] == VT))
    )
    out["softmax_vtvf_ambiguity"] = 1.0 - np.abs(out["prob_VT"] - out["prob_VF"]) / np.maximum(
        out["prob_VT"] + out["prob_VF"], 1e-12
    )
    return out


def _feature_frame(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray, logits: np.ndarray, k: int, split: str) -> pd.DataFrame:
    probs = _probs_from_logits(logits)
    pred = probs.argmax(axis=1)
    y = train_y if split == "train" and emb.shape[0] == train_y.shape[0] else None

    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)
    d_sr, d_vt, d_vf = dist[:, 0], dist[:, 1], dist[:, 2]
    sorted_dist = np.sort(dist, axis=1)

    if split == "train":
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_emb))).fit(train_emb)
        _, idx = nn.kneighbors(emb)
        idx = idx[:, 1:]
    else:
        nn = NearestNeighbors(n_neighbors=min(k, len(train_emb))).fit(train_emb)
        _, idx = nn.kneighbors(emb)
    neigh_y = train_y[idx]
    label_probs = np.stack([(neigh_y == c).mean(axis=1) for c in range(3)], axis=1)
    knn_entropy = _entropy(label_probs) / np.log(3)
    knn_vtvf_mix = 1.0 - np.abs(label_probs[:, VT] - label_probs[:, VF]) / np.maximum(
        label_probs[:, VT] + label_probs[:, VF], 1e-12
    )
    knn_vtvf_mix[(label_probs[:, VT] + label_probs[:, VF]) == 0] = 0.0

    out = pd.DataFrame(
        {
            "confidence": probs.max(axis=1),
            "margin": np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2],
            "entropy": _entropy(probs) / np.log(3),
            "softmax_vtvf_ambiguity": 1.0 - np.abs(probs[:, VT] - probs[:, VF]) / np.maximum(
                probs[:, VT] + probs[:, VF], 1e-12
            ),
            "prototype_vtvf_ambiguity": 1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12),
            "knn_entropy": knn_entropy,
            "knn_vtvf_mix": knn_vtvf_mix,
            "d_sr": d_sr,
            "d_vt": d_vt,
            "d_vf": d_vf,
            "nearest_centroid_distance": sorted_dist[:, 0],
            "centroid_distance_margin": sorted_dist[:, 1] - sorted_dist[:, 0],
            "ventricular_neighbor_fraction": label_probs[:, VT] + label_probs[:, VF],
            "knn_pred_agreement": label_probs[np.arange(len(label_probs)), pred],
        }
    )
    if y is not None:
        out["local_purity"] = label_probs[np.arange(len(label_probs)), y]
    return out


def _run_specs_from_cnn_lstm(root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for comp_dir in sorted(root.glob("seed*_comparison")):
        seed_text = comp_dir.name.replace("seed", "").replace("_comparison", "")
        if not seed_text.isdigit():
            continue
        path = comp_dir / "classification_run_comparison.csv"
        if not path.exists():
            continue
        seed = int(seed_text)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            model = str(row["model"])
            runs.append(
                RunSpec(
                    family="cnn_lstm",
                    model=model,
                    seed=seed,
                    run_dir=_resolve_run_dir(row["run_dir"]),
                    pair_group=f"cnn_lstm_seed{seed}",
                )
            )
    return runs


def _run_specs_from_risk_manifest(manifest: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    if not manifest.exists():
        return runs
    df = pd.read_csv(manifest)
    label_map = {
        "teacher": "Teacher",
        "risk_pro_readable": "ProRisk",
        "risk_pro_plus": "Risk-Pro++",
        "pro": "PRO",
    }
    for _, row in df.iterrows():
        stage = str(row["stage"])
        if stage not in {"teacher", "risk_pro_readable"}:
            continue
        seed = int(row["seed"])
        runs.append(
            RunSpec(
                family="pro_risk",
                model=label_map.get(stage, stage),
                seed=seed,
                run_dir=_resolve_run_dir(row["run_dir"]),
                pair_group=f"pro_risk_seed{seed}",
            )
        )
    return runs


def _atlas_row(run: RunSpec, k: int) -> dict:
    train_emb, train_y, _ = _read_embeddings(run.run_dir, "train")
    emb, y, logits = _read_embeddings(run.run_dir, "test")
    pred_df = _prediction_frame(run.run_dir, "test")
    feats = _feature_frame(train_emb, train_y, emb, logits, k=k, split="test")

    metrics = _load_json(run.run_dir / "metrics.json")
    accuracy = _safe_float(metrics.get("accuracy", (pred_df["y_true"] == pred_df["y_pred"]).mean()))
    macro_f1 = _safe_float(metrics.get("macro_f1"))
    ece = _safe_float(metrics.get("ece"))
    total_errors = _safe_float(metrics.get("total_errors", pred_df["is_error"].sum()))
    vtvf_errors = _safe_float(metrics.get("vtvf_cross_errors", pred_df["is_vtvf_boundary_error"].sum()))

    if len(np.unique(y)) > 1:
        sil = float(silhouette_score(emb, y))
    else:
        sil = float("nan")
    centroids = np.stack([emb[y == c].mean(axis=0) for c in range(3)])
    within = [np.linalg.norm(emb[y == c] - centroids[c], axis=1).mean() for c in range(3)]
    vt_vf_sep = float(np.linalg.norm(centroids[VT] - centroids[VF]) / np.maximum((within[VT] + within[VF]) / 2, 1e-12))

    err = pred_df["is_error"].astype(int).to_numpy()
    boundary = pred_df["is_vtvf_boundary_error"].astype(int).to_numpy()
    vmask = pred_df["is_vtvf"].to_numpy()
    return {
        "family": run.family,
        "model": run.model,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "n_test": int(len(pred_df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "total_errors": total_errors,
        "vtvf_cross_errors": vtvf_errors,
        "vtvf_error_rate_within_vtvf": float(boundary[vmask].mean()) if vmask.any() else float("nan"),
        "mean_confidence": float(pred_df["confidence"].mean()),
        "mean_margin": float(pred_df["margin"].mean()),
        "mean_entropy": float(pred_df["entropy"].mean()),
        "embedding_silhouette": sil,
        "vt_vf_normalized_separation": vt_vf_sep,
        "mean_knn_vtvf_mix": float(feats.loc[vmask, "knn_vtvf_mix"].mean()) if vmask.any() else float("nan"),
        "mean_softmax_vtvf_ambiguity": float(pred_df.loc[vmask, "softmax_vtvf_ambiguity"].mean()) if vmask.any() else float("nan"),
        "entropy_error_auroc": _binary_metric(err, pred_df["entropy"].to_numpy(), roc_auc_score),
        "margin_error_auroc": _binary_metric(err, -pred_df["margin"].to_numpy(), roc_auc_score),
        "softmax_vtvf_boundary_auroc": _binary_metric(boundary[vmask], pred_df.loc[vmask, "softmax_vtvf_ambiguity"].to_numpy(), roc_auc_score)
        if vmask.any()
        else float("nan"),
        "knn_vtvf_boundary_auroc": _binary_metric(boundary[vmask], feats.loc[vmask, "knn_vtvf_mix"].to_numpy(), roc_auc_score)
        if vmask.any()
        else float("nan"),
    }


def _summarise_atlas(atlas: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in atlas.columns
        if col not in {"family", "model", "seed", "run_dir"} and pd.api.types.is_numeric_dtype(atlas[col])
    ]
    rows = []
    for (family, model), sub in atlas.groupby(["family", "model"], sort=False):
        for metric in metric_cols:
            vals = sub[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "metric": metric,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "cv_abs": float(vals.std(ddof=1) / np.maximum(abs(vals.mean()), 1e-12)) if len(vals) > 1 else 0.0,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def _atlas_correlations(atlas: pd.DataFrame) -> pd.DataFrame:
    candidate_x = [
        "embedding_silhouette",
        "vt_vf_normalized_separation",
        "mean_knn_vtvf_mix",
        "mean_softmax_vtvf_ambiguity",
        "entropy_error_auroc",
        "margin_error_auroc",
        "softmax_vtvf_boundary_auroc",
        "knn_vtvf_boundary_auroc",
    ]
    candidate_y = ["accuracy", "macro_f1", "ece", "total_errors", "vtvf_cross_errors"]
    rows = []
    for family, sub in atlas.groupby("family"):
        for x in candidate_x:
            for y in candidate_y:
                if x not in sub or y not in sub:
                    continue
                pair = sub[[x, y]].dropna()
                if len(pair) < 4:
                    continue
                rows.append(
                    {
                        "family": family,
                        "x": x,
                        "y": y,
                        "n": int(len(pair)),
                        "pearson_r": float(pair[x].corr(pair[y], method="pearson")),
                        "spearman_r": float(pair[x].corr(pair[y], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def _latent_strata_for_run(run: RunSpec, out_dir: Path, n_clusters: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_emb, train_y, _ = _read_embeddings(run.run_dir, "train")
    emb, y, logits = _read_embeddings(run.run_dir, "test")
    pred = _prediction_frame(run.run_dir, "test")
    feats = _feature_frame(train_emb, train_y, emb, logits, k=k, split="test")
    scaled = StandardScaler().fit_transform(emb)
    n_clusters_run = min(n_clusters, max(2, len(emb) // 30))
    labels = KMeans(n_clusters=n_clusters_run, n_init=20, random_state=run.seed).fit_predict(scaled)

    sample = pred[["y_true", "y_pred", "confidence", "margin", "entropy", "is_error", "is_vtvf_boundary_error"]].copy()
    sample["cluster"] = labels
    sample["family"] = run.family
    sample["model"] = run.model
    sample["seed"] = run.seed
    for col in ["knn_vtvf_mix", "knn_entropy", "prototype_vtvf_ambiguity", "softmax_vtvf_ambiguity", "ventricular_neighbor_fraction"]:
        sample[col] = feats[col].to_numpy()

    rows = []
    for cluster, sub in sample.groupby("cluster"):
        counts = sub["y_true"].value_counts(normalize=True)
        rows.append(
            {
                "family": run.family,
                "model": run.model,
                "seed": run.seed,
                "cluster": int(cluster),
                "n": int(len(sub)),
                "sr_frac": float(counts.get(0, 0.0)),
                "vt_frac": float(counts.get(1, 0.0)),
                "vf_frac": float(counts.get(2, 0.0)),
                "majority_class": CLASS_NAMES[int(sub["y_true"].mode().iloc[0])],
                "error_rate": float(sub["is_error"].mean()),
                "vtvf_boundary_error_rate": float(sub["is_vtvf_boundary_error"].mean()),
                "mean_confidence": float(sub["confidence"].mean()),
                "mean_entropy": float(sub["entropy"].mean()),
                "mean_margin": float(sub["margin"].mean()),
                "mean_knn_vtvf_mix": float(sub["knn_vtvf_mix"].mean()),
                "mean_proto_vtvf_ambiguity": float(sub["prototype_vtvf_ambiguity"].mean()),
                "mean_softmax_vtvf_ambiguity": float(sub["softmax_vtvf_ambiguity"].mean()),
                "high_conf_error_rate": float(((sub["confidence"] >= 0.90) & sub["is_error"]).mean()),
            }
        )
    cluster_summary = pd.DataFrame(rows)
    tag = f"{run.family}_{run.model.replace('-', '').replace('+', 'plus')}_seed{run.seed}"
    sample.to_csv(out_dir / f"latent_strata_samples_{tag}.csv", index=False)
    return sample, cluster_summary


def _validity_map_for_run(run: RunSpec, out_dir: Path, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_emb, train_y, train_logits = _read_embeddings(run.run_dir, "train")
    test_emb, test_y, test_logits = _read_embeddings(run.run_dir, "test")
    train_pred = _prediction_frame(run.run_dir, "train")
    test_pred = _prediction_frame(run.run_dir, "test")
    train_feat = _feature_frame(train_emb, train_y, train_emb, train_logits, k=k, split="train")
    test_feat = _feature_frame(train_emb, train_y, test_emb, test_logits, k=k, split="test")

    feature_cols = [
        "confidence",
        "margin",
        "entropy",
        "softmax_vtvf_ambiguity",
        "prototype_vtvf_ambiguity",
        "knn_entropy",
        "knn_vtvf_mix",
        "nearest_centroid_distance",
        "centroid_distance_margin",
        "ventricular_neighbor_fraction",
        "knn_pred_agreement",
    ]
    X_train = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    targets = {
        "any_error": (
            train_pred["is_error"].astype(int).to_numpy(),
            test_pred["is_error"].astype(int).to_numpy(),
            np.ones(len(test_pred), dtype=bool),
        ),
        "vtvf_boundary_error": (
            train_pred["is_vtvf_boundary_error"].astype(int).to_numpy(),
            test_pred["is_vtvf_boundary_error"].astype(int).to_numpy(),
            test_pred["is_vtvf"].to_numpy(),
        ),
        "confident_vtvf_boundary_error": (
            ((train_pred["is_vtvf_boundary_error"]) & (train_pred["confidence"] >= 0.90)).astype(int).to_numpy(),
            ((test_pred["is_vtvf_boundary_error"]) & (test_pred["confidence"] >= 0.90)).astype(int).to_numpy(),
            test_pred["is_vtvf"].to_numpy(),
        ),
    }

    score_rows = []
    curve_rows = []
    sample_scores = pd.DataFrame(
        {
            "y_true": test_pred["y_true"].to_numpy(),
            "y_pred": test_pred["y_pred"].to_numpy(),
            "is_error": test_pred["is_error"].to_numpy(),
            "is_vtvf_boundary_error": test_pred["is_vtvf_boundary_error"].to_numpy(),
            "confidence": test_pred["confidence"].to_numpy(),
        }
    )
    for target_name, (y_train_target, y_test_target, eval_mask) in targets.items():
        if len(np.unique(y_train_target)) < 2:
            risk = np.repeat(float(y_train_target.mean()), len(y_test_target))
        else:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=run.seed),
            )
            clf.fit(X_train, y_train_target)
            risk = clf.predict_proba(X_test)[:, 1]
        sample_scores[f"risk_{target_name}"] = risk

        mask = np.asarray(eval_mask, dtype=bool)
        y_eval = y_test_target[mask]
        risk_eval = risk[mask]
        auroc = _binary_metric(y_eval, risk_eval, roc_auc_score)
        aupr = _binary_metric(y_eval, risk_eval, average_precision_score)
        score_rows.append(
            {
                "family": run.family,
                "model": run.model,
                "seed": run.seed,
                "target": target_name,
                "n_eval": int(mask.sum()),
                "n_positive": int(y_eval.sum()),
                "prevalence": float(y_eval.mean()) if len(y_eval) else float("nan"),
                "validity_risk_auroc": auroc,
                "validity_risk_aupr": aupr,
            }
        )

        order = np.argsort(-risk_eval)
        for budget in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            n_review = max(1, int(round(len(order) * budget))) if len(order) else 0
            selected = order[:n_review]
            positives = int(y_eval.sum())
            captured = int(y_eval[selected].sum()) if n_review else 0
            curve_rows.append(
                {
                    "family": run.family,
                    "model": run.model,
                    "seed": run.seed,
                    "target": target_name,
                    "review_budget": budget,
                    "n_review": n_review,
                    "captured_positive": captured,
                    "capture_rate": float(captured / positives) if positives else float("nan"),
                    "precision": float(captured / n_review) if n_review else float("nan"),
                }
            )

    tag = f"{run.family}_{run.model.replace('-', '').replace('+', 'plus')}_seed{run.seed}"
    sample_scores.to_csv(out_dir / f"validity_scores_{tag}.csv", index=False)
    return pd.DataFrame(score_rows), pd.DataFrame(curve_rows)


def _disagreement_for_pair(left: RunSpec, right: RunSpec) -> tuple[dict, pd.DataFrame]:
    left_pred = _prediction_frame(left.run_dir, "test")
    right_pred = _prediction_frame(right.run_dir, "test")
    n = min(len(left_pred), len(right_pred))
    left_pred = left_pred.iloc[:n].reset_index(drop=True)
    right_pred = right_pred.iloc[:n].reset_index(drop=True)
    same_truth = (left_pred["y_true"].to_numpy() == right_pred["y_true"].to_numpy())
    if not same_truth.all():
        keep = same_truth
        left_pred = left_pred.loc[keep].reset_index(drop=True)
        right_pred = right_pred.loc[keep].reset_index(drop=True)
    y = left_pred["y_true"].to_numpy()
    left_error = left_pred["is_error"].to_numpy()
    right_error = right_pred["is_error"].to_numpy()
    any_error = left_error | right_error
    boundary = left_pred["is_vtvf_boundary_error"].to_numpy() | right_pred["is_vtvf_boundary_error"].to_numpy()
    disagree = left_pred["y_pred"].to_numpy() != right_pred["y_pred"].to_numpy()
    conf_gap = np.abs(left_pred["confidence"].to_numpy() - right_pred["confidence"].to_numpy())
    entropy_gap = np.abs(left_pred["entropy"].to_numpy() - right_pred["entropy"].to_numpy())
    soft_risk = disagree.astype(float) + 0.5 * conf_gap + 0.25 * entropy_gap
    vmask = np.isin(y, [VT, VF])

    row = {
        "family": left.family,
        "seed": left.seed,
        "left_model": left.model,
        "right_model": right.model,
        "n_aligned": int(len(y)),
        "same_truth_aligned": bool(same_truth.all()),
        "disagreement_rate": float(disagree.mean()),
        "any_model_error_rate": float(any_error.mean()),
        "vtvf_boundary_error_rate": float(boundary[vmask].mean()) if vmask.any() else float("nan"),
        "disagreement_any_error_auroc": _binary_metric(any_error.astype(int), soft_risk, roc_auc_score),
        "disagreement_vtvf_boundary_auroc": _binary_metric(boundary[vmask].astype(int), soft_risk[vmask], roc_auc_score)
        if vmask.any()
        else float("nan"),
        "error_capture_at_10pct": float(any_error[np.argsort(-soft_risk)[: max(1, int(round(0.10 * len(soft_risk))))]].sum() / np.maximum(any_error.sum(), 1)),
        "boundary_capture_at_10pct_vtvf": float(
            boundary[vmask][np.argsort(-soft_risk[vmask])[: max(1, int(round(0.10 * vmask.sum())))]].sum()
            / np.maximum(boundary[vmask].sum(), 1)
        )
        if vmask.any()
        else float("nan"),
        "both_agree_wrong": int(((~disagree) & any_error).sum()),
        "disagree_and_any_error": int((disagree & any_error).sum()),
        "left_only_correct": int(((~left_error) & right_error).sum()),
        "right_only_correct": int((left_error & (~right_error)).sum()),
    }
    score = pd.DataFrame(
        {
            "family": left.family,
            "seed": left.seed,
            "left_model": left.model,
            "right_model": right.model,
            "y_true": y,
            "left_pred": left_pred["y_pred"],
            "right_pred": right_pred["y_pred"],
            "disagree": disagree,
            "soft_disagreement_risk": soft_risk,
            "any_model_error": any_error,
            "any_vtvf_boundary_error": boundary,
        }
    )
    return row, score


def _paired_deltas(atlas: pd.DataFrame) -> pd.DataFrame:
    pair_defs = [
        ("cnn_lstm", "CNN", "CNN-LSTM"),
        ("pro_risk", "Teacher", "ProRisk"),
    ]
    metrics = [
        "accuracy",
        "macro_f1",
        "ece",
        "total_errors",
        "vtvf_cross_errors",
        "embedding_silhouette",
        "vt_vf_normalized_separation",
        "softmax_vtvf_boundary_auroc",
        "knn_vtvf_boundary_auroc",
    ]
    rows = []
    for family, baseline, comparator in pair_defs:
        sub = atlas[atlas["family"].eq(family)]
        base = sub[sub["model"].eq(baseline)]
        comp = sub[sub["model"].eq(comparator)]
        merged = base.merge(comp, on="seed", suffixes=("_baseline", "_comparator"))
        for _, row in merged.iterrows():
            out = {"family": family, "baseline": baseline, "comparator": comparator, "seed": int(row["seed"])}
            for metric in metrics:
                left = f"{metric}_baseline"
                right = f"{metric}_comparator"
                if left in row and right in row:
                    out[f"{metric}_delta"] = row[right] - row[left]
            rows.append(out)
    return pd.DataFrame(rows)


def _paired_delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, baseline, comparator), sub in deltas.groupby(["family", "baseline", "comparator"], sort=False):
        for col in sub.columns:
            if not col.endswith("_delta"):
                continue
            values = sub[col].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            lo, hi = _bootstrap_ci(values)
            rows.append(
                {
                    "family": family,
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": col,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_positive": int((values > 0).sum()),
                    "n_negative": int((values < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _plot_model_metric(summary: pd.DataFrame, metric: str, out: Path) -> None:
    sub = summary[summary["metric"].eq(metric)].copy()
    if sub.empty:
        return
    sub["label"] = sub["family"] + " / " + sub["model"]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(sub))
    y = sub["mean"].to_numpy()
    yerr = np.vstack([y - sub["bootstrap_ci_low"].to_numpy(), sub["bootstrap_ci_high"].to_numpy() - y])
    plt.bar(x, y, color="#4C78A8", alpha=0.82)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    plt.xticks(x, sub["label"], rotation=25, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def build_outputs(runs: list[RunSpec], out: Path, k: int, n_clusters: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    atlas = pd.DataFrame([_atlas_row(run, k=k) for run in runs])
    atlas.to_csv(out / "direction1_representation_underspecification_atlas.csv", index=False)
    atlas_summary = _summarise_atlas(atlas)
    atlas_summary.to_csv(out / "direction1_atlas_model_summary.csv", index=False)
    correlations = _atlas_correlations(atlas)
    correlations.to_csv(out / "direction1_representation_boundary_correlations.csv", index=False)
    deltas = _paired_deltas(atlas)
    deltas.to_csv(out / "direction1_paired_model_deltas.csv", index=False)
    delta_summary = _paired_delta_summary(deltas)
    delta_summary.to_csv(out / "direction1_paired_delta_summary.csv", index=False)
    for metric in ["accuracy", "vtvf_cross_errors", "embedding_silhouette", "knn_vtvf_boundary_auroc"]:
        _plot_model_metric(atlas_summary, metric, out / "figures" / f"atlas_{metric}.png")

    samples = []
    clusters = []
    validity_scores = []
    validity_curves = []
    for run in runs:
        sample, cluster_summary = _latent_strata_for_run(run, out, n_clusters=n_clusters, k=k)
        samples.append(sample)
        clusters.append(cluster_summary)
        scores, curves = _validity_map_for_run(run, out, k=k)
        validity_scores.append(scores)
        validity_curves.append(curves)
    latent_clusters = pd.concat(clusters, ignore_index=True)
    latent_clusters.to_csv(out / "direction2_hidden_rhythm_strata_clusters.csv", index=False)
    latent_cluster_summary = (
        latent_clusters.groupby(["family", "model"], as_index=False)
        .agg(
            n_clusters=("cluster", "count"),
            mean_cluster_error_rate=("error_rate", "mean"),
            max_cluster_error_rate=("error_rate", "max"),
            mean_cluster_vtvf_boundary_error_rate=("vtvf_boundary_error_rate", "mean"),
            max_cluster_vtvf_boundary_error_rate=("vtvf_boundary_error_rate", "max"),
            mean_high_conf_error_rate=("high_conf_error_rate", "mean"),
            max_high_conf_error_rate=("high_conf_error_rate", "max"),
        )
    )
    latent_cluster_summary.to_csv(out / "direction2_hidden_rhythm_strata_summary.csv", index=False)

    validity_score_df = pd.concat(validity_scores, ignore_index=True)
    validity_curve_df = pd.concat(validity_curves, ignore_index=True)
    validity_score_df.to_csv(out / "direction3_validity_domain_scores.csv", index=False)
    validity_curve_df.to_csv(out / "direction3_validity_domain_review_curves.csv", index=False)
    validity_summary = (
        validity_score_df.groupby(["family", "model", "target"], as_index=False)
        .agg(
            n=("seed", "count"),
            mean_positive=("n_positive", "mean"),
            prevalence_mean=("prevalence", "mean"),
            auroc_mean=("validity_risk_auroc", "mean"),
            auroc_std=("validity_risk_auroc", "std"),
            aupr_mean=("validity_risk_aupr", "mean"),
            aupr_std=("validity_risk_aupr", "std"),
        )
    )
    validity_summary.to_csv(out / "direction3_validity_domain_summary.csv", index=False)

    pair_rows = []
    score_frames = []
    for family, left_model, right_model in [("cnn_lstm", "CNN", "CNN-LSTM"), ("pro_risk", "Teacher", "ProRisk")]:
        sub = {int(run.seed): run for run in runs if run.family == family and run.model == left_model}
        other = {int(run.seed): run for run in runs if run.family == family and run.model == right_model}
        for seed in sorted(set(sub) & set(other)):
            row, scores = _disagreement_for_pair(sub[seed], other[seed])
            pair_rows.append(row)
            score_frames.append(scores)
    disagreement = pd.DataFrame(pair_rows)
    disagreement.to_csv(out / "direction4_model_disagreement_second_opinion.csv", index=False)
    if score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(out / "direction4_model_disagreement_scores.csv", index=False)
    disagreement_summary = (
        disagreement.groupby(["family", "left_model", "right_model"], as_index=False)
        .agg(
            n=("seed", "count"),
            disagreement_rate_mean=("disagreement_rate", "mean"),
            any_error_auroc_mean=("disagreement_any_error_auroc", "mean"),
            vtvf_boundary_auroc_mean=("disagreement_vtvf_boundary_auroc", "mean"),
            error_capture_at_10pct_mean=("error_capture_at_10pct", "mean"),
            boundary_capture_at_10pct_vtvf_mean=("boundary_capture_at_10pct_vtvf", "mean"),
            both_agree_wrong_mean=("both_agree_wrong", "mean"),
        )
    )
    disagreement_summary.to_csv(out / "direction4_model_disagreement_summary.csv", index=False)

    report = {
        "n_runs": len(runs),
        "n_seeds_by_family": atlas.groupby("family")["seed"].nunique().to_dict(),
        "outputs": {
            "direction1_atlas": str(out / "direction1_representation_underspecification_atlas.csv"),
            "direction2_strata": str(out / "direction2_hidden_rhythm_strata_clusters.csv"),
            "direction3_validity": str(out / "direction3_validity_domain_summary.csv"),
            "direction4_disagreement": str(out / "direction4_model_disagreement_summary.csv"),
        },
    }
    (out / "top_journal_reliability_direction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four mechanism-guided reliability analyses across CNN, CNN-LSTM, and ProRisk.")
    parser.add_argument("--cnn-lstm-root", type=Path, default=Path("results/cnn_lstm_baseline_20260626"))
    parser.add_argument(
        "--risk-manifest",
        type=Path,
        default=Path("results/risk_pro_readable_10seed_20260626/risk_pro_readable_manifest_20260626_194754.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/top_journal_reliability_directions_20260626"))
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    runs = _run_specs_from_cnn_lstm(args.cnn_lstm_root) + _run_specs_from_risk_manifest(args.risk_manifest)
    runs = [run for run in runs if (run.run_dir / "embeddings_test.npz").exists() and (run.run_dir / "embeddings_train.npz").exists()]
    if not runs:
        raise SystemExit("No valid runs found.")
    report = build_outputs(runs, out=args.out, k=args.k, n_clusters=args.clusters)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
