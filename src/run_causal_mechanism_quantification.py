from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, davies_bouldin_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .metrics import softmax


LOWER_IS_BETTER_OUTCOMES = {"ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"}
OUTCOMES = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]
PAIR_METADATA_COLS = [
    "role",
    "target_mechanism",
    "target_variables_json",
    "candidate_args_json",
    "hypothesis",
]


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def _safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(average_precision_score(y[mask], s[mask]))


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(p * np.log(p + eps), axis=1)


def _normalise(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _load_metrics(run_dir: Path) -> dict[str, float]:
    raw = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = {k: float(raw[k]) for k in ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors"] if k in raw}
    metrics["error_migration_penalty"] = (
        float(raw.get("vt_as_vf", 0.0))
        + float(raw.get("vf_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vf", 0.0))
    )
    return metrics


def _norm_dist(emb: np.ndarray, y: np.ndarray, i: int, j: int) -> float:
    centroids = np.stack([emb[y == c].mean(axis=0) for c in range(3)])
    within = [np.linalg.norm(emb[y == c] - centroids[c], axis=1).mean() for c in range(3)]
    return float(np.linalg.norm(centroids[i] - centroids[j]) / max((within[i] + within[j]) / 2.0, 1e-12))


def _geometry_metrics(train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray, y: np.ndarray, pred: np.ndarray, k: int) -> dict[str, float]:
    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(test_emb[:, None, :] - centroids[None, :, :], axis=2)
    d_vt, d_vf = dist[:, 1], dist[:, 2]
    prototype_vtvf_ambiguity = 1.0 - np.abs(d_vt - d_vf) / np.maximum(d_vt + d_vf, 1e-12)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_emb)
    knn_dist, idx = nn.kneighbors(test_emb)
    neigh_y = train_y[idx]
    label_probs = np.stack([(neigh_y == c).mean(axis=1) for c in range(3)], axis=1)
    knn_label_entropy = _entropy(label_probs) / np.log(3)
    knn_vtvf_mix = 1.0 - np.abs(label_probs[:, 1] - label_probs[:, 2]) / np.maximum(
        label_probs[:, 1] + label_probs[:, 2], 1e-12
    )
    knn_vtvf_mix[(label_probs[:, 1] + label_probs[:, 2]) == 0] = 0.0

    local_purity = (neigh_y == y[:, None]).mean(axis=1)
    is_error = y != pred
    is_vtvf = np.isin(y, [1, 2])
    is_vtvf_cross_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))

    out = {
        "silhouette_full": float(silhouette_score(test_emb, y)),
        "davies_bouldin_full": float(davies_bouldin_score(test_emb, y)),
        "sr_vt_norm_dist": _norm_dist(test_emb, y, 0, 1),
        "sr_vf_norm_dist": _norm_dist(test_emb, y, 0, 2),
        "vt_vf_norm_dist": _norm_dist(test_emb, y, 1, 2),
        "local_purity_k_mean": float(local_purity.mean()),
        "knn_distance_mean": float(knn_dist.mean()),
        "knn_label_entropy_mean": float(knn_label_entropy.mean()),
        "knn_vtvf_mix_ventricular_mean": float(knn_vtvf_mix[is_vtvf].mean()) if is_vtvf.any() else float("nan"),
        "prototype_vtvf_ambiguity_ventricular_mean": float(prototype_vtvf_ambiguity[is_vtvf].mean())
        if is_vtvf.any()
        else float("nan"),
        "error_local_purity_mean": float(local_purity[is_error].mean()) if is_error.any() else float("nan"),
        "correct_local_purity_mean": float(local_purity[~is_error].mean()) if (~is_error).any() else float("nan"),
        "vtvf_error_knn_mix_mean": float(knn_vtvf_mix[is_vtvf_cross_error].mean())
        if is_vtvf_cross_error.any()
        else float("nan"),
        "correct_vtvf_knn_mix_mean": float(knn_vtvf_mix[is_vtvf & ~is_vtvf_cross_error].mean())
        if (is_vtvf & ~is_vtvf_cross_error).any()
        else float("nan"),
        "prototype_vtvf_ambiguity_auroc": _safe_auc(is_vtvf_cross_error[is_vtvf], prototype_vtvf_ambiguity[is_vtvf])
        if is_vtvf.any()
        else float("nan"),
        "knn_vtvf_mix_auroc": _safe_auc(is_vtvf_cross_error[is_vtvf], knn_vtvf_mix[is_vtvf]) if is_vtvf.any() else float("nan"),
        "knn_label_entropy_any_error_auroc": _safe_auc(is_error, knn_label_entropy),
    }
    return out


def _ambiguity_metrics(logits: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)
    is_error = y != pred
    is_vtvf = np.isin(y, [1, 2])
    is_vtvf_cross_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    softmax_vtvf_ambiguity = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(probs[:, 1] + probs[:, 2], 1e-12)
    confidence = probs.max(axis=1)
    entropy = _entropy(probs) / np.log(probs.shape[1])
    margin = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
    return {
        "confidence_mean": float(confidence.mean()),
        "entropy_mean": float(entropy.mean()),
        "prob_margin_mean": float(margin.mean()),
        "softmax_vtvf_ambiguity_ventricular_mean": float(softmax_vtvf_ambiguity[is_vtvf].mean())
        if is_vtvf.any()
        else float("nan"),
        "softmax_vtvf_ambiguity_auroc": _safe_auc(is_vtvf_cross_error[is_vtvf], softmax_vtvf_ambiguity[is_vtvf])
        if is_vtvf.any()
        else float("nan"),
        "entropy_any_error_auroc": _safe_auc(is_error, entropy),
        "low_margin_any_error_auroc": _safe_auc(is_error, -margin),
    }


def _validity_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "validity_gate_scores_test.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    y = df["y_true"].to_numpy(int)
    pred = df["y_pred"].to_numpy(int)
    is_error = y != pred
    is_vtvf = np.isin(y, [1, 2])
    is_vtvf_cross_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    gate = df["validity_gate"].to_numpy(float)
    boundary = df["boundary_score"].to_numpy(float)
    gate_x_boundary = gate * boundary
    return {
        "validity_gate_mean": float(np.mean(gate)),
        "boundary_score_mean": float(np.mean(boundary)),
        "gate_x_boundary_mean": float(np.mean(gate_x_boundary)),
        "boundary_score_any_error_auroc": _safe_auc(is_error, boundary),
        "validity_gate_any_error_auroc": _safe_auc(is_error, -gate),
        "gate_x_boundary_any_error_auroc": _safe_auc(is_error, gate_x_boundary),
        "boundary_score_vtvf_cross_auroc": _safe_auc(is_vtvf_cross_error[is_vtvf], boundary[is_vtvf])
        if is_vtvf.any()
        else float("nan"),
        "gate_x_boundary_vtvf_cross_auroc": _safe_auc(is_vtvf_cross_error[is_vtvf], gate_x_boundary[is_vtvf])
        if is_vtvf.any()
        else float("nan"),
    }


def _run_mechanism_metrics(run_dir: Path, k: int) -> dict[str, float]:
    train = np.load(run_dir / "embeddings_train.npz")
    test = np.load(run_dir / "embeddings_test.npz")
    train_emb = train["embeddings"].astype(np.float32)
    train_y = train["y"].astype(np.int64)
    test_emb = test["embeddings"].astype(np.float32)
    y = test["y"].astype(np.int64)
    logits = test["logits"].astype(np.float32)
    pred = softmax(logits).argmax(axis=1)
    out: dict[str, float] = {}
    out.update(_geometry_metrics(train_emb, train_y, test_emb, y, pred, k))
    out.update(_ambiguity_metrics(logits, y, pred))
    out.update(_validity_metrics(run_dir))
    return out


def _load_manifest(search_dir: Path, manifest: Path | None) -> pd.DataFrame:
    if manifest is None:
        candidates = sorted(search_dir.glob("model_layer_causal_pareto_search_manifest_*.csv"))
        candidates += sorted(search_dir.glob("mechanism_targeted_ablation_manifest_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No manifest found under {search_dir}")
        manifest = candidates[-1]
    df = pd.read_csv(manifest)
    df["run_dir"] = df["run_dir"].map(lambda x: str(Path(str(x))))
    return df[df["status"].eq("completed")].copy()


def _paired_deltas(run_level: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    baseline = run_level[run_level["candidate"].eq("baseline")].set_index("seed")
    rows: list[dict[str, Any]] = []
    for _, row in run_level[~run_level["candidate"].eq("baseline")].iterrows():
        seed = row["seed"]
        if seed not in baseline.index:
            continue
        base = baseline.loc[seed]
        item: dict[str, Any] = {
            "candidate": row["candidate"],
            "seed": seed,
        }
        for col in PAIR_METADATA_COLS:
            if col in row.index:
                item[col] = row[col]
        for col in value_cols:
            item[f"{col}_baseline"] = base[col]
            item[f"{col}_candidate"] = row[col]
            item[f"{col}_delta"] = row[col] - base[col]
        rows.append(item)
    return pd.DataFrame(rows)


def _metric_family(metric: str) -> str:
    if any(token in metric for token in ["knn", "local_purity", "silhouette", "davies", "norm_dist", "prototype"]):
        return "representation_geometry_knn_prototype"
    if any(token in metric for token in ["softmax", "entropy", "margin", "confidence"]):
        return "softmax_boundary_confidence"
    if any(token in metric for token in ["validity", "gate", "boundary_score"]):
        return "validity_boundary"
    return "other_mechanism"


def _association_rows(deltas: pd.DataFrame, mechanism_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mech in mechanism_cols:
        x = pd.to_numeric(deltas[f"{mech}_delta"], errors="coerce")
        for outcome in OUTCOMES:
            y = pd.to_numeric(deltas[f"{outcome}_delta"], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 4:
                continue
            sx, sy = x[mask].to_numpy(float), y[mask].to_numpy(float)
            sp = spearmanr(sx, sy)
            pe = pearsonr(sx, sy)
            rows.append(
                {
                    "mechanism_variable": mech,
                    "mechanism_family": _metric_family(mech),
                    "outcome": outcome,
                    "n_paired_candidate_seed_rows": int(mask.sum()),
                    "spearman_r": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                    "pearson_r": float(pe.statistic),
                    "pearson_p": float(pe.pvalue),
                    "outcome_lower_is_better": outcome in LOWER_IS_BETTER_OUTCOMES,
                    "interpretation": "negative association helps outcome"
                    if outcome in LOWER_IS_BETTER_OUTCOMES
                    else "positive association helps outcome",
                }
            )
    columns = [
        "mechanism_variable",
        "mechanism_family",
        "outcome",
        "n_paired_candidate_seed_rows",
        "spearman_r",
        "spearman_p",
        "pearson_r",
        "pearson_p",
        "outcome_lower_is_better",
        "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns)


def _intervention_summary(deltas: pd.DataFrame, mechanism_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = deltas.groupby("candidate", sort=True)
    for candidate, sub in grouped:
        for mech in mechanism_cols:
            values = pd.to_numeric(sub[f"{mech}_delta"], errors="coerce")
            rows.append(
                {
                    "candidate": candidate,
                    "role": sub["role"].iloc[0] if "role" in sub.columns else "",
                    "target_mechanism": sub["target_mechanism"].iloc[0] if "target_mechanism" in sub.columns else "",
                    "target_variables_json": sub["target_variables_json"].iloc[0]
                    if "target_variables_json" in sub.columns
                    else "",
                    "mechanism_variable": mech,
                    "mechanism_family": _metric_family(mech),
                    "n_paired_seeds": int(values.notna().sum()),
                    "mechanism_delta_mean": float(values.mean()),
                    "mechanism_delta_std": float(values.std()) if values.notna().sum() > 1 else float("nan"),
                    "positive_delta_seed_count": int((values > 0).sum()),
                    "negative_delta_seed_count": int((values < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _path_summary(
    intervention_effects: pd.DataFrame,
    associations: pd.DataFrame,
    outcome_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "candidate",
        "target_mechanism",
        "target_variables_json",
        "mechanism_family",
        "mechanism_variable",
        "mechanism_delta_mean",
        "linked_outcome",
        "outcome_delta_mean",
        "mechanism_outcome_spearman_r",
        "mechanism_outcome_spearman_p",
        "path_product_proxy",
        "candidate_outcome_improved",
        "evidence_type",
    ]
    if associations.empty or "spearman_r" not in associations.columns:
        return pd.DataFrame(columns=columns)
    best_assoc = associations.copy()
    best_assoc["abs_spearman_r"] = best_assoc["spearman_r"].abs()
    best_assoc = best_assoc.sort_values("abs_spearman_r", ascending=False)
    rows: list[dict[str, Any]] = []
    for _, eff in intervention_effects.iterrows():
        sub = best_assoc[best_assoc["mechanism_variable"].eq(eff["mechanism_variable"])].head(3)
        for _, assoc in sub.iterrows():
            out_match = outcome_summary[
                (outcome_summary["candidate"].eq(eff["candidate"])) & (outcome_summary["outcome"].eq(assoc["outcome"]))
            ]
            if out_match.empty:
                continue
            outcome_delta = float(out_match["outcome_delta_mean"].iloc[0])
            beneficial = outcome_delta <= 0 if assoc["outcome"] in LOWER_IS_BETTER_OUTCOMES else outcome_delta >= 0
            rows.append(
                {
                    "candidate": eff["candidate"],
                    "target_mechanism": eff.get("target_mechanism", ""),
                    "target_variables_json": eff.get("target_variables_json", ""),
                    "mechanism_family": eff["mechanism_family"],
                    "mechanism_variable": eff["mechanism_variable"],
                    "mechanism_delta_mean": eff["mechanism_delta_mean"],
                    "linked_outcome": assoc["outcome"],
                    "outcome_delta_mean": outcome_delta,
                    "mechanism_outcome_spearman_r": assoc["spearman_r"],
                    "mechanism_outcome_spearman_p": assoc["spearman_p"],
                    "path_product_proxy": float(eff["mechanism_delta_mean"]) * float(assoc["spearman_r"]),
                    "candidate_outcome_improved": bool(beneficial),
                    "evidence_type": "paired internal do-intervention proxy, not external causal proof",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _outcome_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, sub in deltas.groupby("candidate", sort=True):
        for outcome in OUTCOMES:
            values = pd.to_numeric(sub[f"{outcome}_delta"], errors="coerce")
            rows.append(
                {
                    "candidate": candidate,
                    "outcome": outcome,
                    "outcome_delta_mean": float(values.mean()),
                    "outcome_delta_std": float(values.std()) if values.notna().sum() > 1 else float("nan"),
                    "good_direction_seed_count": int((values <= 0).sum())
                    if outcome in LOWER_IS_BETTER_OUTCOMES
                    else int((values >= 0).sum()),
                    "n_paired_seeds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _variable_dictionary(mechanism_cols: list[str]) -> pd.DataFrame:
    rows = [
        {
            "variable": "candidate",
            "role": "intervenable",
            "level": "model_training",
            "definition": "do(training constraint/weight configuration)",
            "can_intervene_in_experiment": True,
        },
        {
            "variable": "seed",
            "role": "design/control",
            "level": "split_training_randomness",
            "definition": "paired seed used to compare candidate with same-seed baseline",
            "can_intervene_in_experiment": True,
        },
    ]
    for mech in mechanism_cols:
        rows.append(
            {
                "variable": mech,
                "role": "mechanism/mediator",
                "level": _metric_family(mech),
                "definition": "post-training internal evidence variable computed from logits, embeddings, kNN, prototype distances, or validity gate",
                "can_intervene_in_experiment": False,
            }
        )
    for outcome in OUTCOMES:
        rows.append(
            {
                "variable": outcome,
                "role": "outcome",
                "level": "model_performance",
                "definition": "test-set model-layer outcome",
                "can_intervene_in_experiment": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify do(training constraint) -> mechanism variable -> outcome evidence chains."
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path("results/model_layer_causal_pareto_search_full_20260630"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/causal_mechanism_quantification_20260630"))
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.search_dir, args.manifest)

    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = Path(str(row["run_dir"]))
        if not run_dir.exists():
            run_dir = Path.cwd() / run_dir
        metrics = _load_metrics(run_dir)
        mechanisms = _run_mechanism_metrics(run_dir, args.k)
        rows.append({**row.to_dict(), **metrics, **mechanisms})

    run_level = pd.DataFrame(rows)
    mechanism_cols = [
        col
        for col in run_level.columns
        if col
        not in set(manifest.columns)
        | set(OUTCOMES)
        | {"model", "epochs", "status", "teacher_run_dir", "risk_targets", "run_dir"}
    ]
    run_level.to_csv(args.out / "run_level_mechanism_outcome_table.csv", index=False)

    deltas = _paired_deltas(run_level, mechanism_cols + OUTCOMES)
    deltas.to_csv(args.out / "paired_candidate_seed_mechanism_outcome_deltas.csv", index=False)

    intervention_effects = _intervention_summary(deltas, mechanism_cols)
    intervention_effects.to_csv(args.out / "intervention_to_mechanism_effects.csv", index=False)

    outcome_summary = _outcome_summary(deltas)
    outcome_summary.to_csv(args.out / "intervention_to_outcome_effects.csv", index=False)

    associations = _association_rows(deltas, mechanism_cols)
    associations.to_csv(args.out / "mechanism_to_outcome_association.csv", index=False)

    path_summary = _path_summary(intervention_effects, associations, outcome_summary)
    path_summary.to_csv(args.out / "mediation_or_path_effect_summary.csv", index=False)

    variable_dictionary = _variable_dictionary(mechanism_cols)
    variable_dictionary.to_csv(args.out / "causal_mechanism_variable_dictionary.csv", index=False)

    report = {
        "search_dir": str(args.search_dir),
        "manifest": str(args.manifest) if args.manifest is not None else "auto",
        "n_run_level_rows": int(len(run_level)),
        "n_paired_delta_rows": int(len(deltas)),
        "n_seeds": int(run_level["seed"].nunique()) if "seed" in run_level.columns else 0,
        "n_candidates": int(run_level["candidate"].nunique()) if "candidate" in run_level.columns else 0,
        "n_mechanism_variables": int(len(mechanism_cols)),
        "n_association_rows": int(len(associations)),
        "outputs": {
            "run_level": str(args.out / "run_level_mechanism_outcome_table.csv"),
            "paired_deltas": str(args.out / "paired_candidate_seed_mechanism_outcome_deltas.csv"),
            "intervention_to_mechanism": str(args.out / "intervention_to_mechanism_effects.csv"),
            "intervention_to_outcome": str(args.out / "intervention_to_outcome_effects.csv"),
            "mechanism_to_outcome": str(args.out / "mechanism_to_outcome_association.csv"),
            "path_summary": str(args.out / "mediation_or_path_effect_summary.csv"),
            "variable_dictionary": str(args.out / "causal_mechanism_variable_dictionary.csv"),
        },
        "limitations": [
            "Internal paired do-intervention evidence only; no external validation dataset.",
            "Mechanism variables are post-training mediators/diagnostics, so path effects are causal proxies rather than formal randomized mediation proof.",
            f"{int(run_level['seed'].nunique()) if 'seed' in run_level.columns else 0} seed(s) are available in this quantification run.",
        ],
    }
    (args.out / "causal_mechanism_quantification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
