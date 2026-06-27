from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_run(values: list[str]) -> dict:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--run expects: SEED MODEL RUN_DIR")
    seed, model, run_dir = values
    return {"seed": int(seed), "model": model, "run_dir": Path(run_dir)}


def _load_csv(run: dict, name: str) -> pd.DataFrame:
    path = run["run_dir"] / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_dir", str(run["run_dir"]))
    df.insert(0, "model", run["model"])
    df.insert(0, "seed", run["seed"])
    return df


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    return hsic / max(norm_x * norm_y, 1e-12)


def _svd_project(x: np.ndarray, variance: float = 0.99, max_components: int = 50) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if s.size == 0:
        return x
    explained = np.cumsum(s**2) / max(float(np.sum(s**2)), 1e-12)
    n_components = int(np.searchsorted(explained, variance) + 1)
    n_components = max(1, min(n_components, max_components, vt.shape[0]))
    return x @ vt[:n_components].T


def _invsqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _svcca(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x_proj = _svd_project(x)
    y_proj = _svd_project(y)
    n = min(len(x_proj), len(y_proj))
    if n < 3:
        return np.nan, np.nan, 0
    x_proj = x_proj[:n] - x_proj[:n].mean(axis=0, keepdims=True)
    y_proj = y_proj[:n] - y_proj[:n].mean(axis=0, keepdims=True)
    denom = max(n - 1, 1)
    cxx = (x_proj.T @ x_proj) / denom
    cyy = (y_proj.T @ y_proj) / denom
    cxy = (x_proj.T @ y_proj) / denom
    corr = np.linalg.svd(_invsqrt(cxx) @ cxy @ _invsqrt(cyy), compute_uv=False)
    corr = np.clip(corr, 0.0, 1.0)
    top_k = min(5, len(corr))
    return float(corr.mean()), float(corr[:top_k].mean()), int(len(corr))


def _aligned_arrays(left_path: Path, right_path: Path, rep: str) -> tuple[np.ndarray, np.ndarray, int]:
    left = np.load(left_path / "advanced_test_representations.npz", allow_pickle=True)
    right = np.load(right_path / "advanced_test_representations.npz", allow_pickle=True)
    left_hashes = [str(x) for x in left["sample_hash"]]
    right_hashes = [str(x) for x in right["sample_hash"]]
    right_index = {h: i for i, h in enumerate(right_hashes)}
    pairs = [(i, right_index[h]) for i, h in enumerate(left_hashes) if h in right_index]
    if len(pairs) < 20:
        return np.empty((0, 0)), np.empty((0, 0)), len(pairs)
    li, ri = zip(*pairs)
    return left[rep][list(li)], right[rep][list(ri)], len(pairs)


def _cka_rows(runs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    reps = ["final_embedding", "classifier_logits", "regularity_features"]
    for left, right in combinations(runs, 2):
        same_seed = left["seed"] == right["seed"]
        same_model = left["model"] == right["model"]
        if not (same_seed or same_model):
            continue
        for rep in reps:
            left_arr, right_arr, n_common = _aligned_arrays(left["run_dir"], right["run_dir"], rep)
            if len(left_arr) == 0:
                continue
            svcca_mean, svcca_top5_mean, svcca_components = _svcca(left_arr, right_arr)
            rows.append(
                {
                    "left_seed": left["seed"],
                    "left_model": left["model"],
                    "right_seed": right["seed"],
                    "right_model": right["model"],
                    "comparison_type": "same_seed_model_pair" if same_seed else "same_model_cross_seed",
                    "representation": rep,
                    "n_common_samples": n_common,
                    "linear_cka": _linear_cka(left_arr, right_arr),
                    "svcca_mean_corr": svcca_mean,
                    "svcca_top5_mean_corr": svcca_top5_mean,
                    "svcca_components": svcca_components,
                }
            )
    return rows


def _metric_table(linear: pd.DataFrame, distribution: pd.DataFrame, perturb: pd.DataFrame, concept: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if not linear.empty:
        sub = linear[linear["probe"].eq("vt_vs_vf_binary") & linear["representation"].eq("final_embedding")]
        for _, row in sub.iterrows():
            rows.append({"seed": row["seed"], "model": row["model"], "metric": "vtvf_probe_auroc", "value": row.get("auroc")})
        sub = linear[linear["probe"].eq("sr_vt_vf_multiclass") & linear["representation"].eq("final_embedding")]
        for _, row in sub.iterrows():
            rows.append({"seed": row["seed"], "model": row["model"], "metric": "embedding_probe_macro_f1", "value": row.get("macro_f1")})
            rows.append({"seed": row["seed"], "model": row["model"], "metric": "embedding_probe_vtvf_cross_errors", "value": row.get("vtvf_cross_errors")})

    if not distribution.empty:
        sub = distribution[
            distribution["representation"].eq("final_embedding") & distribution["scope"].eq("VT_vs_VF")
        ]
        for _, row in sub.iterrows():
            rows.append({"seed": row["seed"], "model": row["model"], "metric": "vtvf_mahalanobis_distance", "value": row.get("mahalanobis_centroid_distance")})
            rows.append({"seed": row["seed"], "model": row["model"], "metric": "vtvf_fisher_ratio", "value": row.get("fisher_ratio")})

    if not perturb.empty:
        grouped = perturb.groupby(["seed", "model"], as_index=False).agg(
            embedding_shift_mean=("embedding_shift_mean", "mean"),
            prediction_flip_rate=("prediction_flip_rate", "mean"),
            prototype_flip_rate=("prototype_flip_rate", "mean"),
            neighbor_jaccard_mean=("neighbor_jaccard_mean", "mean"),
        )
        for _, row in grouped.iterrows():
            for metric in ["embedding_shift_mean", "prediction_flip_rate", "prototype_flip_rate", "neighbor_jaccard_mean"]:
                rows.append({"seed": row["seed"], "model": row["model"], "metric": f"perturb_{metric}", "value": row[metric]})

    if not concept.empty:
        sub = concept[concept["representation"].eq("final_embedding")]
        grouped = sub.groupby(["seed", "model"], as_index=False).agg(
            concept_max_corr=("max_abs_dim_correlation", "max"),
            concept_mean_r2=("ridge_r2_from_representation", "mean"),
            concept_best_r2=("ridge_r2_from_representation", "max"),
        )
        for _, row in grouped.iterrows():
            for metric in ["concept_max_corr", "concept_mean_r2", "concept_best_r2"]:
                rows.append({"seed": row["seed"], "model": row["model"], "metric": metric, "value": row[metric]})

    return pd.DataFrame(rows)


def _stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (model, metric), sub in metrics.groupby(["model", "metric"]):
        values = sub["value"].astype(float).dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": mean,
                "std": std,
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "cv_abs": float(std / max(abs(mean), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _paired_deltas(metrics: pd.DataFrame, baseline: str, comparator: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for metric, sub in metrics.groupby("metric"):
        base = sub[sub["model"].eq(baseline)].set_index("seed")
        comp = sub[sub["model"].eq(comparator)].set_index("seed")
        for seed in sorted(set(base.index) & set(comp.index)):
            rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "baseline": baseline,
                    "comparator": comparator,
                    "delta": float(comp.loc[seed, "value"] - base.loc[seed, "value"]),
                    "baseline_value": float(base.loc[seed, "value"]),
                    "comparator_value": float(comp.loc[seed, "value"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate advanced representation diagnostics across ECG runs.")
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("SEED", "MODEL", "RUN_DIR"))
    parser.add_argument("--baseline", default="CNN")
    parser.add_argument("--comparator", default="CNN-LSTM")
    parser.add_argument("--out", type=Path, default=Path("results/advanced_representation_diagnostics"))
    args = parser.parse_args()

    runs = [_parse_run(item) for item in args.run]
    args.out.mkdir(parents=True, exist_ok=True)

    linear = pd.concat([_load_csv(run, "advanced_linear_probe_summary.csv") for run in runs], ignore_index=True)
    distribution = pd.concat([_load_csv(run, "advanced_distribution_geometry.csv") for run in runs], ignore_index=True)
    perturb = pd.concat(
        [_load_csv(run, "advanced_perturbation_representation_stability.csv") for run in runs], ignore_index=True
    )
    concept = pd.concat([_load_csv(run, "advanced_regularity_concept_alignment.csv") for run in runs], ignore_index=True)
    cka = pd.DataFrame(_cka_rows(runs))
    metrics = _metric_table(linear, distribution, perturb, concept)
    stability = _stability_summary(metrics)
    deltas = _paired_deltas(metrics, args.baseline, args.comparator)
    delta_summary = (
        deltas.groupby("metric", as_index=False)
        .agg(mean_delta=("delta", "mean"), median_delta=("delta", "median"), n=("delta", "count"))
        if not deltas.empty
        else pd.DataFrame()
    )

    linear.to_csv(args.out / "advanced_linear_probe_all_runs.csv", index=False)
    distribution.to_csv(args.out / "advanced_distribution_geometry_all_runs.csv", index=False)
    perturb.to_csv(args.out / "advanced_perturbation_stability_all_runs.csv", index=False)
    concept.to_csv(args.out / "advanced_concept_alignment_all_runs.csv", index=False)
    cka.to_csv(args.out / "advanced_representation_cka.csv", index=False)
    metrics.to_csv(args.out / "advanced_six_target_metric_table.csv", index=False)
    stability.to_csv(args.out / "advanced_cross_seed_stability_summary.csv", index=False)
    deltas.to_csv(args.out / "advanced_paired_metric_deltas.csv", index=False)
    delta_summary.to_csv(args.out / "advanced_paired_delta_summary.csv", index=False)

    report = {
        "n_runs": len(runs),
        "runs": [{"seed": r["seed"], "model": r["model"], "run_dir": str(r["run_dir"])} for r in runs],
        "outputs": {
            "linear": str(args.out / "advanced_linear_probe_all_runs.csv"),
            "distribution": str(args.out / "advanced_distribution_geometry_all_runs.csv"),
            "perturbation": str(args.out / "advanced_perturbation_stability_all_runs.csv"),
            "concept": str(args.out / "advanced_concept_alignment_all_runs.csv"),
            "cka": str(args.out / "advanced_representation_cka.csv"),
            "stability": str(args.out / "advanced_cross_seed_stability_summary.csv"),
            "paired_deltas": str(args.out / "advanced_paired_metric_deltas.csv"),
        },
    }
    (args.out / "advanced_representation_aggregate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
