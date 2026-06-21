from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from .data import build_duplicate_family_groups, load_rhythm_windows


def _test_indices(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test = next(splitter.split(idx, y, groups=groups))
    return test


def _split_grouping(run_dir: Path) -> str:
    summary_path = run_dir / "split_summary.json"
    if not summary_path.exists():
        return "record"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return str(summary.get("split_grouping", "record"))


def _metrics(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    correct = y == pred
    vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    if weights is None:
        weights = np.ones(len(y))
    return {
        "accuracy": float(np.average(correct, weights=weights)),
        "macro_f1": float(
            f1_score(y, pred, labels=[0, 1, 2], average="macro", sample_weight=weights)
        ),
        "error_rate": float(np.average(~correct, weights=weights)),
        "vtvf_error_rate": float(np.average(vtvf, weights=weights)),
    }


def _record_weights(record_ids: np.ndarray) -> np.ndarray:
    counts = pd.Series(record_ids).value_counts()
    return np.asarray([1.0 / counts[record] for record in record_ids], dtype=float)


def _bootstrap_indices(
    record_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unique = np.unique(record_ids)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    lookup = {record: np.flatnonzero(record_ids == record) for record in unique}
    return np.concatenate([lookup[record] for record in sampled])


def _risk_capture(y: np.ndarray, pred: np.ndarray, score: np.ndarray, burden: float) -> float:
    vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    order = np.argsort(-score)
    reviewed = order[: max(1, int(round(len(y) * burden)))]
    return float(vtvf[reviewed].sum() / max(vtvf.sum(), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--mitigation-summary", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--selected-risk-root", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path, default=Path("results/record_cluster_statistics_20260620")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_rhythm_windows(args.mat)

    mitigation = pd.read_csv(args.mitigation_summary)
    test_records: dict[int, np.ndarray] = {}
    for seed in [42, 43, 44]:
        seed_runs = mitigation[(mitigation["seed"] == seed) & mitigation["variant"].eq("baseline")]
        if seed_runs.empty:
            raise ValueError(f"Missing baseline run for seed {seed}")
        baseline_dir = Path(str(seed_runs.iloc[0]["run_dir"]))
        split_grouping = _split_grouping(baseline_dir)
        if split_grouping == "duplicate_family":
            groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
        elif split_grouping == "record":
            groups = dataset.record_ids
        else:
            raise ValueError(f"Unsupported split_grouping={split_grouping!r} for {baseline_dir}")
        test_records[seed] = dataset.record_ids[_test_indices(dataset.y, groups, seed)]

    class_rows = []
    bootstrap_rows = []
    for seed in [42, 43, 44]:
        runs = mitigation[
            (mitigation["seed"] == seed)
            & mitigation["variant"].isin(["baseline", "prototype_separation"])
        ].set_index("variant")
        predictions = {
            variant: pd.read_csv(Path(str(runs.loc[variant, "run_dir"])) / "test_predictions.csv")
            for variant in ["baseline", "prototype_separation"]
        }
        records = test_records[seed]
        weights = _record_weights(records)
        for variant, frame in predictions.items():
            y = frame["y_true"].to_numpy(int)
            pred = frame["y_pred"].to_numpy(int)
            class_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "weighting": "window_level",
                    **_metrics(y, pred),
                }
            )
            class_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "weighting": "record_balanced",
                    **_metrics(y, pred, weights),
                }
            )

        rng = np.random.default_rng(1000 + seed)
        base = predictions["baseline"]
        pro = predictions["prototype_separation"]
        y = base["y_true"].to_numpy(int)
        base_pred = base["y_pred"].to_numpy(int)
        pro_pred = pro["y_pred"].to_numpy(int)
        for rep in range(args.bootstrap_reps):
            sampled = _bootstrap_indices(records, rng)
            base_metrics = _metrics(y[sampled], base_pred[sampled])
            pro_metrics = _metrics(y[sampled], pro_pred[sampled])
            bootstrap_rows.append(
                {
                    "seed": seed,
                    "replicate": rep,
                    **{
                        f"pro_minus_baseline_{metric}": pro_metrics[metric] - base_metrics[metric]
                        for metric in base_metrics
                    },
                }
            )
    pd.DataFrame(class_rows).to_csv(
        args.out / "record_balanced_classification_metrics.csv", index=False
    )
    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(args.out / "pro_record_cluster_bootstrap_replicates.csv", index=False)
    ci_rows = []
    for seed, sub in boot_df.groupby("seed"):
        for metric in [col for col in sub.columns if col.startswith("pro_minus_baseline_")]:
            values = sub[metric].to_numpy(float)
            ci_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "mean_difference": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "probability_improvement": float(
                        (values > 0).mean()
                        if metric in {"pro_minus_baseline_accuracy", "pro_minus_baseline_macro_f1"}
                        else (values < 0).mean()
                    ),
                }
            )
    pd.DataFrame(ci_rows).to_csv(args.out / "pro_record_cluster_bootstrap_ci.csv", index=False)

    manifest = pd.read_csv(args.core_manifest)
    teachers = manifest[manifest["stage"].eq("regularity_feature_injection")].set_index("seed")
    risks = manifest[manifest["stage"].eq("risk_aligned_distillation")].set_index("seed")
    risk_boot_rows = []
    risk_summary_rows = []
    for seed in [42, 43, 44]:
        pred = pd.read_csv(Path(str(teachers.loc[seed, "run_dir"])) / "test_predictions.csv")
        risk_run = Path(str(risks.loc[seed, "run_dir"]))
        risk = pd.read_csv(risk_run / "risk_scores_test.csv")
        records = test_records[seed]
        y = pred["y_true"].to_numpy(int)
        y_pred = pred["y_pred"].to_numpy(int)
        score = risk["risk_score"].to_numpy(float)
        observed = _risk_capture(y, y_pred, score, 0.20)
        rng = np.random.default_rng(2000 + seed)
        values = []
        for rep in range(args.bootstrap_reps):
            sampled = _bootstrap_indices(records, rng)
            value = _risk_capture(y[sampled], y_pred[sampled], score[sampled], 0.20)
            values.append(value)
            risk_boot_rows.append({"seed": seed, "replicate": rep, "vtvf_capture_20": value})
        risk_summary_rows.append(
            {
                "seed": seed,
                "observed_vtvf_capture_20": observed,
                "bootstrap_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "n_test_records": int(pd.Series(records).nunique()),
            }
        )
    pd.DataFrame(risk_boot_rows).to_csv(
        args.out / "risk_record_cluster_bootstrap_replicates.csv", index=False
    )
    pd.DataFrame(risk_summary_rows).to_csv(
        args.out / "risk_record_cluster_bootstrap_ci.csv", index=False
    )
    print(args.out)


if __name__ == "__main__":
    main()
