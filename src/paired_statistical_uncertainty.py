from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from .data import build_duplicate_family_groups, load_rhythm_windows, make_splits


LOWER_IS_BETTER = {
    "ece_delta",
    "total_errors_delta",
    "vtvf_cross_errors_delta",
    "error_rate_delta",
    "vtvf_cross_error_rate_within_vtvf_delta",
}


def _parse_comparison(text: str) -> tuple[int, Path]:
    seed, path = text.split("=", 1)
    return int(seed), Path(path)


def _parse_pair(values: list[str]) -> dict:
    if len(values) != 5:
        raise argparse.ArgumentTypeError("--pair expects SEED BASELINE COMPARATOR BASELINE_DIR COMPARATOR_DIR")
    seed, baseline, comparator, baseline_dir, comparator_dir = values
    return {
        "seed": int(seed),
        "baseline": baseline,
        "comparator": comparator,
        "baseline_dir": Path(baseline_dir),
        "comparator_dir": Path(comparator_dir),
    }


def _load_seed_deltas(comparisons: list[tuple[int, Path]]) -> pd.DataFrame:
    rows = []
    for seed, path in comparisons:
        df = pd.read_csv(path / "classification_run_deltas.csv")
        row = df.iloc[0].to_dict()
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    boot = np.empty(n_bootstrap, dtype=float)
    n = len(values)
    for i in range(n_bootstrap):
        boot[i] = rng.choice(values, size=n, replace=True).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    n = len(values)
    if n > 20:
        rng = np.random.default_rng(42)
        samples = rng.choice([-1, 1], size=(200000, n)) * values[None, :]
        return float((np.abs(samples.mean(axis=1)) >= observed - 1e-12).mean())
    means = []
    for signs in itertools.product([-1, 1], repeat=n):
        means.append(abs((values * np.asarray(signs)).mean()))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def _bayesian_mean_interval(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Jeffreys-prior normal model posterior for the paired mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd <= 0:
        return mean, mean, float(mean > 0), float(mean < 0)
    sampled_sigma2 = (n - 1) * sd**2 / rng.chisquare(df=n - 1, size=draws)
    sampled_mean = rng.normal(mean, np.sqrt(sampled_sigma2 / n))
    lo, hi = np.percentile(sampled_mean, [2.5, 97.5])
    return float(lo), float(hi), float((sampled_mean > 0).mean()), float((sampled_mean < 0).mean())


def _paired_tests(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0]
    out = {
        "paired_t_p": np.nan,
        "wilcoxon_p": np.nan,
        "sign_test_p": np.nan,
        "sign_flip_permutation_p": _sign_flip_pvalue(values),
    }
    if len(values) >= 2 and values.std(ddof=1) > 0:
        out["paired_t_p"] = float(stats.ttest_1samp(values, popmean=0.0).pvalue)
    if len(nonzero) >= 2:
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            pass
    positives = int((values > 0).sum())
    negatives = int((values < 0).sum())
    if positives + negatives > 0:
        out["sign_test_p"] = float(stats.binomtest(max(positives, negatives), positives + negatives, 0.5).pvalue)
    return out


def _seed_level_stats(deltas: pd.DataFrame, n_bootstrap: int, bayes_draws: int, rng: np.random.Generator) -> pd.DataFrame:
    metric_cols = [c for c in deltas.columns if c.endswith("_delta")]
    rows = []
    for metric in metric_cols:
        values = deltas[metric].astype(float).to_numpy()
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(values, n_bootstrap, rng)
        bayes_lo, bayes_hi, p_gt0, p_lt0 = _bayesian_mean_interval(values, bayes_draws, rng)
        tests = _paired_tests(values)
        preferred_prob = p_lt0 if metric in LOWER_IS_BETTER else p_gt0
        rows.append(
            {
                "metric": metric,
                "n_paired_seeds": int(len(values)),
                "mean_delta": mean,
                "median_delta": float(np.median(values)),
                "sd_delta": sd,
                "cohen_dz": float(mean / sd) if sd > 0 else np.nan,
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
                "n_zero": int((values == 0).sum()),
                "seed_bootstrap_ci95_low": ci_lo,
                "seed_bootstrap_ci95_high": ci_hi,
                "bayes_credible95_low": bayes_lo,
                "bayes_credible95_high": bayes_hi,
                "posterior_p_delta_gt_0": p_gt0,
                "posterior_p_delta_lt_0": p_lt0,
                "posterior_p_preferred_direction": preferred_prob,
                **tests,
                "seed_differences": ";".join(f"{int(seed)}:{value:.6g}" for seed, value in zip(deltas["seed"], values)),
            }
        )
    return pd.DataFrame(rows)


def _load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(run_dir / "test_predictions.csv")
    return df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int)


def _duplicate_family_test_groups(mat: Path, seed: int) -> np.ndarray:
    dataset = load_rhythm_windows(mat)
    groups = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    splits = make_splits(dataset.x, dataset.y, groups=groups, seed=seed)
    # Reconstruct test indices by matching normalised windows. SHA hashes avoid relying on object identity.
    import hashlib

    digest_to_group: dict[str, str] = {}
    for row, group in zip(dataset.x.reshape(len(dataset.x), -1), groups):
        digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        digest_to_group[digest] = str(group)
    test_groups = []
    for row in splits.x_test.reshape(len(splits.x_test), -1):
        digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        test_groups.append(digest_to_group[digest])
    return np.asarray(test_groups)


def _metrics_for_sample(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = y != pred
    vtvf = np.isin(y, [1, 2])
    cross = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    return {
        "accuracy": float((~error).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "error_rate": float(error.mean()),
        "vtvf_cross_error_rate_within_vtvf": float(cross[vtvf].mean()) if vtvf.any() else np.nan,
    }


def _cluster_bootstrap_pair(pair: dict, mat: Path, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    y_base, pred_base = _load_predictions(pair["baseline_dir"])
    y_comp, pred_comp = _load_predictions(pair["comparator_dir"])
    if not np.array_equal(y_base, y_comp):
        raise ValueError(f"Prediction labels do not match for seed {pair['seed']}")
    groups = _duplicate_family_test_groups(mat, pair["seed"])
    if len(groups) != len(y_base):
        raise ValueError(f"Group/prediction length mismatch for seed {pair['seed']}")

    group_values = np.unique(groups)
    group_to_idx = {group: np.flatnonzero(groups == group) for group in group_values}
    rows = []
    for i in range(n_bootstrap):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        idx = np.concatenate([group_to_idx[group] for group in sampled_groups])
        base = _metrics_for_sample(y_base[idx], pred_base[idx])
        comp = _metrics_for_sample(y_comp[idx], pred_comp[idx])
        row = {"seed": pair["seed"], "bootstrap": i}
        for metric in base:
            row[f"{metric}_delta"] = comp[metric] - base[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_summary(cluster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in cluster.columns if c.endswith("_delta")]
    for metric in metric_cols:
        for seed, sub in cluster.groupby("seed"):
            lo, hi = np.percentile(sub[metric].dropna(), [2.5, 97.5])
            rows.append(
                {
                    "scope": "within_seed_duplicate_family_cluster_bootstrap",
                    "seed": int(seed),
                    "metric": metric,
                    "mean": float(sub[metric].mean()),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                }
            )
        # Pooled across seed bootstraps as descriptive, not a replacement for seed-level inference.
        pooled = cluster[metric].dropna()
        lo, hi = np.percentile(pooled, [2.5, 97.5])
        rows.append(
            {
                "scope": "pooled_seed_cluster_bootstrap_descriptive",
                "seed": -1,
                "metric": metric,
                "mean": float(pooled.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired seed and duplicate-family cluster uncertainty analysis.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--comparison", action="append", type=_parse_comparison, required=True)
    parser.add_argument("--pair", nargs=5, action="append", metavar=("SEED", "BASELINE", "COMPARATOR", "BASELINE_DIR", "COMPARATOR_DIR"))
    parser.add_argument("--out", type=Path, default=Path("results/paired_statistical_uncertainty"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--cluster-bootstrap", type=int, default=2000)
    parser.add_argument("--bayes-draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    deltas = _load_seed_deltas(args.comparison)
    seed_stats = _seed_level_stats(deltas, args.bootstrap, args.bayes_draws, rng)
    deltas.to_csv(args.out / "paired_seed_deltas.csv", index=False)
    seed_stats.to_csv(args.out / "paired_seed_uncertainty_summary.csv", index=False)

    cluster_summary = pd.DataFrame()
    if args.pair:
        pairs = [_parse_pair(item) for item in args.pair]
        cluster = pd.concat(
            [_cluster_bootstrap_pair(pair, args.mat, args.cluster_bootstrap, rng) for pair in pairs],
            ignore_index=True,
        )
        cluster.to_csv(args.out / "duplicate_family_cluster_bootstrap_samples.csv", index=False)
        cluster_summary = _cluster_summary(cluster)
        cluster_summary.to_csv(args.out / "duplicate_family_cluster_bootstrap_summary.csv", index=False)

    report = {
        "n_paired_seeds": int(len(deltas)),
        "bootstrap": args.bootstrap,
        "cluster_bootstrap": args.cluster_bootstrap if args.pair else 0,
        "bayes_draws": args.bayes_draws,
        "note": "Seed-level inference is primary for model comparison; duplicate-family cluster bootstrap describes within-test-set uncertainty without treating windows as independent.",
        "outputs": {
            "paired_seed_deltas": str(args.out / "paired_seed_deltas.csv"),
            "paired_seed_uncertainty_summary": str(args.out / "paired_seed_uncertainty_summary.csv"),
            "duplicate_family_cluster_bootstrap_summary": str(args.out / "duplicate_family_cluster_bootstrap_summary.csv") if args.pair else "",
        },
    }
    (args.out / "paired_statistical_uncertainty_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(seed_stats)
    if not cluster_summary.empty:
        print(cluster_summary)


if __name__ == "__main__":
    main()
