from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_seed_dirs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    reviews = []
    monotonicity = []
    for seed_dir in sorted(root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        summary_path = seed_dir / "risk_corruption_summary.csv"
        review_path = seed_dir / "risk_corruption_review_curves.csv"
        mono_path = seed_dir / "risk_corruption_monotonicity.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, "seed", seed)
            summaries.append(df)
        if review_path.exists():
            df = pd.read_csv(review_path)
            df.insert(0, "seed", seed)
            reviews.append(df)
        if mono_path.exists():
            df = pd.read_csv(mono_path)
            df.insert(0, "seed", seed)
            monotonicity.append(df)
    if not summaries:
        raise FileNotFoundError(f"No seed*/risk_corruption_summary.csv files under {root}")
    return pd.concat(summaries, ignore_index=True), pd.concat(reviews, ignore_index=True), pd.concat(monotonicity, ignore_index=True)


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[value_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/risk_corruption_robustness_20260609"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.root / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, reviews, monotonicity = _read_seed_dirs(args.root)
    summary.to_csv(out_dir / "risk_corruption_run_level.csv", index=False)
    reviews.to_csv(out_dir / "risk_corruption_review_run_level.csv", index=False)
    monotonicity.to_csv(out_dir / "risk_corruption_monotonicity_run_level.csv", index=False)

    metric_cols = [
        "accuracy",
        "all_errors",
        "vtvf_errors",
        "risk_score_mean",
        "risk_score_p90",
        "entropy_mean",
        "error_auroc",
        "vtvf_error_auroc",
    ]
    _mean_std(summary, ["corruption", "severity"], metric_cols).to_csv(out_dir / "risk_corruption_mean_std.csv", index=False)

    review_focus = reviews[reviews["review_burden"].isin([0.10, 0.20, 0.30])].copy()
    review_cols = ["all_error_captured", "vtvf_error_captured", "auto_error_rate", "auto_vtvf_error_rate"]
    _mean_std(review_focus, ["corruption", "severity", "review_burden"], review_cols).to_csv(
        out_dir / "risk_corruption_review_mean_std.csv", index=False
    )

    _mean_std(
        monotonicity,
        ["corruption"],
        ["risk_mean_severity_spearman", "error_count_severity_spearman", "risk_mean_severity_4_minus_0"],
    ).to_csv(out_dir / "risk_corruption_monotonicity_mean_std.csv", index=False)

    # Compact manuscript table: clean, medium, and severe corruption only.
    compact = summary[summary["severity"].isin([0, 2, 4])].copy()
    compact_mean = _mean_std(
        compact,
        ["corruption", "severity"],
        ["risk_score_mean", "error_auroc", "vtvf_error_auroc", "all_errors", "vtvf_errors"],
    )
    compact_mean.to_csv(out_dir / "risk_corruption_compact_manuscript_table.csv", index=False)

    print("Wrote", out_dir)
    print(compact_mean.head(40))


if __name__ == "__main__":
    main()
