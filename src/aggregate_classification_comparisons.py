from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DELTA_COLUMNS = [
    "accuracy_delta",
    "macro_f1_delta",
    "ece_delta",
    "mean_margin_delta",
    "vtvf_cross_errors_delta",
    "total_errors_delta",
]


def _parse_seed_comparison(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Expected SEED=COMPARISON_DIR, for example 42=results/run/seed42_comparison")
    seed, path_text = text.split("=", 1)
    seed = seed.strip()
    if not seed:
        raise argparse.ArgumentTypeError("Seed label cannot be empty.")
    return seed, Path(path_text)


def _load_seed_comparison(seed: str, comparison_dir: Path) -> dict[str, float | int | str]:
    delta_path = comparison_dir / "classification_run_deltas.csv"
    summary_path = comparison_dir / "classification_run_comparison.csv"
    if not delta_path.exists():
        raise FileNotFoundError(f"Missing {delta_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    delta = pd.read_csv(delta_path).iloc[0].to_dict()
    summary = pd.read_csv(summary_path)
    row: dict[str, float | int | str] = {
        "seed": seed,
        "baseline": str(delta["baseline"]),
        "comparator": str(delta["comparator"]),
    }
    for col in DELTA_COLUMNS:
        row[col] = float(delta[col])

    for _, model_row in summary.iterrows():
        prefix = str(model_row["model"]).lower().replace("-", "_")
        for col in [
            "accuracy",
            "macro_f1",
            "ece",
            "total_errors",
            "vtvf_cross_errors",
            "sr_as_vt",
            "sr_as_vf",
            "vt_as_vf",
            "vf_as_vt",
            "mean_margin",
            "low_margin_rate_0p10",
            "low_margin_rate_0p20",
        ]:
            if col in model_row:
                value = model_row[col]
                row[f"{prefix}_{col}"] = float(value) if col not in {"total_errors", "vtvf_cross_errors", "sr_as_vt", "sr_as_vf", "vt_as_vf", "vf_as_vt"} else int(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate paired classification-run comparisons across seeds."
    )
    parser.add_argument(
        "--comparison",
        action="append",
        type=_parse_seed_comparison,
        required=True,
        metavar="SEED=COMPARISON_DIR",
    )
    parser.add_argument("--out", type=Path, default=Path("results/classification_multiseed_summary"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [_load_seed_comparison(seed, path) for seed, path in args.comparison]
    per_seed = pd.DataFrame(rows).sort_values("seed")
    per_seed.to_csv(args.out / "per_seed_model_comparison.csv", index=False)

    aggregate_rows = []
    for col in DELTA_COLUMNS:
        values = per_seed[col].astype(float)
        aggregate_rows.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(values.count()),
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
                "n_zero": int((values == 0).sum()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.out / "aggregate_delta_summary.csv", index=False)

    report = {
        "per_seed": per_seed.to_dict(orient="records"),
        "aggregate_delta_summary": aggregate.to_dict(orient="records"),
        "notes": {
            "delta_direction": "Positive deltas mean the comparator is higher than the baseline. For ECE and errors, negative is better.",
            "mean_margin": "Top-1 probability minus top-2 probability.",
        },
    }
    (args.out / "multiseed_model_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(per_seed)
    print(aggregate)


if __name__ == "__main__":
    main()
