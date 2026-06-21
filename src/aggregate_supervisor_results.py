from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime_supervisor import _load_base, build_supervisor_table, summarise_supervisor


VALUE_COLS = [
    "rate",
    "mean_supervisor_risk",
    "error_rate",
    "vtvf_cross_error_rate",
    "vtvf_cross_errors",
]

TRIGGER_VALUE_COLS = [
    "review_burden",
    "auto_coverage",
    "all_error_captured",
    "vtvf_error_captured",
    "false_alarm_rate",
    "review_error_precision",
    "auto_error_rate",
    "auto_vtvf_error_rate",
]

TRIGGER_POLICIES = {
    "suspect_or_higher": {"SUSPECT", "RECOVER", "HUMAN_REVIEW"},
    "recover_or_human": {"RECOVER", "HUMAN_REVIEW"},
    "human_only": {"HUMAN_REVIEW"},
}


def _has_risk_signals(run_dir: Path) -> bool:
    return any(
        [
            (run_dir / "ambiguity_routing_policy.csv").exists(),
            (run_dir / "stability_scores.csv").exists(),
            (run_dir / "uncertainty_scores.csv").exists() and (run_dir / "ambiguity_scores.csv").exists(),
        ]
    )


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "run_dir" not in df.columns:
        raise ValueError("Manifest must contain a run_dir column.")
    if "seed" not in df.columns:
        df["seed"] = np.arange(len(df))
    if "model" not in df.columns:
        df["model"] = "unknown"
    if "variant" not in df.columns:
        df["variant"] = df["model"]
    return df


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(group))
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _supervisor_for_run(
    run_dir: Path,
    confidence_threshold: float,
    suspect_threshold: float,
    recover_threshold: float,
    human_threshold: float,
    write_run_outputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _load_base(run_dir)
    table = build_supervisor_table(
        base,
        confidence_threshold=confidence_threshold,
        suspect_threshold=suspect_threshold,
        recover_threshold=recover_threshold,
        human_threshold=human_threshold,
    )
    summary = summarise_supervisor(table)
    if write_run_outputs:
        table.to_csv(run_dir / "runtime_supervisor_policy.csv", index=False)
        summary.to_csv(run_dir / "runtime_supervisor_summary.csv", index=False)
    return table, summary


def _trigger_metrics(table: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    if "is_error" not in table:
        return rows
    any_error = table["is_error"].astype(bool).to_numpy()
    if "is_vtvf_cross_error" in table:
        vtvf_error = table["is_vtvf_cross_error"].astype(bool).to_numpy()
    else:
        vtvf_error = np.zeros(len(table), dtype=bool)

    states = table["supervisor_state"].astype(str)
    total_errors = max(int(any_error.sum()), 1)
    total_vtvf = max(int(vtvf_error.sum()), 1)
    n = max(len(table), 1)
    for policy, trigger_states in TRIGGER_POLICIES.items():
        review = states.isin(trigger_states).to_numpy()
        auto = ~review
        reviewed = max(int(review.sum()), 1)
        rows.append(
            {
                "trigger_policy": policy,
                "review_states": "+".join(sorted(trigger_states)),
                "review_burden": float(review.sum() / n),
                "auto_coverage": float(auto.sum() / n),
                "all_error_captured": float(any_error[review].sum() / total_errors),
                "vtvf_error_captured": float(vtvf_error[review].sum() / total_vtvf),
                "false_alarm_rate": float((review & ~any_error).sum() / reviewed),
                "review_error_precision": float(any_error[review].sum() / reviewed),
                "auto_error_rate": float(any_error[auto].mean()) if auto.any() else np.nan,
                "auto_vtvf_error_rate": float(vtvf_error[auto].mean()) if auto.any() else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/supervisor_summary"))
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--suspect-threshold", type=float, default=0.45)
    parser.add_argument("--recover-threshold", type=float, default=0.65)
    parser.add_argument("--human-threshold", type=float, default=0.75)
    parser.add_argument(
        "--allow-prediction-only",
        action="store_true",
        help="Allow runs with only test_predictions.csv. These runs have weak supervisor evidence.",
    )
    parser.add_argument("--write-run-outputs", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    trigger_rows = []
    skipped_rows = []
    for item in manifest.to_dict(orient="records"):
        run_dir = Path(item["run_dir"])
        if not args.allow_prediction_only and not _has_risk_signals(run_dir):
            skipped_rows.append(
                {
                    "model": item.get("model", "unknown"),
                    "variant": item.get("variant", item.get("model", "unknown")),
                    "seed": item.get("seed", np.nan),
                    "run_dir": str(run_dir),
                    "reason": "missing supervisor risk signals; run uncertainty/ambiguity/stability analysis first",
                }
            )
            continue
        try:
            table, summary = _supervisor_for_run(
                run_dir,
                confidence_threshold=args.confidence_threshold,
                suspect_threshold=args.suspect_threshold,
                recover_threshold=args.recover_threshold,
                human_threshold=args.human_threshold,
                write_run_outputs=args.write_run_outputs,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "model": item.get("model", "unknown"),
                    "variant": item.get("variant", item.get("model", "unknown")),
                    "seed": item.get("seed", np.nan),
                    "run_dir": str(run_dir),
                    "reason": str(exc),
                }
            )
            continue
        for row in summary.to_dict(orient="records"):
            run_rows.append(
                {
                    "model": item.get("model", "unknown"),
                    "variant": item.get("variant", item.get("model", "unknown")),
                    "seed": int(item.get("seed", -1)),
                    "run_dir": str(run_dir),
                    **row,
                }
            )
        for row in _trigger_metrics(table):
            trigger_rows.append(
                {
                    "model": item.get("model", "unknown"),
                    "variant": item.get("variant", item.get("model", "unknown")),
                    "seed": int(item.get("seed", -1)),
                    "run_dir": str(run_dir),
                    **row,
                }
            )

    run_df = pd.DataFrame(run_rows)
    trigger_df = pd.DataFrame(trigger_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    run_df.to_csv(args.out_dir / "supervisor_state_run_level.csv", index=False)
    trigger_df.to_csv(args.out_dir / "supervisor_trigger_run_level.csv", index=False)
    if not skipped_df.empty:
        skipped_df.to_csv(args.out_dir / "supervisor_skipped_runs.csv", index=False)

    if not run_df.empty:
        _mean_std(run_df, ["model", "state"], VALUE_COLS).to_csv(
            args.out_dir / "supervisor_by_model_mean_std.csv", index=False
        )
        _mean_std(run_df, ["variant", "state"], VALUE_COLS).to_csv(
            args.out_dir / "supervisor_by_variant_mean_std.csv", index=False
        )
        if not trigger_df.empty:
            _mean_std(trigger_df, ["model", "trigger_policy"], TRIGGER_VALUE_COLS).to_csv(
                args.out_dir / "supervisor_trigger_by_model_mean_std.csv", index=False
            )
            _mean_std(trigger_df, ["variant", "trigger_policy"], TRIGGER_VALUE_COLS).to_csv(
                args.out_dir / "supervisor_trigger_by_variant_mean_std.csv", index=False
            )

        pivot = run_df.pivot_table(
            index=["model", "variant", "seed", "run_dir"],
            columns="state",
            values=["rate", "error_rate", "vtvf_cross_error_rate", "vtvf_cross_errors"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{state}".lower() for metric, state in pivot.columns]
        pivot.reset_index().to_csv(args.out_dir / "supervisor_compact_run_level.csv", index=False)

    print(f"Wrote supervisor summaries to {args.out_dir}")
    if not skipped_df.empty:
        print(f"Skipped {len(skipped_df)} runs; see supervisor_skipped_runs.csv")


if __name__ == "__main__":
    main()
