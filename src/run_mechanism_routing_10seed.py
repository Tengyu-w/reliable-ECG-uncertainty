from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = list(range(42, 52))


def _seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def _resolve_runs() -> dict[int, dict[str, Path]]:
    pairs: dict[int, dict[str, Path]] = {seed: {} for seed in SEEDS}
    core = Path("results/core_interventions_risk_pro_plus")
    readable_single = Path("results/risk_pro_readable_20260626")
    readable_10seed = Path("results/risk_pro_readable_10seed_20260626")

    for root in [core, readable_10seed]:
        if not root.exists():
            continue
        for embeddings in root.glob("*/embeddings_test.npz"):
            run_dir = embeddings.parent
            seed = _seed_from_name(run_dir)
            if seed not in pairs:
                continue
            if "core_regularity_injection" in run_dir.name:
                pairs[seed]["teacher"] = run_dir
            elif "core_risk_pro_readable" in run_dir.name:
                pairs[seed]["second"] = run_dir

    if readable_single.exists():
        for embeddings in readable_single.glob("*/embeddings_test.npz"):
            run_dir = embeddings.parent
            seed = _seed_from_name(run_dir)
            if seed in pairs and "core_risk_pro_readable" in run_dir.name:
                pairs[seed]["second"] = run_dir

    missing = {seed: value for seed, value in pairs.items() if "teacher" not in value or "second" not in value}
    if missing:
        detail = {seed: sorted(value) for seed, value in missing.items()}
        raise FileNotFoundError(f"Missing teacher/second-opinion pairs: {detail}")
    return pairs


def _run_seed(
    seed: int,
    teacher: Path,
    second: Path,
    out_dir: Path,
    budgets: list[float],
    force: bool,
) -> dict[str, Any]:
    seed_out = out_dir / f"seed{seed}"
    done = seed_out / "mechanism_layered_policy_summary.csv"
    if done.exists() and not force:
        return {"seed": seed, "status": "skipped_existing", "out": str(seed_out)}

    seed_out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.evidence_informed_recovery_routing",
        "--run-dir",
        str(teacher),
        "--second-opinion-run-dir",
        str(second),
        "--out",
        str(seed_out),
        "--budgets",
        *[str(value) for value in budgets],
    ]
    subprocess.run(command, check=True)
    return {"seed": seed, "status": "completed", "out": str(seed_out)}


def _read_seed_csv(out_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = out_dir / f"seed{seed}" / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        for col in metric_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(out_dir: Path) -> dict[str, Any]:
    mechanism_policy = _read_seed_csv(out_dir, "mechanism_layered_policy_summary.csv")
    optimized_mechanism_policy = _read_seed_csv(out_dir, "optimized_mechanism_layered_policy_summary.csv")
    total_policy = _read_seed_csv(out_dir, "layered_policy_summary.csv")
    heads = _read_seed_csv(out_dir, "mechanism_risk_head_summary.csv")
    ablations = _read_seed_csv(out_dir, "evidence_ablation_summary.csv")

    if not mechanism_policy.empty:
        mechanism_policy.to_csv(out_dir / "all_seed_mechanism_layered_policy_summary.csv", index=False)
        metrics = [
            "single_label_rate",
            "mechanism_action_rate",
            "vtvf_set_rate",
            "all_error_addressed",
            "vtvf_cross_error_addressed",
            "single_label_error_rate_after_mechanism_routing",
            "single_label_vtvf_cross_error_rate_after_mechanism_routing",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
        ]
        _mean_std(mechanism_policy, ["budget"], metrics).to_csv(
            out_dir / "mechanism_policy_mean_std_by_budget.csv", index=False
        )

    if not optimized_mechanism_policy.empty:
        optimized_mechanism_policy.to_csv(out_dir / "all_seed_optimized_mechanism_layered_policy_summary.csv", index=False)
        metrics = [
            "single_label_rate",
            "mechanism_action_rate",
            "vtvf_set_rate",
            "all_error_addressed",
            "vtvf_cross_error_addressed",
            "single_label_error_rate_after_mechanism_routing",
            "single_label_vtvf_cross_error_rate_after_mechanism_routing",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
        ]
        _mean_std(optimized_mechanism_policy, ["budget"], metrics).to_csv(
            out_dir / "optimized_mechanism_policy_mean_std_by_budget.csv", index=False
        )

    if not total_policy.empty:
        total_policy.to_csv(out_dir / "all_seed_total_risk_layered_policy_summary.csv", index=False)
        metrics = [
            "review_rate",
            "vtvf_set_rate",
            "single_label_rate",
            "all_error_addressed_by_review_or_set",
            "vtvf_cross_error_addressed_by_review_or_set",
            "single_label_error_rate_after_routing",
            "single_label_vtvf_cross_error_rate_after_routing",
            "automatic_unresolved_error_rate",
            "automatic_unresolved_vtvf_cross_error_rate",
        ]
        _mean_std(total_policy, ["budget"], metrics).to_csv(
            out_dir / "total_risk_policy_mean_std_by_budget.csv", index=False
        )

    if not heads.empty:
        heads.to_csv(out_dir / "all_seed_mechanism_risk_head_summary.csv", index=False)
        head_metrics = ["val_positive", "test_positive", "enabled_for_routing", "test_auroc", "test_aupr"]
        if "enabled_for_routing" in heads.columns:
            heads["enabled_for_routing"] = heads["enabled_for_routing"].astype(float)
        _mean_std(heads, ["mechanism"], head_metrics).to_csv(
            out_dir / "mechanism_head_mean_std.csv", index=False
        )

    if not ablations.empty:
        ablations.to_csv(out_dir / "all_seed_evidence_ablation_summary.csv", index=False)
        ablation_metrics = ["auroc", "aupr", "all_error_captured", "vtvf_cross_error_captured", "auto_vtvf_cross_error_rate"]
        _mean_std(ablations, ["feature_group", "target", "budget"], ablation_metrics).to_csv(
            out_dir / "evidence_ablation_mean_std.csv", index=False
        )

    paired_rows = []
    if not mechanism_policy.empty and not total_policy.empty:
        left = mechanism_policy.merge(total_policy, on=["seed", "budget"], suffixes=("_mechanism", "_total"))
        for _, row in left.iterrows():
            paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "mechanism_action_rate": row["mechanism_action_rate"],
                    "total_review_plus_set_rate": row["review_rate"] + row["vtvf_set_rate_total"],
                    "delta_vtvf_cross_addressed_mechanism_minus_total": row["vtvf_cross_error_addressed"]
                    - row["vtvf_cross_error_addressed_by_review_or_set"],
                    "delta_all_error_addressed_mechanism_minus_total": row["all_error_addressed"]
                    - row["all_error_addressed_by_review_or_set"],
                    "delta_auto_vtvf_unresolved_mechanism_minus_total": row[
                        "automatic_unresolved_vtvf_cross_error_rate_mechanism"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_total"],
                    "delta_auto_error_unresolved_mechanism_minus_total": row[
                        "automatic_unresolved_error_rate_mechanism"
                    ]
                    - row["automatic_unresolved_error_rate_total"],
                }
            )
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(out_dir / "paired_mechanism_vs_total_policy.csv", index=False)
        _mean_std(
            paired,
            ["budget"],
            [
                "mechanism_action_rate",
                "total_review_plus_set_rate",
                "delta_vtvf_cross_addressed_mechanism_minus_total",
                "delta_all_error_addressed_mechanism_minus_total",
                "delta_auto_vtvf_unresolved_mechanism_minus_total",
                "delta_auto_error_unresolved_mechanism_minus_total",
            ],
        ).to_csv(out_dir / "paired_mechanism_vs_total_policy_mean_std.csv", index=False)

    optimized_paired_rows = []
    if not optimized_mechanism_policy.empty and not total_policy.empty:
        left = optimized_mechanism_policy.merge(total_policy, on=["seed", "budget"], suffixes=("_mechanism", "_total"))
        for _, row in left.iterrows():
            optimized_paired_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "optimized_mechanism_action_rate": row["mechanism_action_rate"],
                    "total_review_plus_set_rate": row["review_rate"] + row["vtvf_set_rate_total"],
                    "delta_vtvf_cross_addressed_optimized_mechanism_minus_total": row["vtvf_cross_error_addressed"]
                    - row["vtvf_cross_error_addressed_by_review_or_set"],
                    "delta_all_error_addressed_optimized_mechanism_minus_total": row["all_error_addressed"]
                    - row["all_error_addressed_by_review_or_set"],
                    "delta_auto_vtvf_unresolved_optimized_mechanism_minus_total": row[
                        "automatic_unresolved_vtvf_cross_error_rate_mechanism"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_total"],
                    "delta_auto_error_unresolved_optimized_mechanism_minus_total": row[
                        "automatic_unresolved_error_rate_mechanism"
                    ]
                    - row["automatic_unresolved_error_rate_total"],
                }
            )
        optimized_paired = pd.DataFrame(optimized_paired_rows)
        optimized_paired.to_csv(out_dir / "paired_optimized_mechanism_vs_total_policy.csv", index=False)
        _mean_std(
            optimized_paired,
            ["budget"],
            [
                "optimized_mechanism_action_rate",
                "total_review_plus_set_rate",
                "delta_vtvf_cross_addressed_optimized_mechanism_minus_total",
                "delta_all_error_addressed_optimized_mechanism_minus_total",
                "delta_auto_vtvf_unresolved_optimized_mechanism_minus_total",
                "delta_auto_error_unresolved_optimized_mechanism_minus_total",
            ],
        ).to_csv(out_dir / "paired_optimized_mechanism_vs_total_policy_mean_std.csv", index=False)

    fixed_vs_optimized_rows = []
    if not optimized_mechanism_policy.empty and not mechanism_policy.empty:
        left = optimized_mechanism_policy.merge(
            mechanism_policy,
            on=["seed", "budget"],
            suffixes=("_optimized", "_fixed"),
        )
        for _, row in left.iterrows():
            fixed_vs_optimized_rows.append(
                {
                    "seed": int(row["seed"]),
                    "budget": float(row["budget"]),
                    "delta_action_rate_optimized_minus_fixed": row["mechanism_action_rate_optimized"]
                    - row["mechanism_action_rate_fixed"],
                    "delta_vtvf_cross_addressed_optimized_minus_fixed": row["vtvf_cross_error_addressed_optimized"]
                    - row["vtvf_cross_error_addressed_fixed"],
                    "delta_all_error_addressed_optimized_minus_fixed": row["all_error_addressed_optimized"]
                    - row["all_error_addressed_fixed"],
                    "delta_auto_vtvf_unresolved_optimized_minus_fixed": row[
                        "automatic_unresolved_vtvf_cross_error_rate_optimized"
                    ]
                    - row["automatic_unresolved_vtvf_cross_error_rate_fixed"],
                }
            )
        fixed_vs_optimized = pd.DataFrame(fixed_vs_optimized_rows)
        fixed_vs_optimized.to_csv(out_dir / "paired_optimized_vs_fixed_mechanism_policy.csv", index=False)
        _mean_std(
            fixed_vs_optimized,
            ["budget"],
            [
                "delta_action_rate_optimized_minus_fixed",
                "delta_vtvf_cross_addressed_optimized_minus_fixed",
                "delta_all_error_addressed_optimized_minus_fixed",
                "delta_auto_vtvf_unresolved_optimized_minus_fixed",
            ],
        ).to_csv(out_dir / "paired_optimized_vs_fixed_mechanism_policy_mean_std.csv", index=False)

    return {
        "n_mechanism_policy_rows": int(len(mechanism_policy)),
        "n_optimized_mechanism_policy_rows": int(len(optimized_mechanism_policy)),
        "n_total_policy_rows": int(len(total_policy)),
        "n_head_rows": int(len(heads)),
        "n_ablation_rows": int(len(ablations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_20260627"))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = _resolve_runs()
    run_manifest = []
    for seed in SEEDS:
        result = _run_seed(
            seed,
            pairs[seed]["teacher"],
            pairs[seed]["second"],
            args.out,
            args.budgets,
            force=args.force,
        )
        result["teacher"] = str(pairs[seed]["teacher"])
        result["second"] = str(pairs[seed]["second"])
        run_manifest.append(result)
        print(json.dumps(result, ensure_ascii=False))

    aggregate = _aggregate(args.out)
    report = {
        "out": str(args.out),
        "budgets": args.budgets,
        "seeds": SEEDS,
        "runs": run_manifest,
        "aggregate": aggregate,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
