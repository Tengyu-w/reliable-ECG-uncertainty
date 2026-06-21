from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.manifest.parent / "summary"
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    rows = []
    review_rows = []
    for _, item in manifest.iterrows():
        run_dir = Path(str(item["risk_head_run_dir"]))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["test"]
        rows.append({"seed": int(item["seed"]), "variant": item["variant"], **summary})
        curves = pd.read_csv(run_dir / "review_curves.csv")
        curves = curves[curves["review_burden"].isin([0.10, 0.20, 0.30])].copy()
        curves.insert(0, "variant", item["variant"])
        curves.insert(0, "seed", int(item["seed"]))
        review_rows.append(curves)

    for seed in [42, 43, 44]:
        latest = args.full_root / f"seed{seed}" / "heads" / "latest"
        run_dir = Path(latest.read_text(encoding="utf-8").strip())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["test"]
        rows.append({"seed": seed, "variant": "full", **summary})
        curves = pd.read_csv(run_dir / "review_curves.csv")
        curves = curves[curves["review_burden"].isin([0.10, 0.20, 0.30])].copy()
        curves.insert(0, "variant", "full")
        curves.insert(0, "seed", seed)
        review_rows.append(curves)

    metrics = pd.DataFrame(rows).sort_values(["seed", "variant"])
    reviews = pd.concat(review_rows, ignore_index=True).sort_values(
        ["seed", "variant", "review_burden"]
    )
    metrics.to_csv(out / "deployable_ablation_metrics_seed_level.csv", index=False)
    reviews.to_csv(out / "deployable_ablation_review_seed_level.csv", index=False)

    metric_cols = [col for col in metrics.columns if col not in {"seed", "variant"}]
    metric_summary = metrics.groupby("variant")[metric_cols].agg(["mean", "std"])
    metric_summary.to_csv(out / "deployable_ablation_metrics_mean_std.csv")
    review_summary = (
        reviews.groupby(["variant", "review_burden"])[
            ["all_error_captured", "vtvf_error_captured", "auto_error_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    review_summary.columns = [
        "_".join(str(value) for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in review_summary.columns
    ]
    review_summary.to_csv(out / "deployable_ablation_review_mean_std.csv", index=False)

    plt.figure(figsize=(8, 5))
    for variant, sub in review_summary.groupby("variant"):
        plt.errorbar(
            sub["review_burden"],
            sub["vtvf_error_captured_mean"],
            yerr=sub["vtvf_error_captured_std"],
            marker="o",
            capsize=3,
            label=variant,
        )
    plt.xlabel("Review burden")
    plt.ylabel("VT/VF cross-error capture")
    plt.ylim(0, 1.05)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "deployable_ablation_vtvf_capture.png", dpi=200)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
