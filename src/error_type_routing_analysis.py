from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ERROR_TYPES = [
    (0, 1, "SR_to_VT"),
    (0, 2, "SR_to_VF"),
    (1, 0, "VT_to_SR"),
    (2, 0, "VF_to_SR"),
    (1, 2, "VT_to_VF"),
    (2, 1, "VF_to_VT"),
]


def _counts(predictions: pd.DataFrame) -> dict[str, int]:
    y_true = predictions["y_true"].to_numpy(int)
    y_pred = predictions["y_pred"].to_numpy(int)
    return {
        name: int(((y_true == true_label) & (y_pred == pred_label)).sum())
        for true_label, pred_label, name in ERROR_TYPES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate six directional ECG errors and quantify PRO changes and RISK capture."
    )
    parser.add_argument("--mitigation-summary", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/error_type_routing_20260620"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mitigation = pd.read_csv(args.mitigation_summary)
    selected = mitigation[
        mitigation["variant"].isin(["baseline", "prototype_separation"])
    ].copy()
    classification_rows = []
    for _, row in selected.iterrows():
        run_dir = Path(str(row["run_dir"]))
        counts = _counts(pd.read_csv(run_dir / "test_predictions.csv"))
        for error_type, count in counts.items():
            classification_rows.append(
                {
                    "seed": int(row["seed"]),
                    "variant": str(row["variant"]),
                    "error_type": error_type,
                    "count": count,
                    "run_dir": str(run_dir),
                }
            )
    classification_df = pd.DataFrame(classification_rows)
    classification_df.to_csv(args.out / "baseline_vs_pro_error_counts_seed_level.csv", index=False)

    paired = classification_df.pivot_table(
        index=["seed", "error_type"], columns="variant", values="count"
    ).reset_index()
    paired["pro_minus_baseline"] = paired["prototype_separation"] - paired["baseline"]
    paired["relative_change"] = paired["pro_minus_baseline"] / paired["baseline"].replace(0, np.nan)
    paired.to_csv(args.out / "baseline_vs_pro_error_changes_paired.csv", index=False)
    paired.groupby("error_type")[["baseline", "prototype_separation", "pro_minus_baseline"]].agg(
        ["mean", "std"]
    ).to_csv(args.out / "baseline_vs_pro_error_changes_mean_std.csv")

    manifest = pd.read_csv(args.core_manifest)
    teachers = manifest[manifest["stage"].eq("regularity_feature_injection")].set_index("seed")
    risks = manifest[manifest["stage"].eq("risk_aligned_distillation")].set_index("seed")
    routing_rows = []
    for seed in sorted(set(teachers.index) & set(risks.index)):
        teacher_dir = Path(str(teachers.loc[seed, "run_dir"]))
        risk_dir = Path(str(risks.loc[seed, "run_dir"]))
        pred = pd.read_csv(teacher_dir / "test_predictions.csv")
        risk = pd.read_csv(risk_dir / "risk_scores_test.csv")
        if not np.array_equal(pred["y_true"].to_numpy(), risk["y_true"].to_numpy()):
            raise ValueError(f"Prediction/RISK mismatch for seed {seed}")
        y_true = pred["y_true"].to_numpy(int)
        y_pred = pred["y_pred"].to_numpy(int)
        order = np.argsort(-risk["risk_score"].to_numpy(float))
        for burden in [0.10, 0.20, 0.30]:
            reviewed = np.zeros(len(pred), dtype=bool)
            reviewed[order[: max(1, int(round(len(pred) * burden)))]] = True
            for true_label, pred_label, error_type in ERROR_TYPES:
                mask = (y_true == true_label) & (y_pred == pred_label)
                routing_rows.append(
                    {
                        "seed": int(seed),
                        "review_burden": burden,
                        "error_type": error_type,
                        "total_errors": int(mask.sum()),
                        "captured_errors": int((mask & reviewed).sum()),
                        "capture_rate": float((mask & reviewed).sum() / mask.sum())
                        if mask.any()
                        else np.nan,
                        "residual_auto_errors": int((mask & ~reviewed).sum()),
                    }
                )
    routing_df = pd.DataFrame(routing_rows)
    routing_df.to_csv(args.out / "risk_capture_by_error_type_seed_level.csv", index=False)
    routing_summary = (
        routing_df.groupby(["review_burden", "error_type"])
        .agg(
            n_seeds=("seed", "nunique"),
            total_errors_mean=("total_errors", "mean"),
            capture_rate_mean=("capture_rate", "mean"),
            capture_rate_std=("capture_rate", "std"),
            residual_auto_errors_mean=("residual_auto_errors", "mean"),
        )
        .reset_index()
    )
    routing_summary.to_csv(args.out / "risk_capture_by_error_type_mean_std.csv", index=False)

    plot = routing_summary[routing_summary["review_burden"].eq(0.20)]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(
        plot["error_type"],
        plot["capture_rate_mean"],
        yerr=plot["capture_rate_std"].fillna(0),
        capsize=3,
        color=["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"],
    )
    plt.ylabel("RISK capture rate at 20% review burden")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", color="#dddddd", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(args.out / "risk_capture_six_error_types_20pct.png", dpi=200)
    plt.close()

    notes = [
        "# Six-direction error analysis",
        "",
        "- Baseline versus PRO counts come from the paired mitigation experiments.",
        "- RISK capture comes from the separate three-seed core-validation teacher/RISK experiments.",
        "- These two tables answer different questions and must not be presented as one end-to-end jointly trained system.",
        "- VT->VF and VF->VT are the clinically emphasized boundary cross-errors.",
        "- SR-related directional errors remain important for checking whether a boundary-focused method shifts mistakes elsewhere.",
    ]
    (args.out / "analysis_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
