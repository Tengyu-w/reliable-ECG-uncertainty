from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_set(text: str) -> set[int]:
    text = str(text).strip().strip("{}")
    return {int(value) for value in text.split(",") if value.strip()}


def _review_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    burden: float,
) -> dict[str, float | int]:
    any_error = y_true != y_pred
    vtvf_error = ((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))
    order = np.argsort(-np.asarray(score, dtype=float))
    n_review = max(1, int(round(len(order) * burden)))
    review_idx = order[:n_review]
    auto_idx = order[n_review:]
    return {
        "reviewed": n_review,
        "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
        "vtvf_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
        "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else np.nan,
        "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else np.nan,
        "review_error_enrichment": float(
            any_error[review_idx].mean() / max(any_error.mean(), 1e-8)
        ),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _conformal_score(run_dir: Path, method: str, alpha: float) -> np.ndarray:
    sets = pd.read_csv(run_dir / "conformal_sets.csv")
    sub = sets[
        sets["method"].eq(method) & np.isclose(sets["alpha"].astype(float), alpha)
    ].sort_values("index")
    if sub.empty:
        raise ValueError(f"No conformal rows for method={method}, alpha={alpha}: {run_dir}")
    parsed = sub["set"].map(_parse_set)
    vtvf_pair = parsed.map(lambda value: value == {1, 2}).to_numpy(float)
    set_size = sub["set_size"].to_numpy(float)
    logits = np.load(run_dir / "embeddings_test.npz")["logits"]
    probs = _softmax(logits)
    max_prob = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(
        probs.shape[1]
    )
    # Ranking uses deployable conformal outputs and model confidence only.
    # Test labels and contains_true are deliberately excluded.
    return set_size + 2.0 * vtvf_pair + 0.20 * entropy + 0.10 * (1.0 - max_prob)


def _mean_std(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "all_error_captured",
        "vtvf_error_captured",
        "auto_error_rate",
        "auto_vtvf_error_rate",
        "review_error_enrichment",
    ]
    rows = []
    for (method, burden), sub in df.groupby(["method", "review_burden"]):
        row = {"method": method, "review_burden": burden, "n_seeds": len(sub)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uncertainty and RISK review routing under identical test sets and budgets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/unified_review_budget_20260620"))
    parser.add_argument("--conformal-method", choices=["lac", "aps"], default="lac")
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    teachers = manifest[manifest["stage"].eq("regularity_feature_injection")].set_index("seed")
    risks = manifest[manifest["stage"].eq("risk_aligned_distillation")].set_index("seed")
    seeds = sorted(set(teachers.index) & set(risks.index))
    burdens = [0.10, 0.20, 0.30]
    rows = []

    for seed in seeds:
        teacher_dir = Path(str(teachers.loc[seed, "run_dir"]))
        risk_dir = Path(str(risks.loc[seed, "run_dir"]))
        local = pd.read_csv(teacher_dir / "local_rhythm_instability_scores.csv")
        risk = pd.read_csv(risk_dir / "risk_scores_test.csv")
        if not np.array_equal(local["y_true"].to_numpy(), risk["y_true"].to_numpy()):
            raise ValueError(f"Teacher/RISK y_true mismatch for seed {seed}")
        if not np.array_equal(local["y_pred"].to_numpy(), risk["y_pred"].to_numpy()):
            raise ValueError(f"Teacher/RISK y_pred mismatch for seed {seed}")
        y_true = local["y_true"].to_numpy(int)
        y_pred = local["y_pred"].to_numpy(int)
        score_map = {
            "softmax_uncertainty_1_minus_msp": local["msp"].to_numpy(float),
            "entropy": local["entropy"].to_numpy(float),
            "knn_atypicality": local["knn"].to_numpy(float),
            "boundary_lrii": local["boundary_lrii"].to_numpy(float),
            "multi_source_lrii": local["lrii"].to_numpy(float),
            "RISK_distilled_head": risk["risk_score"].to_numpy(float),
            f"conformal_{args.conformal_method}_alpha_{args.conformal_alpha:.2f}": _conformal_score(
                teacher_dir, args.conformal_method, args.conformal_alpha
            ),
        }
        for method, score in score_map.items():
            for burden in burdens:
                rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "review_burden": burden,
                        "n_test": len(y_true),
                        "n_all_errors": int((y_true != y_pred).sum()),
                        "n_vtvf_errors": int(
                            (((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))).sum()
                        ),
                        **_review_metrics(y_true, y_pred, score, burden),
                    }
                )

    seed_df = pd.DataFrame(rows)
    summary_df = _mean_std(seed_df)
    seed_df.to_csv(args.out / "unified_review_budget_seed_level.csv", index=False)
    summary_df.to_csv(args.out / "unified_review_budget_mean_std.csv", index=False)

    paired_rows = []
    risk_df = seed_df[seed_df["method"].eq("RISK_distilled_head")]
    for method in sorted(set(seed_df["method"]) - {"RISK_distilled_head"}):
        baseline = seed_df[seed_df["method"].eq(method)]
        merged = risk_df.merge(
            baseline,
            on=["seed", "review_burden"],
            suffixes=("_risk", "_baseline"),
        )
        for burden, sub in merged.groupby("review_burden"):
            for metric in ["all_error_captured", "vtvf_error_captured", "auto_error_rate"]:
                diff = sub[f"{metric}_risk"] - sub[f"{metric}_baseline"]
                paired_rows.append(
                    {
                        "baseline_method": method,
                        "review_burden": burden,
                        "metric": metric,
                        "n_paired_seeds": len(diff),
                        "risk_minus_baseline_mean": float(diff.mean()),
                        "risk_minus_baseline_std": float(diff.std(ddof=1)),
                        "seed_differences": ";".join(
                            f"{int(seed)}:{value:.6g}"
                            for seed, value in zip(sub["seed"], diff)
                        ),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(
        args.out / "risk_vs_baselines_paired_differences.csv", index=False
    )

    plt.figure(figsize=(9, 5.5))
    for method, sub in summary_df.groupby("method"):
        plt.errorbar(
            sub["review_burden"],
            sub["vtvf_error_captured_mean"],
            yerr=sub["vtvf_error_captured_std"],
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Review burden")
    plt.ylabel("VT/VF cross-error capture")
    plt.ylim(0, 1.05)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out / "unified_vtvf_capture_mean_std.png", dpi=200)
    plt.close()

    notes = [
        "# Unified fixed-budget review comparison",
        "",
        "- All methods are ranked on the same test windows for each seed.",
        "- Results use paired seeds 42, 43, and 44 and fixed review burdens of 10%, 20%, and 30%.",
        f"- Conformal baseline is fixed in advance as {args.conformal_method.upper()} with alpha={args.conformal_alpha:.2f}; it is not selected by test performance.",
        "- `softmax_uncertainty_1_minus_msp` means 1 minus the maximum softmax probability.",
        "- RISK is a review-priority score, not a disease probability or diagnostic output.",
    ]
    (args.out / "comparison_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
