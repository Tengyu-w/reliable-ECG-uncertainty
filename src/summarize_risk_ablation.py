from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing single-seed RISK target ablations.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/risk_ablation_interpretation_20260620")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    review = pd.read_csv(args.review)
    summary.insert(0, "seed", args.seed)
    review.insert(0, "seed", args.seed)
    summary.to_csv(args.out / "risk_ablation_metrics_seed42.csv", index=False)
    review.to_csv(args.out / "risk_ablation_review_seed42.csv", index=False)

    plt.figure(figsize=(7.5, 4.8))
    for target, sub in review.groupby("target"):
        plt.plot(
            sub["review_burden"],
            sub["vtvf_error_captured"],
            marker="o",
            label=target,
        )
    plt.xlabel("Review burden")
    plt.ylabel("VT/VF cross-error capture")
    plt.ylim(0, 1.05)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "risk_ablation_vtvf_capture.png", dpi=200)
    plt.close()

    at_10 = review[review["review_burden"].eq(0.10)].set_index("target")
    full = at_10.loc["full"]
    best_single = at_10.drop(index="full")["vtvf_error_captured"].idxmax()
    report = [
        "# RISK evidence-source ablation interpretation",
        "",
        f"Evidence scope: one teacher split and one risk-head seed (seed {args.seed}). This is mechanism evidence, not a three-seed robustness claim.",
        "",
        "## Confirmed observations",
        "",
        f"- At 10% review burden, the full multi-source target captures {full['vtvf_error_captured']:.1%} of VT/VF cross-errors.",
        f"- The strongest single-source target at 10% is `{best_single}`, capturing {at_10.loc[best_single, 'vtvf_error_captured']:.1%}.",
        f"- Full supervision captures {full['all_error_captured']:.1%} of all errors at 10% burden and leaves an automatic-route error rate of {full['auto_error_rate']:.2%}.",
        "- Neighborhood-only evidence is strong for general error AUROC, while boundary-only evidence is strong for VT/VF-specific AUROC.",
        "- Entropy-only tracks its own soft target very closely, but target-fitting correlation is not the same as review usefulness.",
        "",
        "## Interpretation",
        "",
        "- The full target is most useful at low review burden because it combines complementary evidence rather than relying on one uncertainty mechanism.",
        "- Boundary evidence focuses on dangerous VT/VF ambiguity; neighborhood evidence identifies locally atypical or unstable embeddings; entropy captures output ambiguity.",
        "- The evidence supports the design logic of reliability-privileged distillation, but it does not prove that every component is necessary across seeds.",
        "",
        "## Limitation and upgrade",
        "",
        "- Repeat the ablation across seeds 42/43/44 before presenting component necessity as a final claim.",
        "- The previous VT/VF-mixing-only target collapsed to zero in its saved metadata, so it must not be interpreted as a valid negative result without debugging the target construction.",
    ]
    (args.out / "risk_ablation_interpretation.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(args.out)


if __name__ == "__main__":
    main()
