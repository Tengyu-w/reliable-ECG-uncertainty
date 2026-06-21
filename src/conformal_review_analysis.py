from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import softmax


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _parse_set(text: str) -> set[int]:
    text = str(text).strip().strip("{}")
    if not text:
        return set()
    return {int(x) for x in text.split(",") if x.strip()}


def _natural_review_metrics(sub: pd.DataFrame, review_mask: np.ndarray) -> dict[str, float | int]:
    y = sub["y_true"].to_numpy(int)
    pred = sub["y_pred"].to_numpy(int)
    any_error = y != pred
    vtvf_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    auto_mask = ~review_mask
    return {
        "reviewed": int(review_mask.sum()),
        "review_rate": float(review_mask.mean()),
        "all_error_captured": float((review_mask & any_error).sum() / max(any_error.sum(), 1)),
        "vtvf_error_captured": float((review_mask & vtvf_error).sum() / max(vtvf_error.sum(), 1)),
        "auto_error_rate": float(any_error[auto_mask].mean()) if auto_mask.any() else np.nan,
        "auto_vtvf_error_rate": float(vtvf_error[auto_mask].mean()) if auto_mask.any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    set_path = run_dir / "conformal_sets.csv"
    summary_path = run_dir / "conformal_summary.csv"
    if not set_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run src.conformal_analysis first to create conformal_sets.csv and conformal_summary.csv")

    sets = pd.read_csv(set_path)
    summary = pd.read_csv(summary_path)
    logits = np.load(run_dir / "embeddings_test.npz")["logits"]
    probs = softmax(logits)
    max_prob = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(probs.shape[1])
    rows = []
    for (method, alpha), sub in sets.groupby(["method", "alpha"]):
        idx = sub["index"].to_numpy(int)
        parsed = sub["set"].map(_parse_set)
        set_size = sub["set_size"].to_numpy(int)
        y = sub["y_true"].to_numpy(int)
        pred = sub["y_pred"].to_numpy(int)
        contains_true = sub["contains_true"].to_numpy(bool)
        non_singleton = set_size > 1
        vtvf_pair = parsed.map(lambda s: s == {1, 2}).to_numpy(bool)
        ventricular_multi = parsed.map(lambda s: len(s) > 1 and s.issubset({1, 2})).to_numpy(bool)
        miss_true = ~contains_true

        policies = {
            "non_singleton_set": non_singleton,
            "vtvf_pair_set": vtvf_pair,
            "ventricular_multi_set": ventricular_multi,
            "invalid_or_non_singleton": miss_true | non_singleton,
        }
        for policy, mask in policies.items():
            rows.append(
                {
                    "method": method,
                    "alpha": float(alpha),
                    "policy": policy,
                    **_natural_review_metrics(sub, mask),
                }
            )

        # Convert conformal output into a ranked review score. Set size is the
        # conformal part; entropy and max probability only break ties among
        # equally-sized sets.
        score = set_size.astype(float) + 2.0 * vtvf_pair.astype(float) + 0.20 * entropy[idx] + 0.10 * (1.0 - max_prob[idx])
        order = np.argsort(-score)
        any_error = y != pred
        vtvf_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
        for burden in [0.10, 0.20, 0.30]:
            n_review = int(round(len(sub) * burden))
            review_idx = order[:n_review]
            auto_idx = order[n_review:]
            rows.append(
                {
                    "method": method,
                    "alpha": float(alpha),
                    "policy": f"ranked_conformal_score_top_{int(burden * 100)}pct",
                    "reviewed": int(n_review),
                    "review_rate": float(burden),
                    "all_error_captured": float(any_error[review_idx].sum() / max(any_error.sum(), 1)),
                    "vtvf_error_captured": float(vtvf_error[review_idx].sum() / max(vtvf_error.sum(), 1)),
                    "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else np.nan,
                    "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "conformal_review_summary.csv", index=False)

    plot_df = out[out["policy"].str.startswith("ranked_conformal_score_top_")].copy()
    if not plot_df.empty:
        plot_df["label"] = plot_df["method"] + " alpha=" + plot_df["alpha"].astype(str)
        plt.figure(figsize=(7.0, 4.5))
        for label in plot_df["label"].unique():
            sub = plot_df[plot_df["label"] == label].sort_values("review_rate")
            plt.plot(sub["review_rate"], sub["vtvf_error_captured"], marker="o", label=label)
        plt.xlabel("Review burden")
        plt.ylabel("VT/VF error capture")
        plt.ylim(0, 1.02)
        plt.grid(True, color="#e5e7eb", linewidth=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(run_dir / "conformal_review_vtvf_capture.png", dpi=180)
        plt.close()

    merged = summary.merge(
        out[out["policy"].isin(["non_singleton_set", "vtvf_pair_set"])],
        on=["method", "alpha"],
        how="left",
        suffixes=("", "_review"),
    )
    merged.to_csv(run_dir / "conformal_summary_with_review.csv", index=False)
    print(out.sort_values(["method", "alpha", "policy"]).head(30))


if __name__ == "__main__":
    main()
