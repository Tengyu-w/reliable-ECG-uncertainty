from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _quadrant_label(atypicality: np.ndarray, ambiguity: np.ndarray, atyp_q: float, amb_q: float) -> np.ndarray:
    high_a = atypicality >= atyp_q
    high_b = ambiguity >= amb_q
    labels = np.full(len(atypicality), "typical_auto_classify", dtype=object)
    labels[~high_a & high_b] = "boundary_expert_review"
    labels[high_a & ~high_b] = "ood_signal_quality_review"
    labels[high_a & high_b] = "high_risk_forced_review"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--atypicality", choices=["knn", "mahalanobis"], default="knn")
    parser.add_argument("--ambiguity", choices=["softmax", "prototype", "knn_mix", "vai"], default="softmax")
    parser.add_argument("--quantile", type=float, default=0.8)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    ambiguity_df = pd.read_csv(run_dir / "ambiguity_scores.csv")
    uncertainty_df = pd.read_csv(run_dir / "uncertainty_scores.csv")

    atyp_raw = uncertainty_df[args.atypicality].to_numpy()
    ambiguity_col = {
        "softmax": "softmax_vtvf_ambiguity",
        "prototype": "prototype_vtvf_ambiguity",
        "knn_mix": "knn_vtvf_mix",
        "vai": "ventricular_ambiguity_index",
    }[args.ambiguity]
    amb_raw = ambiguity_df[ambiguity_col].to_numpy()

    atyp = _normalise(atyp_raw)
    amb = _normalise(amb_raw)
    atyp_q = float(np.quantile(atyp, args.quantile))
    amb_q = float(np.quantile(amb, args.quantile))
    labels = _quadrant_label(atyp, amb, atyp_q, amb_q)

    out = pd.DataFrame(
        {
            "y_true": ambiguity_df["y_true"],
            "y_pred": ambiguity_df["y_pred"],
            "atypicality_score": atyp,
            "boundary_ambiguity_score": amb,
            "reliability_pathway": labels,
            "is_error": ambiguity_df["is_any_error"],
            "is_vtvf_boundary_error": ambiguity_df["is_vtvf_boundary_error"],
        }
    )
    out.to_csv(run_dir / "reliability_map_scores.csv", index=False)

    rows = []
    for label, sub in out.groupby("reliability_pathway"):
        rows.append(
            {
                "pathway": label,
                "n": int(len(sub)),
                "fraction": float(len(sub) / len(out)),
                "error_rate": float(sub["is_error"].mean()),
                "vtvf_boundary_error_rate": float(sub["is_vtvf_boundary_error"].mean()),
                "sr_fraction": float((sub["y_true"] == 0).mean()),
                "vt_fraction": float((sub["y_true"] == 1).mean()),
                "vf_fraction": float((sub["y_true"] == 2).mean()),
                "mean_atypicality": float(sub["atypicality_score"].mean()),
                "mean_boundary_ambiguity": float(sub["boundary_ambiguity_score"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("pathway")
    summary.to_csv(run_dir / "reliability_map_summary.csv", index=False)

    color_map = {
        "typical_auto_classify": "tab:green",
        "boundary_expert_review": "tab:orange",
        "ood_signal_quality_review": "tab:purple",
        "high_risk_forced_review": "tab:red",
    }
    plt.figure(figsize=(6, 5))
    for label, sub in out.groupby("reliability_pathway"):
        plt.scatter(
            sub["atypicality_score"],
            sub["boundary_ambiguity_score"],
            s=12,
            alpha=0.65,
            label=label,
            c=color_map.get(label, "gray"),
        )
    plt.axvline(atyp_q, color="black", linestyle="--", linewidth=1)
    plt.axhline(amb_q, color="black", linestyle="--", linewidth=1)
    plt.xlabel(f"Representation atypicality ({args.atypicality})")
    plt.ylabel(f"VT/VF boundary ambiguity ({args.ambiguity})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "atypicality_vs_ambiguity_map.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    is_error = out["is_error"].to_numpy()
    plt.scatter(atyp[~is_error], amb[~is_error], s=10, alpha=0.35, label="correct", c="lightgray")
    plt.scatter(atyp[is_error], amb[is_error], s=18, alpha=0.8, label="error", c="tab:red")
    plt.axvline(atyp_q, color="black", linestyle="--", linewidth=1)
    plt.axhline(amb_q, color="black", linestyle="--", linewidth=1)
    plt.xlabel(f"Representation atypicality ({args.atypicality})")
    plt.ylabel(f"VT/VF boundary ambiguity ({args.ambiguity})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "atypicality_vs_ambiguity_errors.png", dpi=180)
    plt.close()

    print(summary)


if __name__ == "__main__":
    main()
