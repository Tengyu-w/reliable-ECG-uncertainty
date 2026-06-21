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
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _score_table(run_dir: Path, k: int) -> pd.DataFrame:
    uncertainty = pd.read_csv(run_dir / "uncertainty_scores.csv")
    ambiguity = pd.read_csv(run_dir / "ambiguity_scores.csv")
    neigh_path = run_dir / f"embedding_neighborhood_k{k}.csv"
    if not neigh_path.exists():
        raise FileNotFoundError(f"Run embedding_geometry_analysis.py first: {neigh_path}")
    neigh = pd.read_csv(neigh_path)

    y_true = uncertainty["y_true"].to_numpy()
    y_pred = uncertainty["y_pred"].to_numpy()
    any_error = y_true != y_pred
    vtvf_error = ((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))
    ventricular_error = np.isin(y_true, [1, 2]) & any_error

    entropy = _normalise(uncertainty["entropy"].to_numpy())
    knn = _normalise(uncertainty["knn"].to_numpy())
    maha = _normalise(uncertainty["mahalanobis"].to_numpy())
    boundary = _normalise(ambiguity["softmax_vtvf_ambiguity"].to_numpy())
    neigh_instability = _normalise(1.0 - neigh["local_purity"].to_numpy())
    vtvf_mixing = _normalise(neigh["vtvf_mixing"].to_numpy())

    lr_instability = (
        0.25 * entropy
        + 0.20 * knn
        + 0.15 * maha
        + 0.20 * boundary
        + 0.10 * neigh_instability
        + 0.10 * vtvf_mixing
    )
    boundary_focused_lrii = 0.35 * boundary + 0.25 * vtvf_mixing + 0.20 * entropy + 0.20 * neigh_instability
    atypicality_focused_lrii = 0.40 * knn + 0.30 * maha + 0.20 * entropy + 0.10 * neigh_instability

    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "any_error": any_error,
            "ventricular_error": ventricular_error,
            "vtvf_error": vtvf_error,
            "entropy": entropy,
            "msp": _normalise(uncertainty["msp"].to_numpy()),
            "knn": knn,
            "mahalanobis": maha,
            "softmax_vtvf_ambiguity": boundary,
            "local_instability": neigh_instability,
            "vtvf_mixing": vtvf_mixing,
            "lrii": lr_instability,
            "boundary_lrii": boundary_focused_lrii,
            "atypicality_lrii": atypicality_focused_lrii,
        }
    )


def _curve(df: pd.DataFrame, score: str, burdens: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(-df[score].to_numpy())
    n = len(df)
    total_errors = max(int(df["any_error"].sum()), 1)
    total_v_errors = max(int(df["ventricular_error"].sum()), 1)
    total_vtvf = max(int(df["vtvf_error"].sum()), 1)
    rows = []
    for burden in burdens:
        n_review = max(1, int(round(n * burden)))
        review_idx = order[:n_review]
        auto_idx = order[n_review:]
        review = df.iloc[review_idx]
        auto = df.iloc[auto_idx]
        rows.append(
            {
                "score": score,
                "review_burden": float(burden),
                "reviewed": int(n_review),
                "auto_coverage": float(len(auto_idx) / n),
                "all_error_captured": float(review["any_error"].sum() / total_errors),
                "ventricular_error_captured": float(review["ventricular_error"].sum() / total_v_errors),
                "vtvf_error_captured": float(review["vtvf_error"].sum() / total_vtvf),
                "auto_error_rate": float(auto["any_error"].mean()) if len(auto) else float("nan"),
                "auto_vtvf_error_rate": float(auto["vtvf_error"].mean()) if len(auto) else float("nan"),
                "review_error_enrichment": float(review["any_error"].mean() / max(df["any_error"].mean(), 1e-8)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    df = _score_table(run_dir, args.k)
    df.to_csv(run_dir / "local_rhythm_instability_scores.csv", index=False)

    burdens = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    score_cols = [
        "entropy",
        "knn",
        "mahalanobis",
        "softmax_vtvf_ambiguity",
        "local_instability",
        "vtvf_mixing",
        "lrii",
        "boundary_lrii",
        "atypicality_lrii",
    ]
    rows = []
    for score in score_cols:
        rows.extend(_curve(df, score, burdens))
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "review_efficiency_curves.csv", index=False)

    plt.figure(figsize=(7, 5))
    for score in ["entropy", "knn", "softmax_vtvf_ambiguity", "lrii", "boundary_lrii"]:
        sub = out[out["score"] == score]
        plt.plot(sub["review_burden"], sub["vtvf_error_captured"], marker="o", label=score)
    plt.xlabel("Review burden")
    plt.ylabel("VT/VF error captured")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "review_burden_vs_vtvf_error_captured.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for score in ["entropy", "knn", "mahalanobis", "lrii", "atypicality_lrii"]:
        sub = out[out["score"] == score]
        plt.plot(sub["review_burden"], sub["all_error_captured"], marker="o", label=score)
    plt.xlabel("Review burden")
    plt.ylabel("All errors captured")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "review_burden_vs_all_error_captured.png", dpi=180)
    plt.close()

    print(out[out["review_burden"].isin([0.1, 0.2, 0.3])].sort_values(["review_burden", "vtvf_error_captured"], ascending=[True, False]))


if __name__ == "__main__":
    main()
