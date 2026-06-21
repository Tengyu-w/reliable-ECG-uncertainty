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


def _curve(df: pd.DataFrame, burdens: np.ndarray) -> pd.DataFrame:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    any_error = y_true != y_pred
    vtvf_error = ((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))
    ventricular_error = np.isin(y_true, [1, 2]) & any_error
    order = np.argsort(-df["risk_score"].to_numpy())
    rows = []
    total_errors = max(int(any_error.sum()), 1)
    total_v_errors = max(int(ventricular_error.sum()), 1)
    total_vtvf = max(int(vtvf_error.sum()), 1)
    for burden in burdens:
        n_review = max(1, int(round(len(df) * burden)))
        review_idx = order[:n_review]
        auto_idx = order[n_review:]
        rows.append(
            {
                "score": "risk_head",
                "review_burden": float(burden),
                "reviewed": int(n_review),
                "auto_coverage": float(len(auto_idx) / len(df)),
                "all_error_captured": float(any_error[review_idx].sum() / total_errors),
                "ventricular_error_captured": float(ventricular_error[review_idx].sum() / total_v_errors),
                "vtvf_error_captured": float(vtvf_error[review_idx].sum() / total_vtvf),
                "auto_error_rate": float(any_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                "auto_vtvf_error_rate": float(vtvf_error[auto_idx].mean()) if len(auto_idx) else float("nan"),
                "review_error_enrichment": float(any_error[review_idx].mean() / max(any_error.mean(), 1e-8)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    df = pd.read_csv(run_dir / "risk_head_scores_test.csv")
    burdens = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    curves = _curve(df, burdens)
    curves.to_csv(run_dir / "risk_head_review_curves.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(curves["review_burden"], curves["vtvf_error_captured"], marker="o", label="VT/VF errors")
    plt.plot(curves["review_burden"], curves["all_error_captured"], marker="o", label="all errors")
    plt.xlabel("Review burden")
    plt.ylabel("Error captured")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "risk_head_review_burden.png", dpi=180)
    plt.close()
    print(curves[curves["review_burden"].isin([0.1, 0.2, 0.3])])


if __name__ == "__main__":
    main()
