from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    scores = pd.read_csv(run_dir / "uncertainty_scores.csv")
    if "hybrid_entropy_knn" not in scores.columns and {"entropy", "knn"}.issubset(scores.columns):
        e = (scores["entropy"] - scores["entropy"].min()) / max(scores["entropy"].max() - scores["entropy"].min(), 1e-12)
        k = (scores["knn"] - scores["knn"].min()) / max(scores["knn"].max() - scores["knn"].min(), 1e-12)
        scores["hybrid_entropy_knn"] = 0.5 * e + 0.5 * k

    y = scores["y_true"].to_numpy()
    pred = scores["y_pred"].to_numpy()
    rows = []
    for score in ["msp", "entropy", "mahalanobis", "knn", "hybrid_entropy_knn"]:
        if score not in scores:
            continue
        order = np.argsort(scores[score].to_numpy())
        for coverage in np.linspace(0.5, 1.0, 11):
            keep = order[: max(1, int(round(len(order) * coverage)))]
            yk, pk = y[keep], pred[keep]
            row = {
                "score": score,
                "coverage": float(coverage),
                "accepted": int(len(keep)),
                "accuracy": float(np.mean(yk == pk)),
                "vt_to_vf": float(np.mean(pk[yk == 1] == 2)) if np.any(yk == 1) else np.nan,
                "vf_to_vt": float(np.mean(pk[yk == 2] == 1)) if np.any(yk == 2) else np.nan,
            }
            for cls, name in [(0, "sr"), (1, "vt"), (2, "vf")]:
                mask = yk == cls
                row[f"{name}_coverage_within_kept"] = float(mask.mean())
                row[f"{name}_recall"] = float(np.mean(pk[mask] == cls)) if mask.any() else np.nan
                row[f"{name}_kept"] = int(mask.sum())
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "per_class_selective.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
