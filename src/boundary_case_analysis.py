from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CLASS_NAMES, load_rhythm_windows, make_splits


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    scores = pd.read_csv(run_dir / "uncertainty_scores.csv")
    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids)
    x_test, y_test = splits.x_test, splits.y_test

    boundary = scores[((scores["y_true"] == 1) & (scores["y_pred"] == 2)) | ((scores["y_true"] == 2) & (scores["y_pred"] == 1))].copy()
    boundary["case_type"] = boundary.apply(
        lambda r: f"{CLASS_NAMES[int(r['y_true'])]}_as_{CLASS_NAMES[int(r['y_pred'])]}", axis=1
    )
    sort_col = "hybrid_entropy_knn" if "hybrid_entropy_knn" in boundary.columns else "knn"
    boundary = boundary.sort_values(sort_col, ascending=False).head(args.top_k)
    boundary.to_csv(run_dir / "vt_vf_boundary_cases.csv", index=False)

    out_dir = run_dir / "boundary_waveforms"
    out_dir.mkdir(exist_ok=True)
    for rank, idx in enumerate(boundary.index, start=1):
        signal = x_test[idx, 0]
        row = boundary.loc[idx]
        plt.figure(figsize=(7, 2.4))
        plt.plot(np.arange(signal.size) / 100.0, signal, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalised ECG")
        plt.title(
            f"{rank:02d}: true {CLASS_NAMES[int(row['y_true'])]}, "
            f"pred {CLASS_NAMES[int(row['y_pred'])]}, {sort_col}={row[sort_col]:.3f}"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"case_{rank:02d}_{row['case_type']}.png", dpi=180)
        plt.close()

    print(boundary[["y_true", "y_pred", "case_type", sort_col]].head(args.top_k))


if __name__ == "__main__":
    main()
