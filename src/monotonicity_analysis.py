from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    df = pd.read_csv(run_dir / "corruption_severity_metrics.csv")
    rows = []
    for (corruption, score), sub in df.groupby(["corruption", "score"]):
        sub = sub.sort_values("severity")
        rho_mean, p_mean = spearmanr(sub["severity"], sub["ood_mean"])
        rho_auroc, p_auroc = spearmanr(sub["severity"], sub["auroc"])
        slope = np.polyfit(sub["severity"], sub["ood_mean"], 1)[0]
        rows.append(
            {
                "corruption": corruption,
                "score": score,
                "spearman_ood_mean": float(rho_mean),
                "p_ood_mean": float(p_mean),
                "spearman_auroc": float(rho_auroc),
                "p_auroc": float(p_auroc),
                "ood_mean_slope": float(slope),
                "severity1_auroc": float(sub[sub["severity"] == 1]["auroc"].iloc[0]),
                "severity4_auroc": float(sub[sub["severity"] == 4]["auroc"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "severity_monotonicity.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
