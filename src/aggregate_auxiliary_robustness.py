from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(sub))
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate corruption-severity robustness metrics across auxiliary intervention runs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = args.out_dir or args.manifest.parent / "auxiliary_robustness_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in manifest.to_dict(orient="records"):
        run_dir = Path(str(item["run_dir"]))
        path = run_dir / "severity_monotonicity.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["variant"] = item["variant"]
        df["seed"] = int(item["seed"])
        df["run_dir"] = str(run_dir)
        rows.append(df)

    if not rows:
        print(f"No severity_monotonicity.csv files found for {args.manifest}")
        return

    run_level = pd.concat(rows, ignore_index=True)
    run_level.to_csv(out_dir / "severity_monotonicity_run_level.csv", index=False)
    value_cols = [
        "spearman_ood_mean",
        "spearman_auroc",
        "ood_mean_slope",
        "severity1_auroc",
        "severity4_auroc",
    ]
    _mean_std(run_level, ["variant", "score"], value_cols).to_csv(
        out_dir / "severity_monotonicity_by_score_mean_std.csv", index=False
    )
    _mean_std(run_level, ["variant", "corruption", "score"], value_cols).to_csv(
        out_dir / "severity_monotonicity_by_corruption_mean_std.csv", index=False
    )
    print(out_dir)


if __name__ == "__main__":
    main()
