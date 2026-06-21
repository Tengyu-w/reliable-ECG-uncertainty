from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_curves(run_dir: Path, seed: int, version: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "review_curves.csv")
    frame = frame[frame["review_burden"].isin([0.10, 0.20, 0.30])].copy()
    frame.insert(0, "version", version)
    frame.insert(0, "seed", seed)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--deployable-full-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/risk_version_comparison_20260620")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.core_manifest)
    oracle = manifest[manifest["stage"].eq("risk_aligned_distillation")].set_index("seed")
    frames = []
    for seed in [42, 43, 44]:
        frames.append(
            _read_curves(Path(str(oracle.loc[seed, "run_dir"])), seed, "legacy_oracle_style")
        )
        full_dir = Path(
            (
                args.deployable_full_root / f"seed{seed}" / "heads" / "latest"
            ).read_text(encoding="utf-8").strip()
        )
        frames.append(_read_curves(full_dir, seed, "deployable_hand_weighted"))
        selected_dir = Path(
            (args.selected_root / f"seed{seed}" / "heads" / "latest").read_text(
                encoding="utf-8"
            ).strip()
        )
        frames.append(_read_curves(selected_dir, seed, "deployable_validation_selected"))
    seed_df = pd.concat(frames, ignore_index=True)
    seed_df.to_csv(args.out / "risk_versions_seed_level.csv", index=False)
    summary = (
        seed_df.groupby(["version", "review_burden"])[
            [
                "all_error_captured",
                "vtvf_error_captured",
                "auto_error_rate",
                "auto_vtvf_error_rate",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(args.out / "risk_versions_mean_std.csv", index=False)

    plt.figure(figsize=(8, 5))
    for version, sub in summary.groupby("version"):
        plt.errorbar(
            sub["review_burden"],
            sub["vtvf_error_captured_mean"],
            yerr=sub["vtvf_error_captured_std"],
            marker="o",
            capsize=3,
            label=version,
        )
    plt.xlabel("Review burden")
    plt.ylabel("VT/VF cross-error capture")
    plt.ylim(0, 1.05)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "risk_version_vtvf_capture.png", dpi=200)
    plt.close()
    print(args.out)


if __name__ == "__main__":
    main()
