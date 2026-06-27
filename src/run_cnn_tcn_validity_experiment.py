from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CNN+TCN validity-bottleneck model.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--out", type=Path, default=Path("results/cnn_tcn_validity_20260626"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--risk-targets", type=Path, default=None)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--split-grouping", choices=["record", "duplicate_family"], default="duplicate_family")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_rows = ["seed,run_dir"]

    for seed in args.seeds:
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--mat",
            str(args.mat),
            "--model",
            "cnn_tcn_validity",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--split-grouping",
            args.split_grouping,
            "--out",
            str(args.out),
            "--run-suffix",
            f"validity_seed{seed}",
            "--aux-boundary-weight",
            "0.10",
            "--gate-target-weight",
            "0.05",
            "--gate-sparsity-weight",
            "0.01",
            "--stability-consistency-weight",
            "0.04",
            "--embedding-consistency-weight",
            "0.01",
            "--vtvf-readability-weight",
            "0.04",
        ]
        if args.max_windows_per_record is not None:
            cmd.extend(["--max-windows-per-record", str(args.max_windows_per_record)])
        if args.risk_targets is not None:
            cmd.extend(
                [
                    "--risk-targets",
                    str(args.risk_targets),
                    "--risk-boundary-weight",
                    "0.15",
                    "--risk-gate-weight",
                    "0.05",
                    "--risk-entropy-weight",
                    "0.04",
                    "--boundary-ce-weight",
                    "0.30",
                    "--anti-confident-risk-weight",
                    "0.04",
                    "--selective-stability-consistency-weight",
                    "0.05",
                    "--selective-embedding-consistency-weight",
                    "0.01",
                ]
            )
        _run(cmd)
        latest = args.out / "latest"
        run_dir = latest.read_text(encoding="utf-8").strip() if latest.exists() else ""
        manifest_rows.append(f"{seed},{run_dir}")

    manifest = args.out / f"cnn_tcn_validity_manifest_{stamp}.csv"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
