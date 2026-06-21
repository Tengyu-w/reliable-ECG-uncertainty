from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


VARIANTS = {
    "entropy_only": {
        "entropy-weight": 1.0,
        "local-instability-weight": 0.0,
        "vtvf-mixing-weight": 0.0,
        "knn-weight": 0.0,
        "boundary-weight": 0.0,
    },
    "neighborhood_only": {
        "entropy-weight": 0.0,
        "local-instability-weight": 0.6,
        "vtvf-mixing-weight": 0.0,
        "knn-weight": 0.4,
        "boundary-weight": 0.0,
    },
    "boundary_only": {
        "entropy-weight": 0.0,
        "local-instability-weight": 0.0,
        "vtvf-mixing-weight": 0.0,
        "knn-weight": 0.0,
        "boundary-weight": 1.0,
    },
    "vtvf_mixing_only": {
        "entropy-weight": 0.0,
        "local-instability-weight": 0.0,
        "vtvf-mixing-weight": 1.0,
        "knn-weight": 0.0,
        "boundary-weight": 0.0,
    },
}


def _run_job(
    python: str,
    teacher: Path,
    seed: int,
    variant: str,
    weights: dict[str, float],
    epochs: int,
    root: Path,
) -> dict[str, str | int]:
    variant_dir = root / f"seed{seed}" / variant
    target_path = variant_dir / "targets.npz"
    components_dir = variant_dir / "components"
    heads_dir = variant_dir / "heads"
    variant_dir.mkdir(parents=True, exist_ok=True)

    generate_cmd = [
        python,
        "-m",
        "src.generate_risk_targets",
        "--teacher-run-dir",
        str(teacher),
        "--out",
        str(target_path),
        "--components-out-dir",
        str(components_dir),
    ]
    for name, value in weights.items():
        generate_cmd.extend([f"--{name}", str(value)])
    subprocess.run(generate_cmd, check=True)

    train_cmd = [
        python,
        "-m",
        "src.train_embedding_risk_head",
        "--teacher-run-dir",
        str(teacher),
        "--risk-targets",
        str(target_path),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--out",
        str(heads_dir),
        "--run-suffix",
        f"deployable_{variant}_seed{seed}",
    ]
    subprocess.run(train_cmd, check=True)
    run_dir = Path((heads_dir / "latest").read_text(encoding="utf-8").strip())
    return {
        "seed": seed,
        "variant": variant,
        "teacher_run_dir": str(teacher),
        "target_path": str(target_path),
        "components_dir": str(components_dir),
        "risk_head_run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/risk_deployable_ablation_20260620")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    teachers = manifest[manifest["stage"].eq("regularity_feature_injection")]
    jobs = [
        (
            sys.executable,
            Path(str(row["run_dir"])),
            int(row["seed"]),
            variant,
            weights,
            args.epochs,
            args.out,
        )
        for _, row in teachers.iterrows()
        for variant, weights in VARIANTS.items()
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_run_job, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(json.dumps(result, indent=2), flush=True)
            pd.DataFrame(rows).sort_values(["seed", "variant"]).to_csv(
                args.out / "deployable_risk_ablation_manifest.csv", index=False
            )
    print(args.out)


if __name__ == "__main__":
    main()
