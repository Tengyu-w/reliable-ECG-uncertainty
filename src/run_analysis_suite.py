from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _seed_from_checkpoint(run_dir: Path, default: int = 42) -> int:
    path = run_dir / "best_model.pt"
    if not path.exists():
        return default
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return int(state.get("args", {}).get("seed", default))
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "tcn",
            "resnet1d",
            "inception_time",
            "bigru",
            "regularity_fusion",
            "reliability_gated_fusion",
        ],
        required=True,
    )
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    run_dir = args.run_dir
    seed = _seed_from_checkpoint(run_dir)

    _run([py, "-m", "src.evaluate_uncertainty", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.embedding_geometry_analysis", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.selective_analysis", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.ambiguity_analysis", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.conformal_analysis", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.reliability_map", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.review_efficiency_analysis", "--run-dir", str(run_dir)], root)
    _run([py, "-m", "src.regularity_analysis", "--mat", str(args.mat), "--run-dir", str(run_dir), "--seed", str(seed)], root)
    _run([py, "-m", "src.per_class_selective_analysis", "--run-dir", str(run_dir)], root)

    if not args.skip_boundary:
        _run([py, "-m", "src.boundary_case_analysis", "--mat", str(args.mat), "--run-dir", str(run_dir)], root)

    if not args.skip_corruption:
        _run(
            [
                py,
                "-m",
                "src.evaluate_corruption_severity",
                "--mat",
                str(args.mat),
                "--run-dir",
                str(run_dir),
                "--model",
                args.model,
                "--seed",
                str(seed),
            ],
            root,
        )
        _run([py, "-m", "src.monotonicity_analysis", "--run-dir", str(run_dir)], root)

    if not args.skip_stability:
        _run(
            [
                py,
                "-m",
                "src.stability_aware_analysis",
                "--mat",
                str(args.mat),
                "--run-dir",
                str(run_dir),
                "--model",
                args.model,
                "--seed",
                str(seed),
            ],
            root,
        )
        _run([py, "-m", "src.ambiguity_routing_policy", "--run-dir", str(run_dir)], root)
        _run([py, "-m", "src.runtime_supervisor", "--run-dir", str(run_dir)], root)


if __name__ == "__main__":
    main()
