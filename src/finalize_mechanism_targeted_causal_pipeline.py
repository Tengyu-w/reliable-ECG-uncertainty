from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _latest_manifest(run_dir: Path) -> Path | None:
    manifests = sorted(run_dir.glob("mechanism_targeted_ablation_manifest_*.csv"))
    return manifests[-1] if manifests else None


def _manifest_status(manifest: Path | None, expected_rows: int) -> dict[str, Any]:
    if manifest is None:
        return {
            "manifest": None,
            "exists": False,
            "completed_rows": 0,
            "total_rows": 0,
            "expected_rows": expected_rows,
            "is_complete": False,
        }
    df = pd.read_csv(manifest)
    completed = int(df["status"].astype(str).eq("completed").sum()) if "status" in df.columns else 0
    return {
        "manifest": str(manifest),
        "exists": True,
        "completed_rows": completed,
        "total_rows": int(len(df)),
        "expected_rows": expected_rows,
        "is_complete": completed >= expected_rows,
        "candidates_seen": sorted(df["candidate"].astype(str).unique().tolist()) if "candidate" in df.columns else [],
        "seeds_seen": sorted([int(x) for x in df["seed"].unique().tolist()]) if "seed" in df.columns else [],
    }


def _run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _write_status(out: Path, status: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "mechanism_targeted_pipeline_finalize_status.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    quant_dir = Path(args.quant_dir)
    root = Path.cwd()
    expected_rows = int(args.expected_candidates) * int(args.expected_seeds)
    deadline = time.time() + (float(args.max_wait_minutes) * 60.0)

    while True:
        manifest = _latest_manifest(run_dir)
        status = _manifest_status(manifest, expected_rows)
        status.update(
            {
                "checked_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "quant_dir": str(quant_dir),
                "wait": bool(args.wait),
            }
        )
        _write_status(quant_dir, status)
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)

        if status["is_complete"] or args.allow_partial:
            break
        if not args.wait:
            return status
        if args.max_wait_minutes > 0 and time.time() >= deadline:
            status["timed_out"] = True
            _write_status(quant_dir, status)
            return status
        time.sleep(float(args.poll_seconds))

    if manifest is None:
        raise FileNotFoundError(f"No mechanism_targeted_ablation_manifest_*.csv found under {run_dir}")

    quant_cmd = [
        sys.executable,
        "-m",
        "src.run_causal_mechanism_quantification",
        "--search-dir",
        str(run_dir),
        "--manifest",
        str(manifest),
        "--out",
        str(quant_dir),
        "--k",
        str(args.k),
    ]
    if not args.skip_quantification:
        _run(quant_cmd, root)

    summary_cmd = [
        sys.executable,
        "-m",
        "src.summarize_mechanism_targeted_causal_quantification",
        "--quant-dir",
        str(quant_dir),
    ]
    if not args.skip_summary:
        _run(summary_cmd, root)

    final_status = _manifest_status(manifest, expected_rows)
    final_status.update(
        {
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "quant_dir": str(quant_dir),
            "quantification_ran": not args.skip_quantification,
            "summary_ran": not args.skip_summary,
            "outputs": {
                "quant_report": str(quant_dir / "causal_mechanism_quantification_report.json"),
                "verdict_table": str(quant_dir / "mechanism_targeted_verdict_table.csv"),
                "markdown_summary": str(quant_dir / "mechanism_targeted_causal_quantification_summary.md"),
            },
        }
    )
    _write_status(quant_dir, final_status)
    print(json.dumps(final_status, indent=2, ensure_ascii=False), flush=True)
    return final_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize mechanism-targeted causal ablation pipeline.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/mechanism_targeted_causal_ablation_full_20260630"),
    )
    parser.add_argument(
        "--quant-dir",
        type=Path,
        default=Path("results/mechanism_targeted_causal_quantification_full_20260630"),
    )
    parser.add_argument("--expected-candidates", type=int, default=11)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--max-wait-minutes", type=float, default=0.0)
    parser.add_argument("--skip-quantification", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()
    finalize(args)


if __name__ == "__main__":
    main()
