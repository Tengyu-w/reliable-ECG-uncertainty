from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = list(range(42, 52))


def _seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def _top_budget_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = max(1, int(round(len(score) * budget)))
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n]] = True
    return mask


def _ranked_rows(
    seed: int,
    source: str,
    evidence: pd.DataFrame,
    gate: pd.DataFrame,
    budgets: list[float],
) -> list[dict[str, float | int | str]]:
    errors = evidence["is_error"].to_numpy(bool)
    vtvf = evidence["is_vtvf_cross_error"].to_numpy(bool)
    scores = {
        "validity_gate": gate["validity_gate"].to_numpy(float),
        "boundary_score": gate["boundary_score"].to_numpy(float),
        "gate_x_boundary": gate["validity_gate"].to_numpy(float) * gate["boundary_score"].to_numpy(float),
        "gate_minus_confidence": gate["validity_gate"].to_numpy(float) + (1.0 - gate["confidence"].to_numpy(float)),
        "low_validity_model_confidence": 1.0 - gate["confidence"].to_numpy(float),
    }
    rows: list[dict[str, float | int | str]] = []
    for score_name, score in scores.items():
        for budget in budgets:
            mask = _top_budget_mask(score, budget)
            auto = ~mask
            rows.append(
                {
                    "seed": seed,
                    "source": source,
                    "score": score_name,
                    "budget": budget,
                    "action_rate": float(mask.mean()),
                    "all_error_addressed": float((errors & mask).sum() / max(errors.sum(), 1)),
                    "vtvf_cross_error_addressed": float((vtvf & mask).sum() / max(vtvf.sum(), 1)),
                    "automatic_unresolved_error_rate": float((errors & auto).mean()) if auto.any() else np.nan,
                    "automatic_unresolved_vtvf_cross_error_rate": float((vtvf & auto).mean()) if auto.any() else np.nan,
                    "review_error_rate": float(errors[mask].mean()) if mask.any() else np.nan,
                    "review_vtvf_cross_error_rate": float(vtvf[mask].mean()) if mask.any() else np.nan,
                }
            )
    return rows


def _discover_gate_files(root: Path, source: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*/validity_gate_scores_test.csv")):
        seed = _seed_from_name(path.parent)
        if seed is not None:
            out[seed] = path
    return out


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()
    out_dir = args.out or (args.routing_dir / "validity_boundary_signal_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "cnn_tcn_validity_v1": _discover_gate_files(Path("results/cnn_tcn_validity_20260626"), "cnn_tcn_validity_v1"),
        "cnn_tcn_validity_v2_seed42": _discover_gate_files(Path("results/cnn_tcn_validity_v2_20260627"), "cnn_tcn_validity_v2_seed42"),
        "wavelet_tcn_boundary_seed42": _discover_gate_files(Path("results/cnn_wavelet_tcn_boundary_20260627"), "wavelet_tcn_boundary_seed42"),
    }

    rows = []
    alignment_rows = []
    for source, files in sources.items():
        for seed, gate_path in sorted(files.items()):
            seed_dir = args.routing_dir / f"seed{seed}"
            evidence_path = seed_dir / "evidence_scores_test.csv"
            if not evidence_path.exists():
                continue
            evidence = pd.read_csv(evidence_path)
            gate = pd.read_csv(gate_path)
            aligned = len(evidence) == len(gate) and np.array_equal(
                evidence["y_true"].to_numpy(int), gate["y_true"].to_numpy(int)
            )
            alignment_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "gate_path": str(gate_path),
                    "n_evidence": len(evidence),
                    "n_gate": len(gate),
                    "y_true_aligned": aligned,
                    "y_pred_same_as_teacher": bool(
                        aligned and np.array_equal(evidence["y_pred"].to_numpy(int), gate["y_pred"].to_numpy(int))
                    ),
                }
            )
            if not aligned:
                continue
            rows.extend(_ranked_rows(seed, source, evidence, gate, args.budgets))

    all_df = pd.DataFrame(rows)
    align_df = pd.DataFrame(alignment_rows)
    all_df.to_csv(out_dir / "all_seed_validity_boundary_signal_policy.csv", index=False)
    align_df.to_csv(out_dir / "validity_boundary_alignment_manifest.csv", index=False)
    metric_cols = [
        "action_rate",
        "all_error_addressed",
        "vtvf_cross_error_addressed",
        "automatic_unresolved_error_rate",
        "automatic_unresolved_vtvf_cross_error_rate",
        "review_error_rate",
        "review_vtvf_cross_error_rate",
    ]
    summary = _mean_std(all_df, ["source", "score", "budget"], metric_cols)
    summary.to_csv(out_dir / "validity_boundary_signal_mean_std.csv", index=False)
    print(summary.sort_values(["source", "budget", "vtvf_cross_error_addressed_mean"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
