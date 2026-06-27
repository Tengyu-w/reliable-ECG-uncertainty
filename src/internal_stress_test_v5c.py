from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = list(range(42, 52))
SCORE_COLUMNS = [
    "softmax_vtvf_ambiguity",
    "validity_v1_gate_x_boundary",
    "wavelet_vtvf_boundary_risk",
]


def _top_budget_mask(score: np.ndarray, budget: float, candidate: np.ndarray | None = None) -> np.ndarray:
    if candidate is None:
        candidate = np.ones(len(score), dtype=bool)
    eligible = np.flatnonzero(candidate)
    n = min(len(eligible), max(1, int(round(len(score) * budget))))
    mask = np.zeros(len(score), dtype=bool)
    if n <= 0:
        return mask
    ordered = eligible[np.argsort(-score[eligible])]
    mask[ordered[:n]] = True
    return mask


def _robust_scale_from_val(val: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo = float(np.nanquantile(val, 0.05))
    hi = float(np.nanquantile(val, 0.95))
    if hi <= lo:
        return np.zeros(len(x), dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        row["n_rows"] = int(len(sub))
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
            row[f"{col}_min"] = float(sub[col].min())
            row[f"{col}_max"] = float(sub[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _capture_metrics(df: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    y = df["y_true"].to_numpy(int)
    pred = df["y_pred"].to_numpy(int)
    errors = df["is_error"].to_numpy(bool)
    vtvf = df["is_vtvf_cross_error"].to_numpy(bool)
    vt_to_vf = (y == 1) & (pred == 2)
    vf_to_vt = (y == 2) & (pred == 1)
    return {
        "action_rate": float(selected.mean()),
        "all_error_capture": float((errors & selected).sum() / max(errors.sum(), 1)),
        "vtvf_capture": float((vtvf & selected).sum() / max(vtvf.sum(), 1)),
        "vt_to_vf_capture": float((vt_to_vf & selected).sum() / max(vt_to_vf.sum(), 1)),
        "vf_to_vt_capture": float((vf_to_vt & selected).sum() / max(vf_to_vt.sum(), 1)),
        "unresolved_error_rate": float((errors & ~selected).mean()),
        "unresolved_vtvf_rate": float((vtvf & ~selected).mean()),
    }


def _cluster_concentration_rows(seed: int, v5c_dir: Path, budgets: list[float]) -> list[dict[str, Any]]:
    path = v5c_dir / f"seed{seed}" / "v5c_hierarchical_routing_assignments_test.csv"
    usecols = [
        "budget",
        "y_true",
        "y_pred",
        "is_error",
        "is_vtvf_cross_error",
        "latent_cluster",
        "mechanism_action",
        "hierarchical_stage",
    ]
    df = pd.read_csv(path, usecols=usecols)
    rows = []
    for budget in budgets:
        sub = df[np.isclose(df["budget"].to_numpy(float), budget)].copy()
        if sub.empty:
            continue
        selected = ~sub["mechanism_action"].eq("single_label").to_numpy(bool)
        vtvf = sub["is_vtvf_cross_error"].to_numpy(bool)
        captured_vtvf = vtvf & selected
        cluster = sub["latent_cluster"].astype(int).to_numpy()
        total_vtvf = int(vtvf.sum())
        total_captured = int(captured_vtvf.sum())
        cluster_total = pd.Series(cluster[vtvf]).value_counts()
        cluster_captured = pd.Series(cluster[captured_vtvf]).value_counts()
        top_error_share = float(cluster_total.iloc[0] / max(total_vtvf, 1)) if len(cluster_total) else np.nan
        top3_error_share = float(cluster_total.iloc[:3].sum() / max(total_vtvf, 1)) if len(cluster_total) else np.nan
        top_captured_share = float(cluster_captured.iloc[0] / max(total_captured, 1)) if len(cluster_captured) else np.nan
        top3_captured_share = float(cluster_captured.iloc[:3].sum() / max(total_captured, 1)) if len(cluster_captured) else np.nan
        shares = (cluster_captured / max(total_captured, 1)).to_numpy(float) if len(cluster_captured) else np.asarray([])
        hhi = float(np.sum(shares**2)) if shares.size else np.nan
        cluster_rows = []
        for c, count in cluster_total.items():
            mask = vtvf & (cluster == int(c))
            if int(mask.sum()) < 5:
                continue
            cluster_rows.append(float((mask & selected).sum() / mask.sum()))
        top_cluster = int(cluster_total.index[0]) if len(cluster_total) else -1
        without_top = vtvf & (cluster != top_cluster)
        without_top_capture = float((without_top & selected).sum() / max(without_top.sum(), 1))
        metrics = _capture_metrics(sub, selected)
        rows.append(
            {
                "seed": seed,
                "budget": budget,
                "n_samples": int(len(sub)),
                "n_vtvf_errors": total_vtvf,
                "n_captured_vtvf_errors": total_captured,
                "n_vtvf_error_clusters": int(len(cluster_total)),
                "top1_vtvf_error_cluster_share": top_error_share,
                "top3_vtvf_error_cluster_share": top3_error_share,
                "top1_captured_vtvf_cluster_share": top_captured_share,
                "top3_captured_vtvf_cluster_share": top3_captured_share,
                "captured_cluster_hhi": hhi,
                "min_cluster_capture_with_ge5_errors": float(np.min(cluster_rows)) if cluster_rows else np.nan,
                "std_cluster_capture_with_ge5_errors": float(np.std(cluster_rows, ddof=1)) if len(cluster_rows) > 1 else np.nan,
                "capture_without_top_vtvf_error_cluster": without_top_capture,
                **metrics,
            }
        )
    return rows


def _boundary_score_from_subsample(val: pd.DataFrame, test: pd.DataFrame, rng: np.random.Generator, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        sampled = val
    else:
        n = max(10, int(round(len(val) * fraction)))
        idx = rng.choice(len(val), size=n, replace=False)
        sampled = val.iloc[idx]
    scaled = []
    for col in SCORE_COLUMNS:
        scaled.append(_robust_scale_from_val(sampled[col].to_numpy(float), test[col].to_numpy(float)))
    return np.mean(np.stack(scaled, axis=1), axis=1)


def _downsample_rows(
    seed: int,
    boundary_dir: Path,
    budgets: list[float],
    fractions: list[float],
    repeats: int,
) -> list[dict[str, Any]]:
    val_path = boundary_dir / f"seed{seed}" / "evidence_scores_val_boundary_first.csv"
    test_path = boundary_dir / f"seed{seed}" / "evidence_scores_test_boundary_first.csv"
    usecols = [
        "y_true",
        "y_pred",
        "is_error",
        "is_vtvf_cross_error",
        "is_vtvf_candidate",
        *SCORE_COLUMNS,
    ]
    val = pd.read_csv(val_path, usecols=usecols)
    test = pd.read_csv(test_path, usecols=usecols)
    rows = []
    for fraction in fractions:
        n_repeats = 1 if fraction >= 0.999 else repeats
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed * 1000 + rep + int(round(fraction * 100)))
            score = _boundary_score_from_subsample(val, test, rng, fraction)
            candidate = test["is_vtvf_candidate"].to_numpy(bool)
            for budget in budgets:
                selected = _top_budget_mask(score, budget, candidate=candidate)
                rows.append(
                    {
                        "seed": seed,
                        "fraction": fraction,
                        "repeat": rep,
                        "budget": budget,
                        **_capture_metrics(test, selected),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal stress tests for v5c routing evidence.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--v5c-dir", type=Path, default=None)
    parser.add_argument("--boundary-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    v5c_dir = args.v5c_dir or (args.routing_dir / "hierarchical_router_v5c")
    boundary_dir = args.boundary_dir or (args.routing_dir / "boundary_first_router_v5b")
    out_dir = args.out or (args.routing_dir / "internal_stress_v5c")
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows = []
    downsample_rows = []
    manifest = []
    for seed in SEEDS:
        try:
            cluster_rows.extend(_cluster_concentration_rows(seed, v5c_dir, args.budgets))
            downsample_rows.extend(_downsample_rows(seed, boundary_dir, args.budgets, args.fractions, args.repeats))
            manifest.append({"seed": seed, "status": "completed"})
        except Exception as exc:
            manifest.append({"seed": seed, "status": "failed", "error": str(exc)})

    cluster_df = pd.DataFrame(cluster_rows)
    downsample_df = pd.DataFrame(downsample_rows)
    cluster_df.to_csv(out_dir / "cluster_concentration_stress_by_seed.csv", index=False)
    downsample_df.to_csv(out_dir / "validation_downsample_stress_by_repeat.csv", index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "internal_stress_manifest.csv", index=False)

    cluster_summary = _mean_std(
        cluster_df,
        ["budget"],
        [
            "vtvf_capture",
            "all_error_capture",
            "top1_vtvf_error_cluster_share",
            "top3_vtvf_error_cluster_share",
            "top1_captured_vtvf_cluster_share",
            "top3_captured_vtvf_cluster_share",
            "captured_cluster_hhi",
            "min_cluster_capture_with_ge5_errors",
            "capture_without_top_vtvf_error_cluster",
        ],
    )
    cluster_summary.to_csv(out_dir / "cluster_concentration_stress_mean_std.csv", index=False)

    downsample_summary = _mean_std(
        downsample_df,
        ["fraction", "budget"],
        [
            "action_rate",
            "vtvf_capture",
            "all_error_capture",
            "vt_to_vf_capture",
            "vf_to_vt_capture",
            "unresolved_vtvf_rate",
        ],
    )
    downsample_summary.to_csv(out_dir / "validation_downsample_stress_mean_std.csv", index=False)

    report = {
        "routing_dir": str(args.routing_dir),
        "v5c_dir": str(v5c_dir),
        "boundary_dir": str(boundary_dir),
        "out_dir": str(out_dir),
        "budgets": args.budgets,
        "fractions": args.fractions,
        "repeats": args.repeats,
        "manifest": manifest,
    }
    (out_dir / "internal_stress_v5c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Cluster concentration stress")
    print(cluster_summary.to_string(index=False))
    print("\nValidation downsample stress")
    print(downsample_summary.to_string(index=False))


if __name__ == "__main__":
    main()
