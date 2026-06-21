from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.signal import welch
from scipy.stats import kruskal, spearmanr

from .data import CLASS_NAMES, load_rhythm_windows, make_splits


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _seed_from_checkpoint(run_dir: Path, default: int = 42) -> int:
    path = run_dir / "best_model.pt"
    if not path.exists():
        return default
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return int(state.get("args", {}).get("seed", default))
    except Exception:
        return default


def _spectral_features(x: np.ndarray, fs: int) -> dict[str, float]:
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    mask = (freqs >= 0.5) & (freqs <= 40.0)
    freqs, psd = freqs[mask], psd[mask]
    psd = np.maximum(psd, 1e-12)
    p = psd / psd.sum()
    spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(p))
    peak_idx = int(np.argmax(psd))
    dominant_freq = float(freqs[peak_idx])
    dominant_concentration = float(psd[peak_idx] / psd.sum())
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    return {
        "spectral_entropy": float(spectral_entropy),
        "dominant_frequency": dominant_freq,
        "dominant_frequency_concentration": dominant_concentration,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
    }


def _autocorr_features(x: np.ndarray, fs: int) -> dict[str, float]:
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / max(ac[0], 1e-12)
    min_lag = max(1, int(0.12 * fs))
    max_lag = min(len(ac), int(1.5 * fs))
    if max_lag <= min_lag:
        return {"autocorr_peak": 0.0, "autocorr_peak_lag_s": np.nan}
    search = ac[min_lag:max_lag]
    peak_rel = int(np.argmax(search))
    peak = float(search[peak_rel])
    lag = float((min_lag + peak_rel) / fs)
    return {"autocorr_peak": peak, "autocorr_peak_lag_s": lag}


def _sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) > 250:
        x = x[::2]
    r = r_ratio * np.std(x)
    if r <= 0 or len(x) < m + 2:
        return 0.0

    def count_matches(order: int) -> int:
        templates = np.array([x[i : i + order] for i in range(len(x) - order + 1)])
        return len(cKDTree(templates).query_pairs(r, p=np.inf))

    b = count_matches(m)
    a = count_matches(m + 1)
    if a == 0 or b == 0:
        return float(np.log(len(x)))
    return float(-np.log(a / b))


def _features_for_signal(x: np.ndarray, fs: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    zcr = float(np.mean(np.diff(np.signbit(x)) != 0))
    dx = np.diff(x)
    return {
        **_spectral_features(x, fs),
        **_autocorr_features(x, fs),
        "zero_crossing_rate": zcr,
        "line_length": float(np.sum(np.abs(dx)) / len(x)),
        "rms_amplitude": float(np.sqrt(np.mean(x**2))),
        "sample_entropy": _sample_entropy(x),
    }


def _assign_group(df: pd.DataFrame) -> pd.Series:
    y = df["y_true"].to_numpy()
    pred = df["y_pred"].to_numpy()
    vtvf_error = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    groups = np.full(len(df), "other", dtype=object)
    groups[(y == pred) & (y == 1)] = "typical_correct_vt"
    groups[(y == pred) & (y == 2)] = "typical_correct_vf"
    groups[vtvf_error] = "vtvf_boundary_error"

    atypical = df["atypicality_score"].to_numpy() >= df["atypicality_score"].quantile(0.9)
    ambiguous = df["boundary_ambiguity_score"].to_numpy() >= df["boundary_ambiguity_score"].quantile(0.9)
    groups[atypical & ~ambiguous] = "high_atypicality"
    groups[ambiguous & ~atypical] = "high_boundary_ambiguity"
    groups[ambiguous & atypical] = "high_risk_both"
    return pd.Series(groups, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    if not (run_dir / "ambiguity_scores.csv").exists():
        raise FileNotFoundError("Run ambiguity_analysis.py first to create ambiguity_scores.csv")

    ambiguity = pd.read_csv(run_dir / "ambiguity_scores.csv")
    uncertainty = pd.read_csv(run_dir / "uncertainty_scores.csv")
    dataset = load_rhythm_windows(args.mat)
    seed = _seed_from_checkpoint(run_dir) if args.seed is None else args.seed
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=seed)
    x_test = splits.x_test[:, 0, :]

    feature_rows = [_features_for_signal(x, args.sample_rate) for x in x_test]
    features = pd.DataFrame(feature_rows)
    df = pd.concat([ambiguity.reset_index(drop=True), features], axis=1)
    df["atypicality_score"] = _normalise(uncertainty["knn"].to_numpy())
    df["mahalanobis_atypicality"] = _normalise(uncertainty["mahalanobis"].to_numpy())
    df["boundary_ambiguity_score"] = _normalise(df["softmax_vtvf_ambiguity"].to_numpy())
    df["prototype_boundary_ambiguity"] = _normalise(df["prototype_vtvf_ambiguity"].to_numpy())
    df["reliability_group"] = _assign_group(df)
    df.to_csv(run_dir / "regularity_features.csv", index=False)

    feature_cols = [
        "autocorr_peak",
        "spectral_entropy",
        "dominant_frequency_concentration",
        "dominant_frequency",
        "spectral_bandwidth",
        "zero_crossing_rate",
        "line_length",
        "sample_entropy",
    ]
    rows = []
    for col in feature_cols:
        rho_a, p_a = spearmanr(df[col], df["atypicality_score"], nan_policy="omit")
        rho_b, p_b = spearmanr(df[col], df["boundary_ambiguity_score"], nan_policy="omit")
        group_values = [
            g[col].dropna().to_numpy()
            for _, g in df[df["reliability_group"].isin(
                ["typical_correct_vt", "typical_correct_vf", "vtvf_boundary_error", "high_atypicality"]
            )].groupby("reliability_group")
        ]
        h_stat, h_p = (np.nan, np.nan)
        if len(group_values) >= 2 and all(len(v) > 0 for v in group_values):
            h_stat, h_p = kruskal(*group_values)
        rows.append(
            {
                "feature": col,
                "spearman_with_atypicality": float(rho_a),
                "p_atypicality": float(p_a),
                "spearman_with_boundary_ambiguity": float(rho_b),
                "p_boundary_ambiguity": float(p_b),
                "kruskal_group_stat": float(h_stat),
                "kruskal_group_p": float(h_p),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "regularity_summary.csv", index=False)

    group_summary = (
        df.groupby("reliability_group")[feature_cols + ["atypicality_score", "boundary_ambiguity_score"]]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    group_summary.to_csv(run_dir / "regularity_group_summary.csv", index=False)

    plot_groups = ["typical_correct_vt", "typical_correct_vf", "vtvf_boundary_error", "high_atypicality"]
    for col in ["autocorr_peak", "spectral_entropy", "dominant_frequency_concentration", "sample_entropy"]:
        plt.figure(figsize=(7, 4))
        data = [df.loc[df["reliability_group"] == g, col].dropna().to_numpy() for g in plot_groups]
        plt.boxplot(data, tick_labels=plot_groups, showfliers=False)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(run_dir / f"regularity_{col}_by_group.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6, 5))
    colors = df["reliability_group"].map(
        {
            "typical_correct_vt": "tab:orange",
            "typical_correct_vf": "tab:cyan",
            "vtvf_boundary_error": "tab:red",
            "high_atypicality": "tab:purple",
            "high_boundary_ambiguity": "tab:pink",
            "high_risk_both": "black",
            "other": "lightgray",
        }
    )
    plt.scatter(df["spectral_entropy"], df["autocorr_peak"], c=colors, s=12, alpha=0.75)
    plt.xlabel("Spectral entropy")
    plt.ylabel("Autocorrelation peak")
    plt.tight_layout()
    plt.savefig(run_dir / "regularity_entropy_vs_autocorr.png", dpi=180)
    plt.close()

    print(summary)


if __name__ == "__main__":
    main()
