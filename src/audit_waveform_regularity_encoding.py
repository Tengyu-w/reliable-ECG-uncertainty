from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

from .data import (
    REGULARITY_FEATURE_NAMES,
    build_duplicate_family_groups,
    extract_regularity_features_batch,
    load_rhythm_windows,
    make_splits,
)
from .metrics import softmax


OUTCOMES = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]
LOWER_IS_BETTER = {"ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"}


def _groups(dataset, split_grouping: str) -> np.ndarray:
    if split_grouping == "duplicate_family":
        return build_duplicate_family_groups(dataset.x, dataset.record_ids)
    return dataset.record_ids


def _split_data(mat: Path, seed: int, split_grouping: str):
    dataset = load_rhythm_windows(mat)
    return make_splits(dataset.x, dataset.y, groups=_groups(dataset, split_grouping), seed=seed)


def _regularity_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = extract_regularity_features_batch(train_x)
    val_raw = extract_regularity_features_batch(val_x)
    test_raw = extract_regularity_features_batch(test_x)
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-6
    return (
        ((train_raw - mean) / std).astype(np.float32),
        ((val_raw - mean) / std).astype(np.float32),
        ((test_raw - mean) / std).astype(np.float32),
    )


def _load_npz(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["embeddings"].astype(np.float32), data["logits"].astype(np.float32), data["y"].astype(np.int64)


def _load_metrics(run_dir: Path) -> dict[str, float]:
    raw = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    out = {k: float(raw[k]) for k in ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors"] if k in raw}
    out["error_migration_penalty"] = (
        float(raw.get("vt_as_vf", 0.0))
        + float(raw.get("vf_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vf", 0.0))
    )
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return float("nan")
    if kind == "spearman":
        return float(spearmanr(x[mask], y[mask]).statistic)
    return float(pearsonr(x[mask], y[mask]).statistic)


def _ridge_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    model = Ridge(alpha=10.0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _max_abs_dim_corr(rep: np.ndarray, target: np.ndarray) -> float:
    values = []
    for dim in range(rep.shape[1]):
        if np.std(rep[:, dim]) < 1e-8:
            continue
        values.append(abs(_safe_corr(rep[:, dim], target, "pearson")))
    return float(np.nanmax(values)) if values else float("nan")


def _encoding_rows(
    row: pd.Series,
    train_emb: np.ndarray,
    train_logits: np.ndarray,
    test_emb: np.ndarray,
    test_logits: np.ndarray,
    train_reg: np.ndarray,
    test_reg: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reps = {
        "final_embedding": (train_emb, test_emb),
        "classifier_logits": (train_logits, test_logits),
    }
    for rep_name, (x_train, x_test) in reps.items():
        for idx, feature in enumerate(REGULARITY_FEATURE_NAMES):
            rows.append(
                {
                    "candidate": row["candidate"],
                    "seed": int(row["seed"]),
                    "role": row.get("role", ""),
                    "target_mechanism": row.get("target_mechanism", ""),
                    "representation": rep_name,
                    "regularity_feature": feature,
                    "ridge_r2_from_representation": _ridge_r2(x_train, train_reg[:, idx], x_test, test_reg[:, idx]),
                    "max_abs_dim_correlation": _max_abs_dim_corr(x_test, test_reg[:, idx]),
                    "variable_role": "non_intervenable_input_attribute",
                    "interpretation": "encoding/utilization audit; this does not imply the ECG waveform itself changed",
                }
            )
    return rows


def _sensitivity_rows(
    row: pd.Series,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    test_reg: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred = softmax(test_logits).argmax(axis=1)
    is_error = pred != test_y
    is_vtvf_cross = ((test_y == 1) & (pred == 2)) | ((test_y == 2) & (pred == 1))
    is_vtvf = np.isin(test_y, [1, 2])
    for idx, feature in enumerate(REGULARITY_FEATURE_NAMES):
        values = test_reg[:, idx]
        lo, hi = np.nanquantile(values, [0.25, 0.75])
        strata = {
            "low_q25": values <= lo,
            "high_q75": values >= hi,
        }
        item: dict[str, Any] = {
            "candidate": row["candidate"],
            "seed": int(row["seed"]),
            "role": row.get("role", ""),
            "target_mechanism": row.get("target_mechanism", ""),
            "regularity_feature": feature,
            "low_threshold_z": float(lo),
            "high_threshold_z": float(hi),
            "variable_role": "non_intervenable_input_attribute",
        }
        for name, mask in strata.items():
            vmask = mask & is_vtvf
            item[f"{name}_n"] = int(mask.sum())
            item[f"{name}_error_rate"] = float(is_error[mask].mean()) if mask.any() else float("nan")
            item[f"{name}_vtvf_n"] = int(vmask.sum())
            item[f"{name}_vtvf_cross_rate"] = float(is_vtvf_cross[vmask].mean()) if vmask.any() else float("nan")
        item["high_minus_low_error_rate"] = item["high_q75_error_rate"] - item["low_q25_error_rate"]
        item["high_minus_low_vtvf_cross_rate"] = item["high_q75_vtvf_cross_rate"] - item["low_q25_vtvf_cross_rate"]
        item["interpretation"] = "error sensitivity stratified by waveform attribute; the attribute itself is not changed by training"
        rows.append(item)
    return rows


def _paired_deltas(run_level: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    baseline = run_level[run_level["candidate"].eq("baseline")].copy()
    index_cols = ["seed", *group_cols]
    baseline = baseline.set_index(index_cols)
    rows: list[dict[str, Any]] = []
    for _, row in run_level[~run_level["candidate"].eq("baseline")].iterrows():
        key_values = [row["seed"], *[row[col] for col in group_cols]]
        key = tuple(key_values) if len(key_values) > 1 else key_values[0]
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        item: dict[str, Any] = {
            "candidate": row["candidate"],
            "seed": int(row["seed"]),
            "role": row.get("role", ""),
            "target_mechanism": row.get("target_mechanism", ""),
        }
        for col in group_cols:
            item[col] = row[col]
        for col in value_cols:
            item[f"{col}_baseline"] = float(base[col])
            item[f"{col}_candidate"] = float(row[col])
            item[f"{col}_delta"] = float(row[col] - base[col])
        rows.append(item)
    return pd.DataFrame(rows)


def _summarize_deltas(deltas: pd.DataFrame, group_cols: list[str], delta_cols: list[str]) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, sub in deltas.groupby(["candidate", *group_cols], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {"candidate": keys[0]}
        for col, value in zip(group_cols, keys[1:]):
            item[col] = value
        item["n_paired_seeds"] = int(sub["seed"].nunique())
        item["role"] = sub["role"].iloc[0] if "role" in sub.columns else ""
        item["target_mechanism"] = sub["target_mechanism"].iloc[0] if "target_mechanism" in sub.columns else ""
        for col in delta_cols:
            values = pd.to_numeric(sub[f"{col}_delta"], errors="coerce")
            item[f"{col}_delta_mean"] = float(values.mean())
            item[f"{col}_delta_std"] = float(values.std()) if values.notna().sum() > 1 else float("nan")
            item[f"{col}_positive_n"] = int((values > 0).sum())
            item[f"{col}_negative_n"] = int((values < 0).sum())
        rows.append(item)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["status"].astype(str).eq("completed")].copy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    encoding_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        run_dir = Path(str(row["run_dir"]))
        seed = int(row["seed"])
        split_grouping = "record"
        summary_path = run_dir / "split_summary.json"
        if summary_path.exists():
            split_grouping = json.loads(summary_path.read_text(encoding="utf-8")).get("split_grouping", "record")
        key = (seed, split_grouping)
        if key not in split_cache:
            splits = _split_data(args.mat, seed, split_grouping)
            train_reg, val_reg, test_reg = _regularity_features(splits.x_train, splits.x_val, splits.x_test)
            split_cache[key] = (splits.y_train, splits.y_val, splits.y_test, train_reg, val_reg, test_reg)
        train_y_split, val_y_split, test_y_split, train_reg, val_reg, test_reg = split_cache[key]

        val_emb, val_logits, val_y = _load_npz(run_dir, "val")
        test_emb, test_logits, test_y = _load_npz(run_dir, "test")
        # Train embeddings are saved from a shuffled loader in src.train, so use val->test for aligned decoding.
        if not np.array_equal(val_y, val_y_split) or not np.array_equal(test_y, test_y_split):
            raise RuntimeError(f"Split labels do not match run outputs: {run_dir}")

        encoding_rows.extend(_encoding_rows(row, val_emb, val_logits, test_emb, test_logits, val_reg, test_reg))
        sensitivity_rows.extend(_sensitivity_rows(row, test_logits, test_y, test_reg))
        metrics = _load_metrics(run_dir)
        outcome_rows.append({**row.to_dict(), **metrics})

    encoding = pd.DataFrame(encoding_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    outcomes = pd.DataFrame(outcome_rows)

    encoding.to_csv(out / "waveform_regularity_encoding_run_level.csv", index=False)
    sensitivity.to_csv(out / "waveform_regularity_error_sensitivity_run_level.csv", index=False)
    outcomes.to_csv(out / "waveform_regularity_outcome_run_level.csv", index=False)

    encoding_deltas = _paired_deltas(
        encoding,
        ["ridge_r2_from_representation", "max_abs_dim_correlation"],
        ["representation", "regularity_feature"],
    )
    sensitivity_deltas = _paired_deltas(
        sensitivity,
        [
            "high_minus_low_error_rate",
            "high_minus_low_vtvf_cross_rate",
            "high_q75_error_rate",
            "high_q75_vtvf_cross_rate",
            "low_q25_error_rate",
            "low_q25_vtvf_cross_rate",
        ],
        ["regularity_feature"],
    )
    outcome_deltas = _paired_deltas(outcomes, OUTCOMES, [])

    encoding_deltas.to_csv(out / "waveform_regularity_encoding_paired_deltas.csv", index=False)
    sensitivity_deltas.to_csv(out / "waveform_regularity_error_sensitivity_paired_deltas.csv", index=False)
    outcome_deltas.to_csv(out / "waveform_regularity_outcome_paired_deltas.csv", index=False)

    encoding_summary = _summarize_deltas(
        encoding_deltas,
        ["representation", "regularity_feature"],
        ["ridge_r2_from_representation", "max_abs_dim_correlation"],
    )
    sensitivity_summary = _summarize_deltas(
        sensitivity_deltas,
        ["regularity_feature"],
        [
            "high_minus_low_error_rate",
            "high_minus_low_vtvf_cross_rate",
            "high_q75_error_rate",
            "high_q75_vtvf_cross_rate",
        ],
    )
    outcome_summary = _summarize_deltas(outcome_deltas, [], OUTCOMES)

    encoding_summary.to_csv(out / "waveform_regularity_encoding_delta_summary.csv", index=False)
    sensitivity_summary.to_csv(out / "waveform_regularity_error_sensitivity_delta_summary.csv", index=False)
    outcome_summary.to_csv(out / "waveform_regularity_outcome_delta_summary.csv", index=False)

    regularity_focus = {
        "candidate": "regularity_aux_medium",
        "encoding_summary_rows": int((encoding_summary["candidate"] == "regularity_aux_medium").sum())
        if not encoding_summary.empty
        else 0,
        "sensitivity_summary_rows": int((sensitivity_summary["candidate"] == "regularity_aux_medium").sum())
        if not sensitivity_summary.empty
        else 0,
    }
    report = {
        "manifest": str(args.manifest),
        "out": str(out),
        "n_completed_runs": int(len(manifest)),
        "n_seeds": int(manifest["seed"].nunique()),
        "n_candidates": int(manifest["candidate"].nunique()),
        "regularity_features": REGULARITY_FEATURE_NAMES,
        "variable_interpretation": {
            "regularity_features": "non-intervenable ECG input attributes",
            "encoding_metrics": "model representation/logit ability to encode waveform attributes",
            "error_sensitivity_metrics": "model error behavior stratified by waveform attributes",
        },
        "regularity_aux_focus": regularity_focus,
        "limitations": [
            "This audit does not claim that training changes raw ECG waveform features.",
            "Encoding and sensitivity are internal paired evidence, not external validation.",
            "Regularity attributes are used as stratification/explanation variables rather than manipulable mediators.",
        ],
        "outputs": {
            "encoding_run_level": str(out / "waveform_regularity_encoding_run_level.csv"),
            "sensitivity_run_level": str(out / "waveform_regularity_error_sensitivity_run_level.csv"),
            "encoding_summary": str(out / "waveform_regularity_encoding_delta_summary.csv"),
            "sensitivity_summary": str(out / "waveform_regularity_error_sensitivity_delta_summary.csv"),
            "outcome_summary": str(out / "waveform_regularity_outcome_delta_summary.csv"),
        },
    }
    (out / "waveform_regularity_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether model representations encode ECG waveform regularity attributes.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/mechanism_targeted_causal_ablation_full_20260630/"
            "mechanism_targeted_ablation_manifest_20260630_212510.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/waveform_regularity_encoding_audit_20260701"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
