from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import REGULARITY_FEATURE_NAMES, extract_regularity_features_batch, load_rhythm_windows, make_splits
from .metrics import classification_metrics, expected_calibration_error
from .uncertainty import entropy


FEATURE_GROUPS = {
    "frequency": [
        "spectral_entropy",
        "dominant_frequency",
        "dominant_frequency_concentration",
        "spectral_centroid",
        "spectral_bandwidth",
    ],
    "periodicity": ["autocorr_peak", "autocorr_peak_lag_s"],
    "complexity": ["zero_crossing_rate", "line_length"],
}


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _boundary_score(probs: np.ndarray) -> np.ndarray:
    ventricular_mass = probs[:, 1] + probs[:, 2]
    balance = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_mass, 1e-8)
    return ventricular_mass * balance


def _indices(names: list[str]) -> list[int]:
    return [REGULARITY_FEATURE_NAMES.index(name) for name in names]


def _feature_sets() -> dict[str, list[int]]:
    all_names = list(REGULARITY_FEATURE_NAMES)
    sets = {"all": _indices(all_names)}
    for group, names in FEATURE_GROUPS.items():
        sets[f"{group}_only"] = _indices(names)
        keep = [name for name in all_names if name not in names]
        sets[f"without_{group}"] = _indices(keep)
    return sets


def _models(seed: int) -> dict[str, object]:
    return {
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/20260601_feature_only_regularity"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_rhythm_windows(args.mat)
    splits = make_splits(dataset.x, dataset.y, groups=dataset.record_ids, seed=args.seed)
    x_train = extract_regularity_features_batch(splits.x_train)
    x_test = extract_regularity_features_batch(splits.x_test)

    rows = []
    for set_name, idx in _feature_sets().items():
        for model_name, model in _models(args.seed).items():
            model.fit(x_train[:, idx], splits.y_train)
            probs = model.predict_proba(x_test[:, idx])
            pred = probs.argmax(axis=1)
            metrics = classification_metrics(splits.y_test, probs)
            error = (pred != splits.y_test).astype(int)
            boundary_error = (((splits.y_test == 1) & (pred == 2)) | ((splits.y_test == 2) & (pred == 1))).astype(int)
            boundary = _boundary_score(probs)
            rows.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "n_features": len(idx),
                    "features": ",".join(np.asarray(REGULARITY_FEATURE_NAMES)[idx].tolist()),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ece": expected_calibration_error(splits.y_test, probs),
                    "error_auroc_entropy": _safe_auroc(error, entropy(probs)),
                    "vtvf_boundary_auroc": _safe_auroc(boundary_error, boundary),
                    "vtvf_boundary_aupr": _safe_aupr(boundary_error, boundary),
                    "vt_as_vf": int(((splits.y_test == 1) & (pred == 2)).sum()),
                    "vf_as_vt": int(((splits.y_test == 2) & (pred == 1)).sum()),
                }
            )

    out = pd.DataFrame(rows).sort_values(["model", "macro_f1"], ascending=[True, False])
    out.to_csv(args.out_dir / "regularity_feature_group_ablation.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
