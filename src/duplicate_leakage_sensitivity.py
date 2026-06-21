from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from .data import load_rhythm_windows


def _split_indices(y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(outer.split(indices, y, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_rel, val_rel = next(
        inner.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx])
    )
    return {
        "train": trainval_idx[train_rel],
        "val": trainval_idx[val_rel],
        "test": test_idx,
    }


def _hash_rows(x: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in x.reshape(len(x), -1)
        ]
    )


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, keep: np.ndarray
) -> dict[str, float | int]:
    y = y_true[keep]
    pred = y_pred[keep]
    vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    return {
        "n_evaluated": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "total_errors": int((y != pred).sum()),
        "vtvf_cross_errors": int(vtvf.sum()),
    }


def _review_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_score: np.ndarray,
    keep: np.ndarray,
    burden: float,
) -> dict[str, float | int]:
    y = y_true[keep]
    pred = y_pred[keep]
    score = risk_score[keep]
    any_error = y != pred
    vtvf = ((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1))
    order = np.argsort(-score)
    n_review = max(1, int(round(len(y) * burden)))
    review = order[:n_review]
    auto = order[n_review:]
    return {
        "reviewed": n_review,
        "all_error_captured": float(any_error[review].sum() / max(any_error.sum(), 1)),
        "vtvf_error_captured": float(vtvf[review].sum() / max(vtvf.sum(), 1)),
        "auto_error_rate": float(any_error[auto].mean()) if len(auto) else np.nan,
        "auto_vtvf_error_rate": float(vtvf[auto].mean()) if len(auto) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether exact cross-split duplicate windows change PRO or RISK conclusions."
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--mitigation-summary", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/duplicate_leakage_sensitivity_20260620")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset = load_rhythm_windows(args.mat)
    hashes = _hash_rows(dataset.x)
    split_cache = {
        seed: _split_indices(dataset.y, dataset.record_ids, seed) for seed in [42, 43, 44]
    }
    clean_masks: dict[int, np.ndarray] = {}
    duplicate_rows = []
    for seed, splits in split_cache.items():
        reference_hashes = set(hashes[np.concatenate([splits["train"], splits["val"]])])
        test_hashes = hashes[splits["test"]]
        leaked = np.asarray([value in reference_hashes for value in test_hashes])
        clean_masks[seed] = ~leaked
        for test_position in np.flatnonzero(leaked):
            global_idx = int(splits["test"][test_position])
            duplicate_rows.append(
                {
                    "seed": seed,
                    "test_position": int(test_position),
                    "global_index": global_idx,
                    "window_id": str(dataset.window_ids[global_idx]),
                    "record_id": str(dataset.record_ids[global_idx]),
                    "class": int(dataset.y[global_idx]),
                    "hash": str(test_hashes[test_position]),
                }
            )
    pd.DataFrame(duplicate_rows).to_csv(
        args.out / "cross_split_duplicate_test_windows.csv", index=False
    )

    mitigation = pd.read_csv(args.mitigation_summary)
    classification_rows = []
    for _, row in mitigation[
        mitigation["variant"].isin(["baseline", "prototype_separation"])
    ].iterrows():
        seed = int(row["seed"])
        pred = pd.read_csv(Path(str(row["run_dir"])) / "test_predictions.csv")
        full = np.ones(len(pred), dtype=bool)
        clean = clean_masks[seed]
        if len(clean) != len(pred):
            raise ValueError(f"Split length mismatch for mitigation seed {seed}")
        for evaluation_set, keep in [
            ("all_test_windows", full),
            ("exclude_cross_split_exact_duplicates", clean),
        ]:
            classification_rows.append(
                {
                    "seed": seed,
                    "variant": str(row["variant"]),
                    "evaluation_set": evaluation_set,
                    "excluded_windows": int((~keep).sum()),
                    **_classification_metrics(
                        pred["y_true"].to_numpy(int),
                        pred["y_pred"].to_numpy(int),
                        keep,
                    ),
                }
            )
    class_df = pd.DataFrame(classification_rows)
    class_df.to_csv(args.out / "pro_duplicate_sensitivity_seed_level.csv", index=False)

    manifest = pd.read_csv(args.core_manifest)
    teachers = manifest[manifest["stage"].eq("regularity_feature_injection")].set_index("seed")
    risks = manifest[manifest["stage"].eq("risk_aligned_distillation")].set_index("seed")
    review_rows = []
    for seed in sorted(set(teachers.index) & set(risks.index)):
        teacher_dir = Path(str(teachers.loc[seed, "run_dir"]))
        risk_dir = Path(str(risks.loc[seed, "run_dir"]))
        pred = pd.read_csv(teacher_dir / "test_predictions.csv")
        risk = pd.read_csv(risk_dir / "risk_scores_test.csv")
        full = np.ones(len(pred), dtype=bool)
        clean = clean_masks[int(seed)]
        for evaluation_set, keep in [
            ("all_test_windows", full),
            ("exclude_cross_split_exact_duplicates", clean),
        ]:
            for burden in [0.10, 0.20, 0.30]:
                review_rows.append(
                    {
                        "seed": int(seed),
                        "evaluation_set": evaluation_set,
                        "review_burden": burden,
                        "excluded_windows": int((~keep).sum()),
                        **_review_metrics(
                            pred["y_true"].to_numpy(int),
                            pred["y_pred"].to_numpy(int),
                            risk["risk_score"].to_numpy(float),
                            keep,
                            burden,
                        ),
                    }
                )
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(args.out / "risk_duplicate_sensitivity_seed_level.csv", index=False)
    review_summary = (
        review_df.groupby(["evaluation_set", "review_burden"])
        .agg(
            n_seeds=("seed", "nunique"),
            excluded_windows_mean=("excluded_windows", "mean"),
            all_error_captured_mean=("all_error_captured", "mean"),
            all_error_captured_std=("all_error_captured", "std"),
            vtvf_error_captured_mean=("vtvf_error_captured", "mean"),
            vtvf_error_captured_std=("vtvf_error_captured", "std"),
            auto_error_rate_mean=("auto_error_rate", "mean"),
            auto_vtvf_error_rate_mean=("auto_vtvf_error_rate", "mean"),
        )
        .reset_index()
    )
    review_summary.to_csv(args.out / "risk_duplicate_sensitivity_mean_std.csv", index=False)

    plt.figure(figsize=(7.5, 4.8))
    for evaluation_set, sub in review_summary.groupby("evaluation_set"):
        plt.errorbar(
            sub["review_burden"],
            sub["vtvf_error_captured_mean"],
            yerr=sub["vtvf_error_captured_std"],
            marker="o",
            capsize=3,
            label=evaluation_set,
        )
    plt.xlabel("Review burden")
    plt.ylabel("RISK VT/VF cross-error capture")
    plt.ylim(0, 1.05)
    plt.grid(True, color="#dddddd", linewidth=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out / "risk_duplicate_sensitivity.png", dpi=200)
    plt.close()

    notes = [
        "# Exact-duplicate leakage sensitivity",
        "",
        "- Exact duplicate windows found in train/validation are excluded from each seed's test evaluation.",
        "- Models are not retrained; this is a conservative post-hoc sensitivity check of reported conclusions.",
        "- A future definitive rerun should group source records connected by exact duplicate hashes into one duplicate family before splitting.",
        "- Patient-level independence remains unverified because patient identifiers are unavailable.",
    ]
    (args.out / "sensitivity_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
