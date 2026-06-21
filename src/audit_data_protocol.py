from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data import CLASS_NAMES, load_rhythm_windows


def _split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(indices, y, groups=groups))
    val_fraction = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rel, val_rel = next(
        inner.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx])
    )
    return {
        "train": trainval_idx[train_rel],
        "val": trainval_idx[val_rel],
        "test": test_idx,
    }


def _hash_rows(x: np.ndarray, decimals: int | None = None) -> np.ndarray:
    rows = x.reshape(len(x), -1)
    if decimals is not None:
        rows = np.round(rows, decimals=decimals).astype(np.float32)
    return np.asarray(
        [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows]
    )


def _duplicate_summary(
    hashes: np.ndarray,
    split_labels: np.ndarray,
    record_ids: np.ndarray,
    kind: str,
) -> tuple[dict[str, int | str], pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "hash": hashes,
            "split": split_labels,
            "record_id": record_ids,
        }
    )
    grouped = frame.groupby("hash", sort=False).agg(
        occurrences=("hash", "size"),
        split_count=("split", "nunique"),
        record_count=("record_id", "nunique"),
        splits=("split", lambda s: ",".join(sorted(set(s)))),
    )
    duplicates = grouped[grouped["occurrences"] > 1].reset_index()
    cross_split = duplicates[duplicates["split_count"] > 1]
    summary = {
        "duplicate_kind": kind,
        "duplicate_hash_groups": int(len(duplicates)),
        "windows_in_duplicate_groups": int(duplicates["occurrences"].sum()),
        "cross_split_duplicate_hash_groups": int(len(cross_split)),
        "windows_in_cross_split_duplicate_groups": int(cross_split["occurrences"].sum()),
        "cross_record_duplicate_hash_groups": int((duplicates["record_count"] > 1).sum()),
    }
    return summary, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit grouped ECG splits, record contribution, and duplicate-window leakage."
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", type=Path, default=Path("results/data_protocol_audit_20260620"))
    parser.add_argument("--quantized-decimals", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_rhythm_windows(args.mat)
    dataset_summary = {
        "mat_path": str(args.mat),
        "n_windows": int(len(dataset.y)),
        "n_records": int(pd.Series(dataset.record_ids).nunique()),
        "window_length_samples": int(dataset.x.shape[-1]),
        "class_window_counts": {
            CLASS_NAMES[label]: int((dataset.y == label).sum()) for label in range(len(CLASS_NAMES))
        },
        "class_record_counts": {
            CLASS_NAMES[label]: int(pd.Series(dataset.record_ids[dataset.y == label]).nunique())
            for label in range(len(CLASS_NAMES))
        },
        "patient_identifier_available": False,
        "split_unit": "source rhythm record",
    }

    split_rows: list[dict[str, int | float | str]] = []
    contribution_rows: list[dict[str, int | float | str]] = []
    membership_rows: list[dict[str, int | str]] = []
    leakage_rows: list[dict[str, int | str]] = []
    seed_split_labels: dict[int, np.ndarray] = {}

    for seed in args.seeds:
        splits = _split_indices(dataset.y, dataset.record_ids, seed)
        labels = np.full(len(dataset.y), "unassigned", dtype=object)
        for split, idx in splits.items():
            labels[idx] = split
        seed_split_labels[seed] = labels

        split_records = {
            split: set(dataset.record_ids[idx].tolist()) for split, idx in splits.items()
        }
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            leakage_rows.append(
                {
                    "seed": seed,
                    "comparison": f"{left}_vs_{right}",
                    "overlapping_record_count": len(split_records[left] & split_records[right]),
                }
            )

        for split, idx in splits.items():
            split_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "n_windows": len(idx),
                    "n_records": len(split_records[split]),
                    "window_fraction": len(idx) / len(dataset.y),
                    **{
                        f"{CLASS_NAMES[label]}_windows": int((dataset.y[idx] == label).sum())
                        for label in range(len(CLASS_NAMES))
                    },
                    **{
                        f"{CLASS_NAMES[label]}_records": int(
                            pd.Series(dataset.record_ids[idx][dataset.y[idx] == label]).nunique()
                        )
                        for label in range(len(CLASS_NAMES))
                    },
                }
            )
            counts = pd.Series(dataset.record_ids[idx]).value_counts()
            for record_id, count in counts.items():
                label = int(dataset.y[np.flatnonzero(dataset.record_ids == record_id)[0]])
                contribution_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "record_id": record_id,
                        "class": CLASS_NAMES[label],
                        "n_windows": int(count),
                        "share_of_split": float(count / len(idx)),
                    }
                )
            for i in idx:
                membership_rows.append(
                    {
                        "seed": seed,
                        "index": int(i),
                        "window_id": str(dataset.window_ids[i]),
                        "record_id": str(dataset.record_ids[i]),
                        "class": CLASS_NAMES[int(dataset.y[i])],
                        "split": split,
                    }
                )

    split_df = pd.DataFrame(split_rows)
    contribution_df = pd.DataFrame(contribution_rows)
    membership_df = pd.DataFrame(membership_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    split_df.to_csv(args.out / "split_class_record_summary.csv", index=False)
    contribution_df.to_csv(args.out / "record_window_contribution.csv", index=False)
    membership_df.to_csv(args.out / "split_membership.csv", index=False)
    leakage_df.to_csv(args.out / "record_overlap_audit.csv", index=False)

    # Duplicate leakage is split-dependent. Audit each seed with both exact
    # hashes and a conservative quantized-hash proxy for nearly identical rows.
    exact_hashes = _hash_rows(dataset.x)
    quantized_hashes = _hash_rows(dataset.x, decimals=args.quantized_decimals)
    duplicate_summaries = []
    for seed, labels in seed_split_labels.items():
        for kind, hashes in [
            ("exact_float32", exact_hashes),
            (f"rounded_{args.quantized_decimals}_decimals_proxy", quantized_hashes),
        ]:
            summary, details = _duplicate_summary(
                hashes, labels, dataset.record_ids, kind=kind
            )
            summary["seed"] = seed
            duplicate_summaries.append(summary)
            details.insert(0, "seed", seed)
            details.to_csv(args.out / f"duplicate_groups_seed{seed}_{kind}.csv", index=False)
    duplicate_df = pd.DataFrame(duplicate_summaries)
    duplicate_df.to_csv(args.out / "duplicate_leakage_summary.csv", index=False)

    contribution_summary = (
        contribution_df.groupby(["seed", "split"])["share_of_split"]
        .agg(["max", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "max": "max_single_record_share",
                "mean": "mean_record_share",
                "median": "median_record_share",
            }
        )
    )
    contribution_summary.to_csv(args.out / "record_contribution_summary.csv", index=False)

    (args.out / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    max_overlap = int(leakage_df["overlapping_record_count"].max())
    max_cross_exact = int(
        duplicate_df.loc[
            duplicate_df["duplicate_kind"] == "exact_float32",
            "cross_split_duplicate_hash_groups",
        ].max()
    )
    report = [
        "# ECG data and split protocol audit",
        "",
        f"- Windows: {dataset_summary['n_windows']}; source records: {dataset_summary['n_records']}.",
        "- Split unit: source rhythm record, reproduced with GroupShuffleSplit for seeds 42/43/44.",
        f"- Maximum record overlap across train/validation/test: {max_overlap}.",
        f"- Maximum exact cross-split duplicate hash groups: {max_cross_exact}.",
        f"- Quantized duplicate proxy: windows rounded to {args.quantized_decimals} decimals before hashing.",
        "- Limitation: patient identifiers are unavailable, so this establishes record-level rather than patient-level independence.",
        "- Interpretation: windows from one source record never cross splits; however, record-level grouping cannot rule out multiple records from the same unknown patient.",
        "",
        "This audit is a research-data integrity check, not clinical validation.",
    ]
    (args.out / "audit_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
