from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data import build_duplicate_family_groups, load_rhythm_windows


def _indices(y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    index = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval, test = next(outer.split(index, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_rel, val_rel = next(inner.split(trainval, y[trainval], groups[trainval]))
    return {"train": trainval[train_rel], "val": trainval[val_rel], "test": test}


def _hashes(x: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in x.reshape(len(x), -1)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--out", type=Path, default=Path("results/duplicate_family_split_audit_20260620")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset = load_rhythm_windows(args.mat)
    families = build_duplicate_family_groups(dataset.x, dataset.record_ids)
    hashes = _hashes(dataset.x)
    rows = []
    for seed in args.seeds:
        splits = _indices(dataset.y, families, seed)
        hash_sets = {name: set(hashes[idx]) for name, idx in splits.items()}
        family_sets = {name: set(families[idx]) for name, idx in splits.items()}
        record_sets = {name: set(dataset.record_ids[idx]) for name, idx in splits.items()}
        for name, idx in splits.items():
            rows.append(
                {
                    "seed": seed,
                    "split": name,
                    "n_windows": len(idx),
                    "n_records": len(record_sets[name]),
                    "n_duplicate_families": len(family_sets[name]),
                    "SR_windows": int((dataset.y[idx] == 0).sum()),
                    "VT_windows": int((dataset.y[idx] == 1).sum()),
                    "VF_windows": int((dataset.y[idx] == 2).sum()),
                }
            )
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            rows.append(
                {
                    "seed": seed,
                    "split": f"{left}_vs_{right}_audit",
                    "n_windows": np.nan,
                    "n_records": len(record_sets[left] & record_sets[right]),
                    "n_duplicate_families": len(family_sets[left] & family_sets[right]),
                    "SR_windows": len(hash_sets[left] & hash_sets[right]),
                    "VT_windows": np.nan,
                    "VF_windows": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "duplicate_family_split_summary.csv", index=False)
    notes = [
        "# Duplicate-family split audit",
        "",
        f"- Source records: {pd.Series(dataset.record_ids).nunique()}.",
        f"- Exact-duplicate-connected families: {pd.Series(families).nunique()}.",
        "- In audit rows, `n_records` is record overlap, `n_duplicate_families` is family overlap, and `SR_windows` stores exact hash overlap.",
        "- All three overlap values should be zero for every pairwise split audit row.",
    ]
    (args.out / "audit_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
