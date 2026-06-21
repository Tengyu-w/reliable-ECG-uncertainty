from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import softmax


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def _lac_sets(probs: np.ndarray, qhat: float) -> list[list[int]]:
    return [np.where(1.0 - row <= qhat)[0].tolist() for row in probs]


def _aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    for row, label in zip(probs, y):
        order = np.argsort(row)[::-1]
        cumsum = np.cumsum(row[order])
        scores.append(cumsum[np.where(order == label)[0][0]])
    return np.asarray(scores)


def _aps_sets(probs: np.ndarray, qhat: float) -> list[list[int]]:
    sets = []
    for row in probs:
        order = np.argsort(row)[::-1]
        cumsum = np.cumsum(row[order])
        keep = order[cumsum <= qhat].tolist()
        if len(keep) < len(order):
            keep.append(int(order[len(keep)]))
        sets.append(sorted(set(keep)))
    return sets


def _evaluate_sets(sets: list[list[int]], y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    contains = np.asarray([int(label in s) for label, s in zip(y, sets)])
    sizes = np.asarray([len(s) for s in sets])
    singleton = sizes == 1
    vtvf_set = np.asarray([set(s) == {1, 2} for s in sets])
    ventricular_contains = np.asarray([set(s).issubset({1, 2}) and len(s) > 0 for s in sets])
    return {
        "coverage": float(contains.mean()),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": float((pred[singleton] == y[singleton]).mean()) if singleton.any() else np.nan,
        "vtvf_pair_rate": float(vtvf_set.mean()),
        "ventricular_only_set_rate": float(ventricular_contains.mean()),
        "vtvf_true_pair_rate": float(vtvf_set[np.isin(y, [1, 2])].mean()),
        "sr_coverage": float(contains[y == 0].mean()),
        "vt_coverage": float(contains[y == 1].mean()),
        "vf_coverage": float(contains[y == 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    val = np.load(run_dir / "embeddings_val.npz")
    test = np.load(run_dir / "embeddings_test.npz")
    val_probs, val_y = softmax(val["logits"]), val["y"]
    test_probs, test_y = softmax(test["logits"]), test["y"]
    pred = test_probs.argmax(axis=1)

    rows = []
    set_rows = []
    for alpha in args.alphas:
        lac_scores = 1.0 - val_probs[np.arange(len(val_y)), val_y]
        lac_q = _quantile(lac_scores, alpha)
        lac_sets = _lac_sets(test_probs, lac_q)
        rows.append({"method": "lac", "alpha": alpha, "qhat": lac_q, **_evaluate_sets(lac_sets, test_y, pred)})

        aps_scores = _aps_scores(val_probs, val_y)
        aps_q = _quantile(aps_scores, alpha)
        aps_sets = _aps_sets(test_probs, aps_q)
        rows.append({"method": "aps", "alpha": alpha, "qhat": aps_q, **_evaluate_sets(aps_sets, test_y, pred)})

        for method, sets in [("lac", lac_sets), ("aps", aps_sets)]:
            for i, s in enumerate(sets):
                set_rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "index": i,
                        "y_true": int(test_y[i]),
                        "y_pred": int(pred[i]),
                        "set": "{" + ",".join(map(str, s)) + "}",
                        "set_size": len(s),
                        "contains_true": int(test_y[i] in s),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "conformal_summary.csv", index=False)
    pd.DataFrame(set_rows).to_csv(run_dir / "conformal_sets.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
