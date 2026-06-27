from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .metrics import softmax
from .uncertainty import entropy, knn_distance, msp, prototype_distance


RISK_COMPONENTS = [
    "entropy",
    "msp",
    "knn",
    "prototype",
    "local_instability",
    "vtvf_mixing",
    "softmax_vtvf_ambiguity",
]

BOUNDED_COMPONENTS = {
    "entropy",
    "msp",
    "local_instability",
    "vtvf_mixing",
    "softmax_vtvf_ambiguity",
}


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(run_dir / f"embeddings_{split}.npz")
    return data["logits"], data["embeddings"], data["y"]


def _scale_with_train(x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(train_x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _scale_component(name: str, x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    if name in BOUNDED_COMPONENTS:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return _scale_with_train(x, train_x)


def _neighbor_scores(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    emb: np.ndarray,
    query_pred: np.ndarray,
    k: int,
    is_train: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_neighbors = k + 1 if is_train else k
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_emb)
    _, idx = nn.kneighbors(emb)
    if is_train:
        idx = idx[:, 1:]
    labels = train_y[idx]
    predicted_support = (labels == query_pred[:, None]).mean(axis=1)

    vtvf_mixing = np.zeros(len(query_pred), dtype=np.float32)
    vt_mask = query_pred == 1
    vf_mask = query_pred == 2
    vtvf_mixing[vt_mask] = (labels[vt_mask] == 2).mean(axis=1)
    vtvf_mixing[vf_mask] = (labels[vf_mask] == 1).mean(axis=1)
    return (1.0 - predicted_support).astype(np.float32), vtvf_mixing.astype(np.float32)


def _softmax_boundary(probs: np.ndarray) -> np.ndarray:
    ventricular_prob = probs[:, 1] + probs[:, 2]
    balance = 1.0 - np.abs(probs[:, 1] - probs[:, 2]) / np.maximum(ventricular_prob, 1e-8)
    return (ventricular_prob * balance).astype(np.float32)


def _components(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    logits: np.ndarray,
    emb: np.ndarray,
    y: np.ndarray,
    k: int,
    is_train: bool,
) -> dict[str, np.ndarray]:
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    local_instability, vtvf_mixing = _neighbor_scores(
        train_emb,
        train_y,
        emb,
        y_pred,
        k=k,
        is_train=is_train,
    )
    return {
        "entropy": (entropy(probs) / np.log(probs.shape[1])).astype(np.float32),
        "msp": msp(probs).astype(np.float32),
        "knn": knn_distance(train_emb, emb, k=k + 1 if is_train else k).astype(np.float32),
        "prototype": prototype_distance(train_emb, train_y, emb).astype(np.float32),
        "local_instability": local_instability,
        "vtvf_mixing": vtvf_mixing,
        "softmax_vtvf_ambiguity": _softmax_boundary(probs),
        "y_true": y.astype(np.int64),
        "y_pred": y_pred,
    }


def _risk_from_components(
    comp: dict[str, np.ndarray],
    train_comp: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scaled = {
        name: _scale_component(name, comp[name], train_comp[name])
        for name in RISK_COMPONENTS
    }
    risk = np.zeros_like(scaled["entropy"], dtype=np.float32)
    for name, weight in weights.items():
        risk += float(weight) * scaled[name]
    risk = np.clip(risk, 0.0, 1.0).astype(np.float32)
    return risk, scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--components-out-dir",
        type=Path,
        default=None,
        help="Optional directory for component CSVs; defaults to the teacher run for backward compatibility.",
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--entropy-weight", type=float, default=0.30)
    parser.add_argument("--msp-weight", type=float, default=0.0)
    parser.add_argument("--local-instability-weight", type=float, default=0.25)
    parser.add_argument("--vtvf-mixing-weight", type=float, default=0.20)
    parser.add_argument("--knn-weight", type=float, default=0.15)
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.10)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.teacher_run_dir)
    out = args.out or (run_dir / "risk_targets.npz")
    components_out_dir = args.components_out_dir or run_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    components_out_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "entropy": args.entropy_weight,
        "msp": args.msp_weight,
        "local_instability": args.local_instability_weight,
        "vtvf_mixing": args.vtvf_mixing_weight,
        "knn": args.knn_weight,
        "prototype": args.prototype_weight,
        "softmax_vtvf_ambiguity": args.boundary_weight,
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one risk component weight must be positive.")
    weights = {name: value / weight_sum for name, value in weights.items()}

    train_logits, train_emb, train_y = _load(run_dir, "train")
    split_data = {
        "train": (train_logits, train_emb, train_y, True),
        "val": (*_load(run_dir, "val"), False),
        "test": (*_load(run_dir, "test"), False),
    }
    raw_components = {
        split: _components(train_emb, train_y, logits, emb, y, k=args.k, is_train=is_train)
        for split, (logits, emb, y, is_train) in split_data.items()
    }

    risks: dict[str, np.ndarray] = {}
    for split, comp in raw_components.items():
        risk, scaled = _risk_from_components(comp, raw_components["train"], weights)
        risks[split] = risk
        df = pd.DataFrame({name: values for name, values in scaled.items()})
        df["risk_target"] = risk
        df["y_true"] = comp["y_true"]
        df["y_pred"] = comp["y_pred"]
        df["is_error"] = df["y_true"] != df["y_pred"]
        df.to_csv(components_out_dir / f"risk_target_components_{split}.csv", index=False)

    np.savez(out, train=risks["train"], val=risks["val"], test=risks["test"])
    meta = {
        "teacher_run_dir": str(run_dir),
        "components_out_dir": str(components_out_dir),
        "k": args.k,
        "weights": weights,
        "neighbor_definition": "query predicted label versus training-neighbor ground-truth labels",
        "uses_query_ground_truth_for_risk_evidence": False,
        "scaling": {
            "bounded_components": "identity clip to [0,1]",
            "knn": "training 5th-95th percentile",
        },
        "risk_mean": {split: float(values.mean()) for split, values in risks.items()},
        "risk_std": {split: float(values.std()) for split, values in risks.items()},
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
