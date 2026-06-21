from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | list[list[int]]]:
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sensitivity = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    specificity = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificity.append(tn / max(tn + fp, 1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "nll": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "macro_sensitivity": float(np.mean(sensitivity)),
        "macro_specificity": float(np.mean(specificity)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class_index": int(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        ],
        "vt_as_vf": int(cm[1, 2]),
        "vf_as_vt": int(cm[2, 1]),
        "vt_as_sr": int(cm[1, 0]),
        "vf_as_sr": int(cm[2, 0]),
        "sr_as_vt": int(cm[0, 1]),
        "sr_as_vf": int(cm[0, 2]),
        "vtvf_cross_errors": int(cm[1, 2] + cm[2, 1]),
        "total_errors": int((y_true != y_pred).sum()),
    }


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)
