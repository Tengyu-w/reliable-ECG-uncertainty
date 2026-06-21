from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .metrics import softmax


def msp(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs.max(axis=1)


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=1)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    return -temperature * (np.log(np.exp(z).sum(axis=1)) + logits.max(axis=1) / temperature)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(temp: float) -> float:
        probs = softmax(logits, temperature=temp)
        return float(-np.log(probs[np.arange(y.size), y] + 1e-12).mean())

    result = minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded")
    return float(result.x)


def prototype_distance(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray) -> np.ndarray:
    centroids = np.stack([train_emb[train_y == c].mean(axis=0) for c in np.unique(train_y)])
    distances = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) ** 0.5
    return distances.min(axis=1)


def mahalanobis_distance(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray) -> np.ndarray:
    scores = []
    for c in np.unique(train_y):
        class_emb = train_emb[train_y == c]
        cov = LedoitWolf().fit(class_emb)
        centered = emb - cov.location_
        md = np.sqrt(np.sum(centered @ cov.precision_ * centered, axis=1))
        scores.append(md)
    return np.min(np.stack(scores, axis=1), axis=1)


def knn_distance(train_emb: np.ndarray, emb: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_emb)
    distances, _ = nn.kneighbors(emb)
    return distances[:, -1]


def error_detection_metrics(y_true: np.ndarray, probs: np.ndarray, scores: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    y_pred = probs.argmax(axis=1)
    is_error = (y_pred != y_true).astype(int)
    rows = []
    for name, score in scores.items():
        if len(np.unique(is_error)) < 2:
            auroc = np.nan
            aupr = np.nan
        else:
            auroc = roc_auc_score(is_error, score)
            aupr = average_precision_score(is_error, score)
        rows.append({"score": name, "error_auroc": float(auroc), "error_aupr": float(aupr)})
    return rows
