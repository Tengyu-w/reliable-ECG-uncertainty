from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.io import loadmat
from sklearn.model_selection import GroupShuffleSplit, train_test_split


LABELS = {"SR": 0, "VT": 1, "VF": 2}
CLASS_NAMES = ["SR", "VT", "VF"]
REGULARITY_FEATURE_NAMES = [
    "spectral_entropy",
    "dominant_frequency",
    "dominant_frequency_concentration",
    "spectral_centroid",
    "spectral_bandwidth",
    "autocorr_peak",
    "autocorr_peak_lag_s",
    "zero_crossing_rate",
    "line_length",
]


@dataclass(frozen=True)
class ECGSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class ECGDataset:
    x: np.ndarray
    y: np.ndarray
    window_ids: np.ndarray
    record_ids: np.ndarray


def build_duplicate_family_groups(
    x: np.ndarray,
    record_ids: np.ndarray,
    decimals: int | None = None,
) -> np.ndarray:
    """Merge records connected by identical windows into one split group."""
    records = np.asarray(record_ids)
    unique_records = np.unique(records)
    parent = {record: record for record in unique_records}

    def find(record: str) -> str:
        root = record
        while parent[root] != root:
            root = parent[root]
        while parent[record] != record:
            next_record = parent[record]
            parent[record] = root
            record = next_record
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    hash_owner: dict[str, str] = {}
    rows = x.reshape(len(x), -1)
    for row, record in zip(rows, records):
        values = row
        if decimals is not None:
            values = np.round(values, decimals=decimals).astype(np.float32)
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        owner = hash_owner.get(digest)
        if owner is None:
            hash_owner[digest] = str(record)
        else:
            union(owner, str(record))
    return np.asarray([f"duplicate_family::{find(str(record))}" for record in records])


def _first_channel(cell: np.ndarray) -> np.ndarray:
    x = np.asarray(cell).squeeze()
    if x.ndim == 2:
        x = x[:, 0]
    return x.astype(np.float32)


def _windows(signal: np.ndarray, length: int, stride: int) -> list[np.ndarray]:
    if signal.size < length:
        return []
    return [signal[start : start + length] for start in range(0, signal.size - length + 1, stride)]


def load_rhythm_windows(
    mat_path: str | Path,
    sample_rate: int = 100,
    seconds: int = 5,
    stride_seconds: int = 5,
    max_windows_per_record: int | None = None,
) -> ECGDataset:
    """Load RHYTHMS.mat and convert each record into fixed 5 s windows.

    Returns x shaped [n, 1, length], labels, window ids, and original record ids.
    Record ids are used for leakage-free grouped train/validation/test splits.
    """
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    length = sample_rate * seconds
    stride = sample_rate * stride_seconds

    xs: list[np.ndarray] = []
    ys: list[int] = []
    window_ids: list[str] = []
    record_ids: list[str] = []
    for rhythm, label in LABELS.items():
        for idx, cell in enumerate(mat[rhythm].ravel()):
            signal = _first_channel(cell)
            segments = _windows(signal, length=length, stride=stride)
            if max_windows_per_record is not None:
                segments = segments[:max_windows_per_record]
            record_id = f"{rhythm}_{idx:04d}"
            for seg_idx, segment in enumerate(segments):
                segment = normalize_window(segment)
                xs.append(segment[None, :])
                ys.append(label)
                window_ids.append(f"{record_id}_{seg_idx:04d}")
                record_ids.append(record_id)

    return ECGDataset(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        window_ids=np.asarray(window_ids),
        record_ids=np.asarray(record_ids),
    )


def normalize_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + eps)


def extract_regularity_features(x: np.ndarray, sample_rate: int = 100) -> np.ndarray:
    """Fast rhythm/frequency descriptors for one normalised ECG window."""
    signal = np.asarray(x, dtype=np.float32).squeeze()
    freqs, psd = welch(signal, fs=sample_rate, nperseg=min(256, len(signal)))
    mask = (freqs >= 0.5) & (freqs <= 40.0)
    freqs, psd = freqs[mask], np.maximum(psd[mask], 1e-12)
    p = psd / psd.sum()
    spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(p))
    peak_idx = int(np.argmax(psd))
    dominant_frequency = float(freqs[peak_idx])
    dominant_concentration = float(psd[peak_idx] / psd.sum())
    spectral_centroid = float(np.sum(freqs * p))
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * p)))

    centred = signal - signal.mean()
    ac = np.correlate(centred, centred, mode="full")[len(signal) - 1 :]
    ac = ac / max(float(ac[0]), 1e-12)
    min_lag = max(1, int(0.12 * sample_rate))
    max_lag = min(len(ac), int(1.5 * sample_rate))
    autocorr_peak = 0.0
    autocorr_peak_lag_s = 0.0
    if max_lag > min_lag:
        search = ac[min_lag:max_lag]
        peak_rel = int(np.argmax(search))
        autocorr_peak = float(search[peak_rel])
        autocorr_peak_lag_s = float((min_lag + peak_rel) / sample_rate)

    dx = np.diff(signal)
    return np.asarray(
        [
            spectral_entropy,
            dominant_frequency,
            dominant_concentration,
            spectral_centroid,
            spectral_bandwidth,
            autocorr_peak,
            autocorr_peak_lag_s,
            float(np.mean(np.diff(np.signbit(signal)) != 0)),
            float(np.sum(np.abs(dx)) / len(signal)),
        ],
        dtype=np.float32,
    )


def extract_regularity_features_batch(x: np.ndarray, sample_rate: int = 100) -> np.ndarray:
    return np.stack([extract_regularity_features(window, sample_rate=sample_rate) for window in x]).astype(np.float32)


def make_splits(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> ECGSplits:
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(x, y, groups=groups))
        val_fraction = val_size / (1.0 - test_size)
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_rel, val_rel = next(
            splitter.split(x[trainval_idx], y[trainval_idx], groups=groups[trainval_idx])
        )
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        return ECGSplits(x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx])

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_trainval,
    )
    return ECGSplits(x_train, y_train, x_val, y_val, x_test, y_test)
