from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = np.array(["SR", "VT", "VF"])


def make_waveforms(n_per_class: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label in range(3):
        for _ in range(n_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            effective_label = label
            if label in (1, 2) and rng.random() < 0.18:
                effective_label = 1 if label == 2 else 2
            if effective_label == 0:
                freq = rng.normal(5.0, 0.4)
                signal = 0.9 * np.sin(2 * np.pi * freq * t + phase)
                signal += 0.20 * np.sin(2 * np.pi * 2 * freq * t + phase / 2)
                noise = rng.normal(0, 0.12, size=length)
            elif effective_label == 1:
                freq = rng.normal(9.0, 0.9)
                signal = 0.75 * np.sin(2 * np.pi * freq * t + phase)
                signal += 0.28 * np.sign(np.sin(2 * np.pi * freq * t + phase))
                noise = rng.normal(0, 0.18, size=length)
            else:
                freqs = rng.normal([8.0, 13.0, 17.0], [1.5, 2.0, 2.5])
                amps = rng.uniform(0.18, 0.45, size=3)
                signal = sum(
                    amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                    for amp, freq in zip(amps, freqs)
                )
                signal += rng.normal(0, 0.35, size=length)
                noise = rng.normal(0, 0.22, size=length)
            baseline = rng.normal(0, 0.05) + 0.05 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t)
            x = signal + noise + baseline
            x = (x - x.mean()) / (x.std() + 1e-8)
            xs.append(x.astype(np.float32))
            ys.append(label)
    return np.stack(xs), np.array(ys)


def spectral_entropy(power: np.ndarray) -> float:
    p = power / (power.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def extract_features(x: np.ndarray, fs: float) -> np.ndarray:
    rows = []
    for row in x:
        freqs, power = welch(row, fs=fs, nperseg=min(128, len(row)))
        dominant = freqs[np.argmax(power)]
        entropy = spectral_entropy(power)
        line_length = np.abs(np.diff(row)).sum()
        autocorr = np.corrcoef(row[:-1], row[1:])[0, 1]
        rows.append(
            [
                row.mean(),
                row.std(),
                np.percentile(row, 95) - np.percentile(row, 5),
                dominant,
                entropy,
                line_length,
                autocorr,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = pred == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def vtvf_cross_errors(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int((((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))).sum())


def plot_embedding(x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    coords = PCA(n_components=2, random_state=0).fit_transform(x_test)
    colors = np.array(["#4C78A8", "#F58518", "#54A24B"])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, name in enumerate(CLASS_NAMES):
        mask = y_test == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7, label=name, color=colors[label])
    cross = ((y_test == 1) & (y_pred == 2)) | ((y_test == 2) & (y_pred == 1))
    ax.scatter(coords[cross, 0], coords[cross, 1], s=55, marker="x", linewidth=1.5, color="#D62728", label="VT/VF cross-error")
    ax.set_title("Synthetic quick demo: PCA geometry and VT/VF cross-errors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public synthetic ECG-like smoke demo.")
    parser.add_argument("--n-per-class", type=int, default=160)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("quick_demo") / "output")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    x, y = make_waveforms(args.n_per_class, args.length, args.seed)
    features = extract_features(x, fs=float(args.length))
    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.35, random_state=args.seed, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=args.seed),
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)
    y_pred = prob.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "ece": expected_calibration_error(y_test, prob),
        "vtvf_cross_errors": vtvf_cross_errors(y_test, y_pred),
        "total_errors": int((y_test != y_pred).sum()),
    }

    pd.DataFrame([metrics]).to_csv(args.out / "synthetic_quick_demo_metrics.csv", index=False)
    pd.DataFrame(
        {
            "y_true": CLASS_NAMES[y_test],
            "y_pred": CLASS_NAMES[y_pred],
            "prob_SR": prob[:, 0],
            "prob_VT": prob[:, 1],
            "prob_VF": prob[:, 2],
        }
    ).to_csv(args.out / "synthetic_quick_demo_predictions.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(args.out / "synthetic_quick_demo_confusion_matrix.csv")
    plot_embedding(x_test, y_test, y_pred, args.out / "synthetic_quick_demo_embedding.png")

    print("Synthetic quick demo complete.")
    print("This is a non-clinical smoke test, not dissertation evidence.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Output directory: {args.out.resolve()}")


if __name__ == "__main__":
    main()
