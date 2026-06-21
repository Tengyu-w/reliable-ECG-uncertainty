from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    args = parser.parse_args()

    print("Variables:")
    for item in whosmat(args.mat):
        print(" ", item)

    mat = loadmat(args.mat, squeeze_me=False, struct_as_record=False)
    for key in ("SR", "VT", "VF"):
        cells = mat[key].ravel()
        lengths = []
        shapes = []
        for cell in cells:
            x = np.asarray(cell).squeeze()
            if x.ndim == 2:
                x = x[:, 0]
            lengths.append(x.size)
            shapes.append(np.asarray(cell).shape)
        print(
            f"{key}: n={len(lengths)}, min={min(lengths)}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}, "
            f"first_shapes={shapes[:5]}"
        )


if __name__ == "__main__":
    main()
