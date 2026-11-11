"""Minimal DeepBE ABE8e(V106W) example using extracted release weights.

The upstream demo scripts load all 9 PAM-variant models at once. For a compact
reproducibility check, this script uses a SpCas9-feature example and loads only
the corresponding PAM model plus the DeepBE_8e SavedModel.
"""

from __future__ import annotations

import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def one_hot_window(seq: str, start: int, length: int) -> np.ndarray:
    arr = np.zeros((length, 4), dtype=int)
    for i in range(length):
        nt = seq[i + start]
        if nt in "Aa":
            arr[i, 0] = 1
        elif nt in "Cc":
            arr[i, 1] = 1
        elif nt in "Gg":
            arr[i, 2] = 1
        elif nt in "Tt":
            arr[i, 3] = 1
        elif nt in "Xx":
            pass
        else:
            raise ValueError(f"Non-ATGC character {nt!r} in {seq!r}")
    return arr


def main() -> None:
    base = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "DeepBE"
    weights = base / "resources" / "extracted_models"
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = "GGATGAACAACAAACTGCCTTAGTAGGTCA"
    feature = 0  # SpCas9

    final_model = tf.keras.models.load_model(str(weights / "DeepBE" / "DeepBE_8e_model"), compile=False)
    pam_model = tf.keras.models.load_model(str(weights / "PAM" / "PAM_variant_SpCas9_model.h5"), compile=False)

    x_seq = np.stack([one_hot_window(target, start=5, length=10)])
    x_pam = np.stack([one_hot_window(target, start=0, length=30)])
    x_prop = np.stack([one_hot_window(target, start=4, length=20)])

    pam_score = float(pam_model.predict(x_pam, batch_size=128, verbose=0)[0, 0])
    pam_features = np.zeros((1, 9), dtype=float)
    pam_features[0, feature] = pam_score
    pam_sum = pam_features.sum(axis=1)

    raw = pd.DataFrame(final_model.predict([[x_seq, pam_sum], x_prop], batch_size=128, verbose=0))

    short_start = 6
    short_length = 8
    alist = []
    for x in range(short_length):
        b = []
        for i in range(1, 2**x + 1):
            for k in range(1, 2 ** (7 - x) + 1):
                b.append(2 ** (8 - x) * i - k - 1)
        alist.append(b)

    motif = target[short_start : short_start + short_length]
    non = [2 ** (7 - k) for k, nt in enumerate(motif) if nt in "Aa"]
    on = []
    for j in range(len(non)):
        for k in itertools.combinations(non, j + 1):
            on.append(sum(k))
    on = [x - 1 for x in on]
    off = list(set(range(255)) - set(on))

    eff = raw.iloc[0].sum()
    raw.iloc[0, off] = 0
    patt = raw.iloc[0].sum()
    raw.iloc[0] = (raw.iloc[0] / patt) * eff

    rows = []
    for idx, score in enumerate(raw.iloc[0].tolist()):
        if score == 0:
            continue
        edited = target
        for k, indices in enumerate(alist):
            if idx in indices:
                edited = edited[: k + short_start] + "g" + edited[k + 1 + short_start :]
        rows.append(
            {
                "target + PAM": target,
                "edited output": edited,
                "PAM_variant": feature,
                "Prediction score": float(score),
            }
        )

    result = pd.DataFrame(rows)
    out = out_dir / "DeepBE_8e_SpCas9_minimal_prediction.csv"
    result.to_csv(out, index=False)
    print(f"Output: {out}")
    print(f"Rows: {len(result)}")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
