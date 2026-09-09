"""
SAMI-Audio — Compute Global Normalization Stats
=================================================
Computes log-mel min and max (robust percentiles) across the full NSynth
dataset. Outputs data/norm_stats.json for use by _normalize_global.

Run once after downloading NSynth, before precompute_mels.py.
"""

import json
import numpy as np
from data.nsynth import NSynthDataset


def main(root: str = "data/nsynth-train", out: str = "data/norm_stats.json"):
    """Calcola i percentili 0.1/99.9 del log-mel su tutto il dataset.

    La normalizzazione globale usa due costanti fisse (robuste agli outlier)
    invece della normalizzazione per-sample, che distruggeva la dinamica
    relativa tra campioni.

    Parameters
    ----------
    root : str
        Cartella NSynth di input.
    out : str
        Percorso del JSON di output con log_mel_min/log_mel_max.
    """
    dataset = NSynthDataset(root=root, normalize=False)
    N = len(dataset)
    print(f"Scanning {N} samples...")

    vals = []
    for i in range(N):
        mel, _ = dataset[i]
        vals.append(mel.numpy().flatten())
        if i % 5000 == 0:
            print(f"  {i}/{N}")

    allv = np.concatenate(vals)
    lo = float(np.percentile(allv, 0.1))
    hi = float(np.percentile(allv, 99.9))
    stats = {"log_mel_min": lo, "log_mel_max": hi}
    json.dump(stats, open(out, "w"))
    print(f"Saved: {out}")
    print(f"  log_mel_min = {lo:.4f}")
    print(f"  log_mel_max = {hi:.4f}")
    print(f"  → normalization to [-1, 1]: 2*(x - {lo:.3f})/({hi:.3f} - {lo:.3f}) - 1")


if __name__ == "__main__":
    main()
