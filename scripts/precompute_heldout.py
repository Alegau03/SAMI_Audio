#!/usr/bin/env python3
"""
SAMI-Audio — Precompute the held-out (NSynth test) mel cache
==============================================================
Reuses the EXACT same preprocessing as the train cache
(scripts/precompute_mels.py, same NSynthDataset -> same mel config and the
same data/norm_stats.json global constants — normalization is part of the
model and must NOT be recomputed on the held-out set).

Run AFTER downloading and filtering data/nsynth-test
(4 families, pitch 48-84; filter script identical to data/download.sh):

    python scripts/precompute_heldout.py

Output:
    data/mel_cache_test.npy   — (N, 1, 128, 256) float32 memmap
    data/mel_meta_test.pkl    — metadata list (pitch, instrument_family, ...)
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from precompute_mels import precompute

ROOT = "data/nsynth-test"
CACHE = "data/mel_cache_test.npy"
META_OUT = "data/mel_meta_test.pkl"


def main():
    assert os.path.isdir(os.path.join(ROOT, "audio")), (
        f"{ROOT}/audio missing — run the download+filter step first")
    precompute(root=ROOT, cache=CACHE, meta_out=META_OUT)
    print(f"[DONE] {CACHE} + {META_OUT}")


if __name__ == "__main__":
    main()
