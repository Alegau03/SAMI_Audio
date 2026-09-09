#!/usr/bin/env python3
"""
SAMI-Audio — Demo Inference Data
=================================
Prepares data/demo_inference_data.npz, a small self-contained artifact that
lets the interactive demo notebook (interactive_demo.ipynb) run WITHOUT the
raw NSynth dataset. The notebook only needs this file plus the two final
checkpoints.

Contents of the .npz:
    xA, xB          (1, 128, 256) log-mel of note A (guitar, MIDI 60) and
                    B (brass, MIDI 67), globally normalized to [-1, 1]
    wavA, wavB      (1, N) original NSynth audio of A and B (16 kHz)
    mu_ref          (N_REF, 32) encoded latents of a fixed reference set
    pitch_ref       (N_REF,) MIDI pitch of the reference set
    family_ref      (N_REF,) label-encoded family of the reference set
    mu_mean, mu_std (32,) standardization constants of the reference set
    w_pitch         (32,) pitch direction in the standardized latent space
    metaA, metaB    metadata dicts (entry_id, pitch, family)

Run once (after training):  python scripts/make_demo_inference_data.py
Output:                 data/demo_inference_data.npz (~1.5 MB, git-tracked)
"""
import os, sys, pickle
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nsynth import CachedMelDataset, NSYNTH_SAMPLE_RATE
from models.encoder import MelEncoder
from sklearn.preprocessing import LabelEncoder

CKPT = "checkpoints/nsynth/sami_d32/model_final.pt"
OUT = "data/demo_inference_data.npz"
N_REF = 8000
PITCH_A, PITCH_B = 60, 67
FAM_A, FAM_B = "guitar", "brass"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"[INFO] Device: {DEVICE}")
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    encoder = MelEncoder(in_channels=1, latent_dim=32, base_channels=64,
                         input_size=(128, 256)).to(DEVICE)
    sd = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items()
          if k.startswith("encoder.")}
    encoder.load_state_dict(sd, strict=True)
    encoder.eval()

    ds = CachedMelDataset()
    metas = ds.metas

    def find(family, pitch):
        for i, m in enumerate(metas):
            if m["instrument_family"] == family and m["pitch"] == pitch:
                return i
        raise RuntimeError(f"{family}/{pitch} not in cache")

    iA, iB = find(FAM_A, PITCH_A), find(FAM_B, PITCH_B)
    print(f"[INFO] A: idx={iA} {metas[iA]['instrument_family']}/{metas[iA]['pitch']}  "
          f"B: idx={iB} {metas[iB]['instrument_family']}/{metas[iB]['pitch']}")

    xA = torch.from_numpy(np.asarray(ds.arr[iA]).copy()).unsqueeze(0).to(DEVICE)
    xB = torch.from_numpy(np.asarray(ds.arr[iB]).copy()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        muA, _ = encoder(xA)
        muB, _ = encoder(xB)

    rng = np.random.default_rng(42)
    idx_ref = rng.choice(len(ds), N_REF, replace=False)
    mu_ref = []
    with torch.no_grad():
        for b in range(0, N_REF, 256):
            x = torch.from_numpy(np.asarray(ds.arr[idx_ref[b:b + 256]]).copy()).to(DEVICE)
            mu, _ = encoder(x)
            mu_ref.append(mu.cpu().numpy())
    mu_ref = np.concatenate(mu_ref)
    pitch_ref = np.array([metas[i]["pitch"] for i in idx_ref], dtype=float)
    family_ref = LabelEncoder().fit_transform(
        [metas[i]["instrument_family"] for i in idx_ref])

    mu_mean = mu_ref.mean(0)
    mu_std = mu_ref.std(0) + 1e-8
    mu_s = (mu_ref - mu_mean) / mu_std

    lo, hi = np.percentile(pitch_ref, [33, 67])
    w_pitch = mu_s[pitch_ref >= hi].mean(0) - mu_s[pitch_ref <= lo].mean(0)
    w_pitch = w_pitch / (np.linalg.norm(w_pitch) + 1e-12)

    def read_wav(entry_id):
        wav, sr = sf.read(os.path.join("data/nsynth-train/audio", f"{entry_id}.wav"))
        wav = torch.from_numpy(wav.T).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != NSYNTH_SAMPLE_RATE:
            raise RuntimeError(f"unexpected sr {sr}")
        return wav

    wavA = read_wav(metas[iA]["entry_id"])
    wavB = read_wav(metas[iB]["entry_id"])

    np.savez_compressed(
        OUT,
        xA=xA.cpu().numpy(), xB=xB.cpu().numpy(),
        wavA=wavA.numpy(), wavB=wavB.numpy(),
        mu_ref=mu_ref, pitch_ref=pitch_ref, family_ref=family_ref,
        mu_mean=mu_mean, mu_std=mu_std, w_pitch=w_pitch,
        muA=muA.cpu().numpy(), muB=muB.cpu().numpy(),
        metaA=metas[iA], metaB=metas[iB],
    )
    sz = os.path.getsize(OUT) / 1e6
    print(f"[DONE] {OUT} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
