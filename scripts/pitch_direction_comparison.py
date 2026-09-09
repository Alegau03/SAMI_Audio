#!/usr/bin/env python3
"""
Pitch-direction correlation corr(k, f0) — SAMI vs beta-VAE
============================================================
Runs the same procedure on BOTH models with the SAME setup:

  1. mu over 8000 reference samples (seed 42) from the model's own encoder;
  2. standardization + intervention direction w_pitch = top-minus-bottom
     pitch tercile means (normalized, standardized space);
  3. mu_A = real guitar/MIDI-60 note; sweep k in {-3..3} along w_pitch;
  4. generate: SAMI  -> guided DDIM (denoiser, seeded per k for reproducibility);
               beta-VAE -> direct MelDecoder decode (deterministic by nature);
  5. CREPE median f0 (confidence > 0.5) per k; Pearson corr(k, f0) over the
     measurable ks.

The generation mechanisms differ BY CONSTRUCTION (SAMI has no decoder
network; the beta-VAE has no guided sampler): the metric is the same
directional-control question, "does moving mu along w_pitch move the
fundamental?", asked of each model's own generative path.

USAGE: python scripts/pitch_direction_comparison.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nsynth import CachedMelDataset, mel_to_audio
from models.encoder import MelEncoder

SAMICKPT = "checkpoints/nsynth/sami_d32/model_final.pt"
VAECKPT = "checkpoints/nsynth/vae_d32_global/model_final.pt"
# Held-out evaluation (NSynth test split): reference set AND note A/B come
# from the held-out cache (precompute con scripts/precompute_heldout.py).
EVAL_CACHE = "data/mel_cache_test.npy"
EVAL_META = "data/mel_meta_test.pkl"
N_REF = None  # None = use ALL held-out samples as reference
KS = [-3, -2, -1, 0, 1, 2, 3]
SEED = 42
OUT_JSON = "pitch_direction_comparison.json"


def crepe_f0(wav, device, sr=16000, conf_thresh=0.5):
    import torchcrepe
    audio = wav.reshape(1, -1)
    out = torchcrepe.predict(
        audio, sr, hop_length=256, fmin=50, fmax=2000,
        model="full", batch_size=64, device=device, return_periodicity=True)
    if isinstance(out, tuple):
        f0, conf = out[0], out[1]
    else:
        f0 = out
        conf = torch.ones_like(f0)
    f0_np = f0[0].cpu().numpy()
    conf_np = conf[0].cpu().numpy()
    mask = ~np.isnan(f0_np) & (conf_np > conf_thresh)
    if mask.sum() < 5:
        return float("nan")
    return float(np.median(f0_np[mask]))


def load_bvae(ckpt_path, device):
    from models.vae import MelDecoder, BetaVAE
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = MelEncoder(in_channels=1, latent_dim=32, base_channels=64,
                     input_size=(128, 256)).to(device)
    sd_enc = {k[len("encoder."):]: v for k, v in ck["model_state_dict"].items()
              if k.startswith("encoder.")}
    enc.load_state_dict(sd_enc, strict=True)
    dec = MelDecoder(latent_dim=32, base_channels=128).to(device)
    sd_dec = {k[len("decoder."):]: v for k, v in ck["model_state_dict"].items()
              if k.startswith("decoder.")}
    dec.load_state_dict(sd_dec, strict=True)
    enc.eval()
    dec.eval()
    return enc, dec


def load_sami(ckpt_path, device):
    from models.unet import MelUNet
    from models.sami import SAMI
    from models.losses.diffusion import DiffusionSchedule
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = MelEncoder(in_channels=1, latent_dim=32, base_channels=64,
                     input_size=(128, 256)).to(device)
    sd = {k[len("encoder."):]: v for k, v in ck["model_state_dict"].items()
          if k.startswith("encoder.")}
    enc.load_state_dict(sd, strict=True)
    den = MelUNet(in_channels=1, base_channels=128,
                  channel_mult=(1, 1, 2, 2), time_dim=128).to(device)
    den.load_state_dict(torch.load(
        "checkpoints/nsynth/denoiser_2d/model_final.pt",
        map_location=device, weights_only=False)["model_state_dict"])
    for p in den.parameters():
        p.requires_grad = False
    den.eval()
    sami = SAMI(enc, den, DiffusionSchedule(T=1000, s=0.008), beta=1e-5,
                frozen_denoiser=True, oversample_t=True, free_bits=0.5).to(device)
    enc.eval()
    return sami


def ref_stats(encoder, ds, device):
    """mu_mean, mu_std, w_pitch (tercile intervention direction) for a model.
    Uses ALL held-out samples as reference (N_REF=None) or a fixed subset."""
    metas = ds.metas
    n = len(ds) if N_REF is None else min(N_REF, len(ds))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), n, replace=False) if N_REF is not None else np.arange(n)
    mu_ref = []
    with torch.no_grad():
        for b in range(0, n, 256):
            x = torch.from_numpy(np.asarray(ds.arr[idx[b:b + 256]]).copy()).to(device)
            mu, _ = encoder(x)
            mu_ref.append(mu.cpu().numpy())
    mu_ref = np.concatenate(mu_ref)
    pitch_ref = np.array([metas[i]["pitch"] for i in idx], dtype=float)
    mu_mean = mu_ref.mean(0)
    mu_std = mu_ref.std(0) + 1e-8
    mu_s = (mu_ref - mu_mean) / mu_std
    lo, hi = np.percentile(pitch_ref, [33, 67])
    w = mu_s[pitch_ref >= hi].mean(0) - mu_s[pitch_ref <= lo].mean(0)
    w = w / (np.linalg.norm(w) + 1e-12)
    return mu_mean, mu_std, w


def find_idx(metas, family, pitch):
    for i, m in enumerate(metas):
        if m["instrument_family"] == family and m["pitch"] == pitch:
            return i
    raise RuntimeError(f"{family}/{pitch} not found")


def sweep_corr(generate_fn, muA, mu_mean, mu_std, w_pitch, device):
    """Sweep mu_A + k*w_pitch; returns {k: f0} and Pearson corr over valid ks."""
    f0s = {}
    for k in KS:
        mu_k_s = (muA - mu_mean) / mu_std + k * w_pitch
        mu_k = mu_k_s * mu_std + mu_mean
        wav = generate_fn(mu_k)
        f0s[k] = crepe_f0(wav, device)
    ks_ok = [k for k in KS if not np.isnan(f0s[k])]
    f_ok = [f0s[k] for k in ks_ok]
    corr = float(np.corrcoef(ks_ok, f_ok)[0, 1]) if len(ks_ok) >= 3 else float("nan")
    return {str(k): (None if np.isnan(v) else round(v, 1)) for k, v in f0s.items()}, corr, len(ks_ok)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    ds = CachedMelDataset(cache=EVAL_CACHE, meta_path=EVAL_META)
    metas = ds.metas
    iA = find_idx(metas, "guitar", 60)
    xA = torch.from_numpy(np.asarray(ds.arr[iA]).copy()).unsqueeze(0).to(device)
    results = {}

    # ---- SAMI: guided DDIM sampling (seeded per k) ----
    print("[INFO] Loading SAMI...")
    sami = load_sami(SAMICKPT, device)
    with torch.no_grad():
        muA, _ = sami.encoder(xA)
    muA = muA.cpu().numpy()[0]
    mm, ms, wp = ref_stats(sami.encoder, ds, device)

    def gen_sami(mu_v):
        g = torch.Generator(device=device)
        g.manual_seed(1000)          # same initial noise for every k: clean comparison
        with torch.no_grad():
            xgen = sami.sample_seeded(torch.from_numpy(mu_v).unsqueeze(0).to(device),
                                      (1, 128, 256), n_steps=50, generator=g)
        return mel_to_audio(torch.from_numpy(xgen[0, 0].cpu().numpy()).unsqueeze(0))

    print("=== SAMI: sweep mu_A + k*w_pitch (guided DDIM, seeded) ===")
    f0s_sami, corr_sami, n_sami = sweep_corr(gen_sami, muA, mm, ms, wp, device)
    results["sami"] = {"corr(k,f0)": corr_sami, "valid_ks": n_sami, "f0_by_k": f0s_sami}
    print(f"SAMI  corr(k,f0) = {corr_sami:.3f}  ({n_sami}/7 valid ks)")

    # ---- beta-VAE: direct decoder decode ----
    print("[INFO] Loading beta-VAE...")
    enc_vae, dec_vae = load_bvae(VAECKPT, device)
    with torch.no_grad():
        muA_v, _ = enc_vae(xA)
    muA_v = muA_v.cpu().numpy()[0]
    mm_v, ms_v, wp_v = ref_stats(enc_vae, ds, device)

    def gen_bvae(mu_v):
        with torch.no_grad():
            xgen = dec_vae(torch.from_numpy(mu_v).unsqueeze(0).to(device))
        return mel_to_audio(torch.from_numpy(xgen[0, 0].cpu().numpy()).unsqueeze(0))

    print("=== beta-VAE: sweep mu_A + k*w_pitch (direct decode) ===")
    f0s_vae, corr_vae, n_vae = sweep_corr(gen_bvae, muA_v, mm_v, ms_v, wp_v, device)
    results["bvae"] = {"corr(k,f0)": corr_vae, "valid_ks": n_vae, "f0_by_k": f0s_vae}
    print(f"beta-VAE corr(k,f0) = {corr_vae:.3f}  ({n_vae}/7 valid ks)")

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
