# SAMI-Audio

**Porting SAMI (Score-based Autoencoder for Multiscale Inference) from images to audio.**
Antonio Pietro Romito (1932500) & Alessandro Gautieri (2041850) — Deep Learning & Applied AI 2025/26, Sapienza.

We take SAMI (Lyo, Simoncelli & Savin, 2025) and apply it to instrumental notes from NSynth (Engel et al., 2017). The goal: learn a latent space that separates **pitch** from **timbre** *without any supervision on those factors*. The twist of SAMI — and the reason we chose it — is that there is **no decoder network**: the "decoder" is a frozen diffusion denoiser, and the encoder influences generation only through a guidance gradient. Disentanglement is meant to emerge from that score mechanism, not from a reconstruction bottleneck.

This repo is the code and the artifacts behind our report; the report itself has the full story and the numbers.

---

## What we found (short version)

All metrics below are on the **held-out NSynth test split** (instrument-disjoint from training), computed with the *same* probes on both SAMI and a β-VAE baseline retrained at D=32 for a fair comparison.

- The latent **does** encode pitch and timbre linearly and without supervision: pitch is linearly decodable (R² ≈ 0.88), timbre is recovered by a k-NN at **0.97** (chance 25%), and the pitch/timbre directions come out essentially **orthogonal** (|cos| ≈ 0.03 vs. 0.28 for the β-VAE).
- Against the β-VAE, SAMI wins on the *separation-oriented* metrics (timbre, MIG-family, orthogonality). The β-VAE scores higher only on the raw pitch probe — but that reflects **more redundant pitch information**, not better disentanglement (its MIG is lower).
- The honest limit: **generative pitch control is weak**. In the frozen-denoiser regime the guidance often can't overcome the initial sampling noise, so the timbre-transfer demo succeeds on roughly half the seeds. We characterize *why* rather than hide it.

The single most useful lesson of the project: **most of our early "failures" were the measuring instrument, not the model.** We found and fixed four measurement artifacts (an unregularized probe giving fake R², a collapse guard reading the wrong KL, a categorical factor scored with the wrong metric, an FFT-peak pitch estimator returning harmonics). A validated metric was the precondition for every correct decision.

---

## How it works (the mechanism)

A classic VAE pushes the latent through a deterministic decoder in one shot, and a strong decoder learns to ignore the latent (posterior collapse). SAMI removes that decoder entirely:

```
ε̂(x_t, t, z) = ε_θ(x_t, t) − s · γ_t · g_t ,   g_t = ∇_{x_t} log q_φ(z | x_t)
```

The frozen denoiser `ε_θ` generates; the encoder's latent `z` only *bends* the denoising trajectory through the guidance gradient `g_t`, applied at every reverse-diffusion step. Because the gradient flows back into the encoder, this bending is also what *trains* the representation — a sampling technique turned into a learning signal.

Two things made this actually work on audio, and both were non-obvious:
- **Free bits.** Without them the posterior collapsed at every β. The KL has only two bad equilibria (degenerate at β=0, collapsed at β>0); giving each latent dimension a free KL budget (λ=0.5) creates the middle ground where the encoder can hold information.
- **High-noise oversampling.** We measured that the guidance only has leverage at high noise levels (the latent barely affects low-noise denoising), so we sample timesteps from Beta(4,1) instead of uniformly, concentrating training where the encoder actually gets a signal.

---

## Repository structure

```
progetto-deep/
├── data/
│   ├── nsynth.py             # NSynthDataset + CachedMelDataset, global norm, mel_to_audio
│   └── norm_stats.json       # global normalization constants (fixed, reused at eval)
├── models/
│   ├── encoder.py            # MelEncoder: Half-UNet → (μ, σ²), flatten (no global pool)
│   ├── unet.py               # MelUNet: 2D denoiser (the frozen "decoder")
│   ├── sami.py               # SAMI core: guidance gradient, loss, guided DDIM sampling
│   ├── vae.py                # β-VAE baseline
│   └── losses/               # DiffusionSchedule, KL (+ free bits), Mahalanobis log-prob
├── scripts/                  # training, precompute, metrics, demo, figures
├── plots/
│   ├── finals/               # report figures
│   └── demo/                 # timbre-transfer audio (per s, α, seed)
├── train.py                  # toy / disks / denoiser / SAMI training entry points
├── evaluate.py               # metrics (MIG, R² probe, kNN, cosine, PR)
├── metrics_comparison.py     # paired SAMI-vs-β-VAE evaluation on the held-out split
├── interactive_demo.ipynb    # inference-only demo (listen to a transfer inline)
└── README.md
```

Not in the repo (on the cluster): model checkpoints, raw NSynth audio, the ~7.6 GB mel cache, logs, and the Singularity container.

---

## Reproducing the pipeline

The order matters (normalization constants and the mel cache are built once and reused):

| Step | What it does | Command |
|------|--------------|---------|
| Dataset | filter NSynth to 4 families, pitch 48–84 | `bash data/download.sh` |
| Norm stats | global min/max constants → `norm_stats.json` | `python scripts/compute_norm_stats.py` |
| Mel cache | precompute mels (kills the I/O bottleneck) | `python scripts/precompute_mels.py` |
| Denoiser (3a) | unconditional DDPM on NSynth | `sbatch scripts/train_denoiser_2d.slurm` |
| Encoder (3b) | frozen-SAMI encoder (D=32, β=1e-5, free bits) | `sbatch scripts/train_sami_encoder.slurm` |
| Metrics | paired SAMI vs β-VAE on the held-out test split | `python scripts/metrics_comparison.py` |
| Demo | timbre transfer (s=5, α=0.3) | `python scripts/demo_fase2.py` |

**A note on the cluster.** Jobs on the Sapienza cluster are capped at 29 minutes, and one training epoch can exceed that on some nodes. We save per-step checkpoints (every 1500 steps) and auto-resume mid-epoch — without this the training would never make progress. This constraint shaped a lot of the engineering (mel caching, batch size, checkpoint frequency).

---

## The demo

`plots/demo/` contains the generated audio, one WAV per condition, named `s{scale}_a{alpha}_seed{seed}_{A,B,T}.wav`:
- `_A` — source note (guitar, MIDI 60), whose **timbre** we keep;
- `_B` — target note (brass, MIDI 67), whose **pitch** we transfer;
- `_T` — the transfer.

The transfer works cleanly on some seeds and not others (e.g. `seed1006` succeeds, `seed1000` doesn't) — this is the frozen-regime limit described in the report, not a bug: the outcome depends on the initial noise x_T. `interactive_demo.ipynb` lets you run one transfer end-to-end and listen inline; it needs only the two checkpoints from the GitHub release (`model_final_denoiser.pt`, `model_final_sami.pt`).

---

## Setup

Python 3.12, PyTorch, torchaudio, scikit-learn, and `torchcrepe` for pitch estimation; Griffin-Lim (or HiFi-GAN) for mel→audio in the demo only. See `pyproject.toml` / `requirements.txt`. NSynth is filtered to 4 families (guitar, keyboard, string, brass), MIDI pitch [48, 84], mel shape (1, 128, 256), globally normalized to [−1, 1].