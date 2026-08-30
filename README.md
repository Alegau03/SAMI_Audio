# SAMI-Audio

**Score-based Autoencoders for Multiscale Inference (SAMI) applied to the unsupervised representation of musical instrument notes.**

An audio-domain port of SAMI (Lyo, Simoncelli & Savin, 2025) on NSynth (Engel et al., 2017): a variational encoder learns a latent space where **pitch** and **timbre** are separable and manipulable, with no supervision on the factors. The "decoder" is not a separate network but the denoiser of a diffusion model, guided from the latent through a score gradient.

**Project — Deep Learning & Applied AI 2025/26, Sapienza University of Rome.**

---

## Key results

| Metric | Value | Notes |
|--------|-------|-------|
| Pitch (decodability) | **R² = 0.60** | Ridge probe + train/test split + noise control |
| Timbre (decodability) | **accuracy = 0.91** | k-NN on μ, chance baseline 25% |
| Pitch/timbre separability | **cos(w_pitch, w_family) = 0.15** | orthogonal (random baseline 3σ = 0.53) |
| Timbre transfer | **3/7 seeds** with full transfer | s=5, α=0.3 (honest outcome, fully documented) |
| Posterior collapse | **Solved with free bits** | raw KL alive (31 nats at D=128) |

**Artifacts in this repository:**
- Figure of the report: `plots/finals/` (7 figures)
- Demo audio: `plots/demo/` (147 WAV: original notes, transfer, controls)
- Model checkpoints and full documentation live on the cluster (not pushed)

---

## Repository structure

```
progetto-deep/
├── data/
│   ├── nsynth.py             # Dataset, global normalization, mel_to_audio
│   └── norm_stats.json       # Global normalization constants
├── models/
│   ├── encoder.py            # MelEncoder: Half-UNet → (μ, σ²)
│   ├── unet.py               # MelUNet: 2D denoiser (the "decoder")
│   ├── sami.py               # SAMI core: guidance, loss, sampling
│   ├── vae.py                # β-VAE baseline
│   └── losses/               # DiffusionSchedule, KL, Mahalanobis
├── scripts/                  # Active pipeline (training, demo, figures)
├── plots/
│   ├── finals/               # Report figures (7 PNG)
│   └── demo/                 # Demo audio (147 WAV)
├── train.py                  # Training (toy, disks, denoiser, SAMI)
├── evaluate.py               # Metrics (MIG, R² probe, frechet_mel)
├── interactive_demo.ipynb    # Interactive Notebook for demo
├── pyproject.toml / requirements.txt / pdm.lock
└── README.md
```

Not in this repository (kept on the cluster): `checkpoints/` (model weights), `data/nsynth-train/` (raw audio), `data/mel_cache.npy` (7.6 GB cache), Singularity container, logs, archive of past experiments, and the full documentation (scientific report, diagnostics, phase notes — detailed in the project report PDF).

---

## Pipeline (reproduction)

| Phase | What | Script | Duration |
|-------|------|--------|----------|
| **0 — Dataset** | Filter NSynth (4 families, pitch 48-84) | `data/download.sh` | — |
| **0.5 — Normalization** | Global min-max constants | `python scripts/compute_norm_stats.py` | ~10 min |
| **0.5 — Mel cache** | Pre-compute mels (removes I/O bottleneck) | `python scripts/precompute_mels.py` | ~1-2 h |
| **1 — Toy** | Mechanism validation (sinusoids) | `python train.py --mode sami` | minutes |
| **2 — Baseline** | β-VAE on NSynth (D=128, β=0.01) | `python train.py --mode vae-nsynth` | hours |
| **2.7 — Disks gate** | 2D stack validation (3 factors) | `python train.py --mode sami --dataset disks` | minutes |
| **3a — Denoiser** | Unconditional DDPM on NSynth | `sbatch scripts/train_denoiser_2d.slurm` | ~1 day |
| **3a — DDIM gate** | Check plausible mels | `python scripts/ddim_gate.py` | minutes |
| **3b — Encoder** | Frozen SAMI (D=32, β=1e-5, free bits) | `sbatch scripts/train_sami_encoder.slurm` | ~1 day |
| **Demo** | Timbre transfer (s=5, α=0.3) | `python scripts/demo_fase2.py` | ~5 min |
| **Figures** | 7 report figures | `bash scripts/run_report_figures.sh` | ~3 min |

**SLURM infrastructure:** 29-min jobs with auto-resume. Per-step checkpoints (every 1500 steps) guarantee that a job killed mid-epoch resumes from the latest step — without them training would never progress (one epoch exceeds the limit on some nodes).

---

## Demo audio

The sounds generated in the final demo are in **`plots/demo/`** — one WAV per condition:

- `_A.wav` — original note (guitar, MIDI 60)
- `_B.wav` — target note (trumpet, MIDI 67)
- `_T.wav` — **transfer** (pitch of B on the timbre of A)

**The most significant (s=5, α=0.3):**

| File | What it shows |
|------|---------------|
| [s5.0_a0.3_seed1006_T.wav](plots/demo/s5.0_a0.3_seed1006_T.wav) | **Perfect transfer** — pitch toward trumpet, guitar timbre preserved |
| [s5.0_a0.3_seed1001_T.wav](plots/demo/s5.0_a0.3_seed1001_T.wav) | Shifted pitch, timbre preserved (with α=0.5 the timbre degraded) |
| [s5.0_a0.3_seed1000_T.wav](plots/demo/s5.0_a0.3_seed1000_T.wav) | Failure: guidance does not move the pitch (T=A) |

Full configurations (s∈{3,5,7} × α∈{0.3,0.5,0.7,1.0}, 8 seeds each) are organized by prefix `s{scale}_a{alpha}_seed{seed}_{A,B,T}.wav`.

**Report figure** (A\|T\|B spectrograms of the three cases): `plots/finals/fig6_transfer_demo.png`.

---

## Metrics and method (summary)

- **R² probe (Ridge + split + noise control)** — how linearly decodable a factor is from the latent. The margin over the control is the primary metric.
- **Classification accuracy** — the correct metric for timbre (categorical); Ridge R² on a categorical target is an artifact (0.29 → 0.91 with the right metric).
- **Per-dimension KL** — distinguishes entanglement (information spread out) from disentanglement (few loaded dims).
- **Cosine between generative directions** — factor orthogonality, interpreted against a random baseline.
- **CREPE** — perceptual fundamental frequency for pitch (robust to harmonics and octave errors).

Four measurement artifacts were discovered and fixed during the research — the measurement instrument is part of the experiment.

---

## Method (the idea)

In a classic VAE, a deterministic decoder compresses the latent in a single z→x pass, and an expressive decoder tends to ignore z (posterior collapse). **SAMI replaces the decoder with a pre-trained frozen diffusion model**: the encoder influences generation via the guidance gradient

```
ε̂(x_t, t, z) = ε_θ(x_t, t) − s · γ_t · g_t,      g_t = ∇_{x_t} log q_φ(z | x_t)
```

applied step-by-step during the reverse diffusion. The "decoder" is therefore the **guided denoiser** — an iterative generative process, not a network. This:

1. removes the encoder/decoder competition at the root of posterior collapse,
2. acts per **noise scale** (pitch emerges at high t, timbre at low t),
3. gives the encoder a richer, more localized learning signal.

Without labels, the learned latent is **linearly separable** in pitch and timbre (R² 0.60, accuracy 0.91) with **orthogonal** directions (cos = 0.15).

---

## Interactive demo notebook

**`interactive_demo.ipynb`** is a self-contained, inference-only notebook: it loads
the two final checkpoints, performs the timbre transfer (guitar A + pitch of trumpet B)
and lets you **listen** to the result inline. It needs no raw NSynth data — everything
is packed in `data/demo_inference_data.npz` (already in the repo).

To run it you need: PyTorch, torchaudio, matplotlib, scikit-learn (and optionally
`torchcrepe` for the pitch measurements) plus the two `model_final.pt` checkpoints,
which are **not pushed to this repo** — download them from the
[GitHub release](https://github.com/Alegau03/progetto-deep/releases) and place them at:

| Release asset | Rename to |
|---|---|
| `model_final_denoiser.pt` | `checkpoints/nsynth/denoiser_2d/model_final.pt` |
| `model_final_sami.pt` | `checkpoints/nsynth/sami_d32/model_final.pt` |

The markdown cells explain step by step what each cell does and what you should hear.

The full pre-generated audio of every (s, α, seed) configuration is in `plots/demo/`.

---

## Reproducibility

- Python 3.12, PyTorch, `torchcrepe` (PyTorch port), HiFi-GAN / Griffin-Lim for audio, see `pyproject.toml` and `requirements.txt`
- Training runs on the DI Sapienza SLURM cluster (RTX 6000, 29-min jobs, Singularity container, auto-resume)
- Raw NSynth filtered to 4 families (guitar, keyboard, string, brass), MIDI pitch [48, 84], 61,531 samples, mel (1,128,256), global normalization to [-1,1]


---
