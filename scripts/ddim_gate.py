#!/usr/bin/env python3
"""
Qualità del denoiser: DDIM sampling incondizionato dal denoiser pre-addestrato.
Genera sample, li de-normalizza, salva PNG e audio WAV per ispezione.
"""
import os, sys, json, torch
import numpy as np
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import MelUNet
from models.losses.diffusion import DiffusionSchedule
from data.nsynth import mel_to_audio

CKPT = "checkpoints/nsynth/denoiser_2d/denoiser_epoch_0058.pt"
OUT = "plots/ddim_gate"
N_SAMPLES = 10
N_STEPS = 50
T = 1000

stats = json.load(open("data/norm_stats.json"))
LOG_MIN, LOG_MAX = stats["log_mel_min"], stats["log_mel_max"]


def denormalize(mel):
    """[-1,1] -> log-mel grezzo"""
    return (mel + 1.0) / 2.0 * (LOG_MAX - LOG_MIN) + LOG_MIN


def main():
    """Genera campioni DDIM dal denoiser e li salva come PNG + WAV.

    Verifica la qualità del denoiser: deve produrre mel con struttura
    armonica (fondamentale + armoniche) e pitch definito, condizione
    necessaria per una guidance sensata nel training dell'encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    denoiser = MelUNet(in_channels=1, base_channels=128,
                       channel_mult=(1, 1, 2, 2), time_dim=128).to(device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()
    print(f"[INFO] Loaded {CKPT} (epoch {ckpt.get('epoch', '?')})")

    diffusion = DiffusionSchedule(T=T, s=0.008).to(device)
    os.makedirs(OUT, exist_ok=True)

    step_idx = torch.linspace(T - 1, 0, N_STEPS, dtype=torch.long, device=device)

    with torch.no_grad():
        xt = torch.randn(N_SAMPLES, 1, 128, 256, device=device)
        for i in range(N_STEPS - 1):
            t_curr = step_idx[i].expand(N_SAMPLES)
            t_next = step_idx[i + 1].expand(N_SAMPLES)
            eps_pred = denoiser(xt, t_curr)

            alpha_c = diffusion.alphas_cumprod[t_curr]
            alpha_n = diffusion.alphas_cumprod[t_next]
            while alpha_c.dim() < xt.dim():
                alpha_c = alpha_c.unsqueeze(-1)
                alpha_n = alpha_n.unsqueeze(-1)

            x0_pred = (xt - torch.sqrt(1 - alpha_c) * eps_pred) / torch.sqrt(alpha_c)
            x0_pred = x0_pred.clamp(-1, 1)

            noise_scale = torch.sqrt(1 - alpha_n)
            x0_scale = torch.sqrt(alpha_n)
            xt = x0_scale * x0_pred + noise_scale * eps_pred
        x0 = xt.clamp(-1, 1)

    print(f"[INFO] Samples: {tuple(x0.shape)}  range=[{x0.min():.2f}, {x0.max():.2f}]")

    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    for idx in range(N_SAMPLES):
        mel = x0[idx, 0].cpu().numpy()
        mel_log = denormalize(mel)
        ax = axes[idx // 2][idx % 2]
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                       vmin=-1, vmax=1)
        ax.set_title(f"sample {idx}  (log-mel {mel_log.min():.1f}..{mel_log.max():.1f})")
        ax.set_xlabel("time frames")
        ax.set_ylabel("mel bins")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.tight_layout()
    png_path = os.path.join(OUT, "ddim_gate.png")
    fig.savefig(png_path, dpi=120)
    print(f"[INFO] PNG salvato: {png_path}")

    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    for idx in [0, 1, 2, 3, 4]:
        mel_t = torch.from_numpy(denormalize(x0[idx, 0].cpu().numpy()))
        audio = mel_to_audio(mel_t)
        wav_path = os.path.join(OUT, "audio", f"sample_{idx}.wav")
        torchaudio.save(wav_path, audio.cpu().reshape(1, -1), 16000)
        print(f"[INFO] WAV salvato: {wav_path}")


if __name__ == "__main__":
    main()
