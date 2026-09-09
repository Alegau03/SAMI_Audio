"""
SAMI-Audio — Training entry points
==================================
Covers toy model training (SAMI or β-VAE on synthetic sinusoids), the β-VAE
baseline on NSynth mel-spectrograms, and the two-stage SAMI pipeline
(unconditional denoiser pre-training, then the encoder with frozen denoiser).

Usage:
    .venv/bin/python train.py --mode vae                     # β-VAE toy
    .venv/bin/python train.py --mode vae-nsynth              # β-VAE NSynth
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.encoder import ToyEncoder
from models.encoder import MelEncoder
from models.unet import ToyUNet, MelUNet
from models.losses.diffusion import DiffusionSchedule
from models.sami import SAMI
from models.vae import VaeDecoder, BetaVAE, MelDecoder


# ---------------------------------------------------------------------------
# Toy dataset: synthetic sinusoids
# ---------------------------------------------------------------------------

def generate_toy_dataset(
    n_freqs: int = 10,
    n_amps: int = 5,
    n_per_class: int = 1000,
    signal_length: int = 256,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate synthetic 1D sinusoids with known frequency and amplitude labels.

    Each sample: y(t) = A · sin(2π · f · t / L) + N(0, noise_std)

    Returns
    -------
    signals : (N, 1, L) tensor in approximate range [-1, 1].
    freq_labels : (N,) integer frequency class index [0, n_freqs).
    amp_labels : (N,) integer amplitude class index [0, n_amps).
    """
    rng = np.random.default_rng(seed)
    L = signal_length
    t = np.arange(L)

    frequencies = np.linspace(1.0, 20.0, n_freqs)
    amplitudes = np.linspace(0.2, 1.0, n_amps)

    signals_list: list[np.ndarray] = []
    freq_label_list: list[int] = []
    amp_label_list: list[int] = []

    for fi, f in enumerate(frequencies):
        for ai, A in enumerate(amplitudes):
            for _ in range(n_per_class):
                phase = rng.uniform(0, 2 * np.pi)
                y = A * np.sin(2.0 * np.pi * f * t / L + phase)
                y += rng.normal(0, noise_std, size=L)
                signals_list.append(y)
                freq_label_list.append(fi)
                amp_label_list.append(ai)

    signals = np.stack(signals_list, axis=0)
    signals = signals[:, np.newaxis, :].astype(np.float32)

    return torch.from_numpy(signals), np.array(freq_label_list), np.array(amp_label_list)


class SinusoidDataset(Dataset):
    """Simple Dataset wrapper for toy sinusoid data."""

    def __init__(
        self,
        signals: torch.Tensor,
        freq_labels: np.ndarray,
        amp_labels: np.ndarray,
    ) -> None:
        self.signals = signals
        self.freq_labels = freq_labels
        self.amp_labels = amp_labels

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self.signals[idx], self.freq_labels[idx], self.amp_labels[idx]


# ---------------------------------------------------------------------------
# Disks dataset: 2D synthetic images (3 factors)
# ---------------------------------------------------------------------------

def generate_disks_dataset(
    n_samples: int = 10000,
    resolution: int = 32,
    disk_radius: int = 5,
    disk_intensity: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D images with a white disk on gray background.

    3 independent factors:
      - cx : disk center x ∈ [0, 1]
      - cy : disk center y ∈ [0, 1]
      - I_bg : background intensity ∈ [0.2, 0.8]

    Images are (N, 1, R, R) float32 in [-1, 1].

    Returns
    -------
    images : (N, 1, R, R) tensor.
    cx_labels, cy_labels, I_bg_labels : (N,) float arrays.
    """
    rng = np.random.default_rng(seed)

    cx = rng.uniform(0, 1, n_samples)
    cy = rng.uniform(0, 1, n_samples)
    I_bg = rng.uniform(0.2, 0.8, n_samples)

    yy, xx = np.mgrid[0:resolution, 0:resolution]
    images = np.zeros((n_samples, 1, resolution, resolution), dtype=np.float32)

    for i in range(n_samples):
        img = np.full((resolution, resolution), I_bg[i], dtype=np.float32)
        px = int(cx[i] * resolution)
        py = int(cy[i] * resolution)
        px = max(0, min(resolution - 1, px))
        py = max(0, min(resolution - 1, py))
        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        img[dist < disk_radius] = disk_intensity
        images[i, 0] = img

    # Normalize to [-1, 1] globally
    images = 2.0 * images - 1.0

    return torch.from_numpy(images), cx, cy, I_bg


class DisksDataset(Dataset):
    """Simple Dataset wrapper for disks data."""

    def __init__(
        self,
        images: torch.Tensor,
        cx_labels: np.ndarray,
        cy_labels: np.ndarray,
        I_bg_labels: np.ndarray,
    ) -> None:
        self.images = images
        self.cx_labels = cx_labels
        self.cy_labels = cy_labels
        self.I_bg_labels = I_bg_labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return (self.images[idx],
                self.cx_labels[idx], self.cy_labels[idx], self.I_bg_labels[idx])


# ---------------------------------------------------------------------------
# Denoiser pre-training (unconditional DDPM)
# ---------------------------------------------------------------------------

def pretrain_denoiser(config: dict) -> ToyUNet:
    """
    Pre-train the unconditional denoiser on synthetic sinusoids.

    Standard DDPM training: no encoder, no guidance, no KL.
    Returns the trained denoiser and saves a checkpoint.

    Returns
    -------
    denoiser : ToyUNet trained unconditionally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    print("[INFO]  Generating toy dataset...")
    signals, _, _ = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, np.zeros(len(signals)), np.zeros(len(signals)))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    denoiser = ToyUNet(
        in_channels=1,
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    )
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    denoiser.to(device)
    diffusion.to(device)

    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[INFO]  Denoiser parameters: {total_params:,}")

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["denoiser_epochs"] * len(loader)
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    denoiser.train()
    denoiser_epochs: int = config["denoiser_epochs"]
    T_diff: int = config["T"]

    for epoch in range(1, denoiser_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Denoiser Epoch {epoch:3d}/{denoiser_epochs}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)
            B = x.shape[0]

            t = torch.randint(0, T_diff, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(x)
            xt = diffusion.q_sample(x, t, eps)

            eps_pred = denoiser(xt, t)
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Denoiser Epoch {epoch:3d} | loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    ckpt_path = os.path.join(config["checkpoint_dir"], "denoiser_pretrained.pt")
    torch.save({
        "epoch": denoiser_epochs,
        "model_state_dict": denoiser.state_dict(),
        "config": config,
    }, ckpt_path)
    print(f"[INFO]  Denoiser saved: {ckpt_path}")

    return denoiser


# ---------------------------------------------------------------------------
# VAE training (β-VAE baseline)
# ---------------------------------------------------------------------------

def train_vae(config: dict) -> tuple[BetaVAE, dict]:
    """
    Train a β-VAE on synthetic sinusoids.

    Standard VAE: encoder → z → decoder → x_recon.
    Loss = MSE(x_recon, x) + β · KL(N(μ,σ²) || N(0,I)).

    Returns the trained model and a dict of training metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    print("[INFO]  Generating toy dataset...")
    signals, freq_labels, amp_labels = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, freq_labels, amp_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    signal_length = config["signal_length"]
    latent_dim = config["latent_dim"]

    encoder = ToyEncoder(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=config["base_channels"],
        signal_length=signal_length,
    )
    decoder = VaeDecoder(latent_dim=latent_dim, signal_length=signal_length)
    vae = BetaVAE(encoder, decoder, beta=config["beta"])
    vae.to(device)

    start_epoch = 1
    if config.get("resume"):
        print(f"[INFO]  Resuming from: {config['resume']}")
        ckpt = torch.load(config["resume"], map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO]  Resuming at epoch {start_epoch}")

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"[INFO]  Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(loader)
    )

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-toy", config=config)
        print("[INFO]  wandb logging enabled")

    beta_val = config["beta"]
    warmup_epochs = config.get("vae_warmup", 5)
    print(f"[INFO]  β-VAE with β = {beta_val} (fixed)")
    if warmup_epochs > 0:
        ae_epochs = max(1, warmup_epochs // 3)
        ramp_epochs = warmup_epochs - ae_epochs
        print(f"[INFO]  Warmup: β=0 for {ae_epochs} epochs, then ramp to {beta_val} over {ramp_epochs} epochs")

    vae.train()
    stats: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}
    global_step = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        if warmup_epochs > 0:
            ae_epochs = max(1, warmup_epochs // 3)
            if epoch <= ae_epochs:
                vae.beta = 0.0
            elif epoch <= warmup_epochs:
                progress = (epoch - ae_epochs) / (warmup_epochs - ae_epochs)
                vae.beta = beta_val * progress
            else:
                vae.beta = beta_val
        else:
            vae.beta = beta_val
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)

            optimizer.zero_grad()
            loss, recon, kl = vae(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            global_step += 1

            stats["loss"].append(loss.item())
            stats["recon"].append(recon.item())
            stats["kl"].append(kl.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
            )

            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon": recon.item(),
                    "train/kl": kl.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(loader)
        avg_recon = epoch_recon / len(loader)
        avg_kl = epoch_kl / len(loader)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}  recon={avg_recon:.4f}  KL={avg_kl:.4f}  β={vae.beta:.4g}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"toy_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "stats": stats,
                "mode": "vae",
            }, ckpt_path)
            print(f"  [CKPT] Saved: {ckpt_path}")

    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": vae.state_dict(),
        "config": config,
        "stats": stats,
        "mode": "vae",
    }, final_path)
    print(f"[INFO]  Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()

    return vae, stats


# ---------------------------------------------------------------------------
# NSynth β-VAE training (baseline)
# ---------------------------------------------------------------------------

def train_vae_nsynth(config: dict) -> tuple[BetaVAE, dict]:
    """
    Train a β-VAE on NSynth mel-spectrograms.

    Standard VAE: MelEncoder → z → MelDecoder → x_recon.
    Loss = MSE(x_recon, x) + β · KL(N(μ,σ²) || N(0,I)).

    Supports --resume to continue training from a saved checkpoint.
    Returns the trained model and a dict of training metrics.
    """
    from data.nsynth import NSynthDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    print("[INFO]  Loading NSynth dataset...")
    dataset = NSynthDataset(root=config.get("nsynth_root", "data/nsynth-train"))
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=config.get("num_workers", 2), pin_memory=True,
    )
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    encoder = MelEncoder(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config.get("base_channels_vae", 64),
        input_size=(128, 256),
    )
    decoder = MelDecoder(
        latent_dim=config["latent_dim"],
        base_channels=config.get("decoder_base_channels", 128),
    )
    vae = BetaVAE(encoder, decoder, beta=config["beta"])
    vae.to(device)

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"[INFO]  Total parameters: {total_params:,}")

    start_epoch = 1
    optimizer_state = None
    wandb_id = None
    stats: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}

    resume_path = config.get("resume")
    if resume_path and os.path.isfile(resume_path):
        print(f"[INFO]  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_state = ckpt.get("optimizer_state_dict")
        stats = ckpt.get("stats", stats)
        wandb_id = ckpt.get("wandb_id")
        print(f"[INFO]  Resuming at epoch {start_epoch}/{config['epochs']}")
        if wandb_id:
            print(f"[INFO]  Wandb run: {wandb_id}")

    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    scheduler_steps = max(1, (config["epochs"] - start_epoch + 1) * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_steps)

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project="sami-audio-nsynth",
            config=config,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )
        if wandb_id is None:
            wandb_id = wandb.run.id
        print("[INFO]  wandb logging enabled")

    beta_val = config["beta"]
    warmup_epochs = config.get("vae_warmup", 0)
    if warmup_epochs > 0:
        ae_epochs = max(1, warmup_epochs // 3)
        print(f"[INFO]  β-VAE NSynth with β = {beta_val}, D_latent = {config['latent_dim']}")
        print(f"[INFO]  Warmup: β=0 for {ae_epochs} epochs, then ramp to {beta_val} over {warmup_epochs - ae_epochs} epochs")
    else:
        print(f"[INFO]  β-VAE NSynth with β = {beta_val} (fixed), D_latent = {config['latent_dim']}")

    vae.train()
    global_step = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        if warmup_epochs > 0:
            ae_epochs = max(1, warmup_epochs // 3)
            if epoch <= ae_epochs:
                vae.beta = 0.0
            elif epoch <= warmup_epochs:
                progress = (epoch - ae_epochs) / max(1, warmup_epochs - ae_epochs)
                vae.beta = beta_val * progress
            else:
                vae.beta = beta_val
        else:
            vae.beta = beta_val

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for x, _ in pbar:
            x = x.to(device)

            optimizer.zero_grad()
            loss, recon, kl = vae(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            global_step += 1

            stats["loss"].append(loss.item())
            stats["recon"].append(recon.item())
            stats["kl"].append(kl.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
            )

            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon": recon.item(),
                    "train/kl": kl.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(loader)
        avg_recon = epoch_recon / len(loader)
        avg_kl = epoch_kl / len(loader)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}  recon={avg_recon:.4f}  KL={avg_kl:.4f}  β={vae.beta:.4g}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"nsynth_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "stats": stats,
                "mode": "vae-nsynth",
                "wandb_id": wandb_id,
            }, ckpt_path)
            print(f"  [CKPT] Saved: {ckpt_path}")

    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": vae.state_dict(),
        "config": config,
        "stats": stats,
        "mode": "vae-nsynth",
    }, final_path)
    print(f"[INFO]  Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()

    return vae, stats


# ---------------------------------------------------------------------------
# Denoiser pre-training on NSynth (2D, unconditional DDPM)
# ---------------------------------------------------------------------------

def pretrain_denoiser_2d(config: dict) -> MelUNet:
    """
    Pre-train the unconditional MelUNet denoiser on NSynth mel-spectrograms.

    Standard DDPM training: no encoder, no guidance, no KL.
    """
    from data.nsynth import CachedMelDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    cache = config.get("mel_cache", "data/mel_cache.npy")
    meta_cache = config.get("mel_meta", "data/mel_meta.pkl")
    use_cache = os.path.isfile(cache) and os.path.isfile(meta_cache)
    if use_cache:
        dataset = CachedMelDataset(cache=cache, meta_path=meta_cache)
        print(f"[INFO]  Using cached mels: {cache}")
    else:
        from data.nsynth import NSynthDataset
        dataset = NSynthDataset(root=config.get("nsynth_root", "data/nsynth-train"))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=config.get("num_workers", 2), pin_memory=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    denoiser = MelUNet(in_channels=1, base_channels=config.get("denoiser_channels", 128),
                       channel_mult=(1, 1, 2, 2), time_dim=128)
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    denoiser.to(device); diffusion.to(device)

    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[INFO]  Denoiser parameters: {total_params:,}")

    start_epoch = 1; optimizer_state = None; wandb_id = None
    resume_path = config.get("resume")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_state = ckpt.get("optimizer_state_dict")
        wandb_id = ckpt.get("wandb_id")
        print(f"[INFO]  Resuming at epoch {start_epoch}/{config['epochs']}")

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config["lr"])
    if optimizer_state: optimizer.load_state_dict(optimizer_state)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, (config["epochs"] - start_epoch + 1) * len(loader)))

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-nsynth", config=config,
                   id=wandb_id, resume="allow" if wandb_id else None)
        if wandb_id is None: wandb_id = wandb.run.id

    T_diff: int = config["T"]
    denoiser.train()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Denoiser E {epoch:3d}/{config['epochs']}", leave=False)
        for x, _ in pbar:
            x = x.to(device); B = x.shape[0]
            t = torch.randint(0, T_diff, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(x)
            xt = diffusion.q_sample(x, t, eps)
            eps_pred = denoiser(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), config["grad_clip"])
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Denoiser Epoch {epoch:3d} | loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"denoiser_epoch_{epoch:04d}.pt")
            torch.save({"epoch": epoch, "model_state_dict": denoiser.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "config": config,
                        "wandb_id": wandb_id, "mode": "sami-deno"}, ckpt_path)
            print(f"  [CKPT] {ckpt_path}")

    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({"epoch": config["epochs"], "model_state_dict": denoiser.state_dict(),
                "config": config, "mode": "sami-deno"}, final_path)
    print(f"[INFO]  Denoiser saved: {final_path}")
    if use_wandb: wandb.finish()
    return denoiser


# ---------------------------------------------------------------------------
# SAMI encoder training on NSynth (denoiser frozen)
# ---------------------------------------------------------------------------

def _probe_r2(encoder, cache_arr, metas, device, n: int = 2000, seed: int = 42) -> dict:
    """
    Linear probe R² of mu onto pitch and instrument_family on a fixed subset.

    Used for early stopping. Uses an internal train/test split
    (fit on half, score on the held-out half) AND Ridge regularization:
    with D=128 latent dims and near-constant mu (collapsed posterior), a
    plain LinearRegression inflates R² via huge coefficients (overfitting).
    Verified: LR gave R²=0.49 on a collapsed encoder, Ridge(α=1) gives 0.005.

    Also computes a noise control (Ridge on pure Gaussian noise with the
    same split): r2_pitch is meaningful only if clearly above r2_control.

    Returns a dict with r2_pitch, r2_family, r2_control, kl_per_dim stats
    and the latent matrix.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cache_arr), min(n, len(cache_arr)), replace=False)
    mels = torch.from_numpy(np.asarray(cache_arr[idx])).to(device)

    mu_all = []
    sigma2_all = []
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 256):
            mu, sigma2 = encoder(mels[b:b + 256])
            mu_all.append(mu.cpu().numpy())
            sigma2_all.append(sigma2.cpu().numpy())
    if was_training:
        encoder.train()
    mu = np.concatenate(mu_all, axis=0)
    sigma2 = np.concatenate(sigma2_all, axis=0)

    # KL per-dimensione (senza floor): il discriminante tra entanglement
    # (informazione spalmata su molte dim) e disentanglement (poche dim
    # cariche, il resto a prior). Stesso totale, significati opposti.
    kl_per_dim = 0.5 * np.mean(mu ** 2 + sigma2 - np.log(sigma2) - 1.0, axis=0)
    n_dim_gt_1 = int(np.sum(kl_per_dim > 1.0))
    n_dim_gt_0_5 = int(np.sum(kl_per_dim > 0.5))
    n_dim_lt_0_1 = int(np.sum(kl_per_dim < 0.1))
    top5 = [round(v, 3) for v in np.sort(kl_per_dim)[::-1][:5]]

    pitch = np.array([metas[i]["pitch"] for i in idx], dtype=float)
    family_str = np.array([metas[i]["instrument_family"] for i in idx])
    family = LabelEncoder().fit_transform(family_str)

    def split_score(X, y, alpha=1.0):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.5, random_state=seed)
        return float(Ridge(alpha=alpha).fit(X_tr, y_tr).score(X_te, y_te))

    r2_pitch = split_score(mu, pitch)
    r2_family = split_score(mu, family)
    r2_control = split_score(rng.standard_normal(mu.shape), pitch)
    return {"r2_pitch": r2_pitch, "r2_family": r2_family,
            "r2_control": r2_control, "mu": mu,
            "kl_per_dim": kl_per_dim, "n_dim_gt_1": n_dim_gt_1,
            "n_dim_gt_0_5": n_dim_gt_0_5, "n_dim_lt_0_1": n_dim_lt_0_1,
            "top5_kl": top5}


def train_sami_nsynth(config: dict) -> tuple[SAMI, dict]:
    """
    Train the SAMI encoder on NSynth with frozen denoiser.

    SAMI.forward handles guidance g_t via autograd.grad with create_graph=True.
    Uses CachedMelDataset for fast I/O, oversample_t for high-t bias.
    Optional early stopping on R²(mu→pitch) and posterior-collapse guard on L_z.
    """
    from data.nsynth import CachedMelDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    cache = config.get("mel_cache", "data/mel_cache.npy")
    meta_cache = config.get("mel_meta", "data/mel_meta.pkl")
    use_cache = os.path.isfile(cache) and os.path.isfile(meta_cache)
    if use_cache:
        dataset = CachedMelDataset(cache=cache, meta_path=meta_cache)
        print(f"[INFO]  Using cached mels: {cache}")
    else:
        from data.nsynth import NSynthDataset
        dataset = NSynthDataset(root=config.get("nsynth_root", "data/nsynth-train"))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    encoder = MelEncoder(in_channels=1, latent_dim=config["latent_dim"],
                         base_channels=config.get("base_channels_vae", 64),
                         input_size=(128, 256))

    denoiser = MelUNet(in_channels=1, base_channels=config.get("denoiser_channels", 128),
                       channel_mult=(1, 1, 2, 2), time_dim=128)
    d_ckpt = torch.load(config["denoiser_checkpoint"], map_location=device, weights_only=False)
    denoiser.load_state_dict(d_ckpt["model_state_dict"])
    for p in denoiser.parameters():
        p.requires_grad = False
    print(f"[INFO]  Denoiser frozen ({sum(p.numel() for p in denoiser.parameters()):,} params)")

    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    diffusion.to(device)
    sami = SAMI(encoder, denoiser, diffusion, beta=config["beta"],
                 frozen_denoiser=True,
                 oversample_t=config.get("oversample_t", False),
                 amp_denoiser=config.get("amp", False),
                 free_bits=config.get("free_bits", 0.0))
    sami.to(device)

    trainable = sum(p.numel() for p in sami.parameters() if p.requires_grad)
    print(f"[INFO]  Trainable params: {trainable:,}")

    start_epoch = 1; optimizer_state = None; wandb_id = None
    global_step = 0; skip_batches = 0
    stats: dict[str, list[float]] = {"loss": [], "L_x": [], "L_z": []}
    resume_path = config.get("resume")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sami.load_state_dict(ckpt["model_state_dict"])
        # Se il checkpoint è di metà epoca (step), riparte dallo stesso epoch
        # saltando i batch già fatti; se è di fine epoca, va all'epoch+1.
        ckpt_epoch = ckpt.get("epoch", 0)
        mid_epoch = ckpt.get("mid_epoch", False)
        if mid_epoch:
            start_epoch = ckpt_epoch
            global_step = ckpt.get("global_step", 0)
            skip_batches = global_step % max(1, len(loader))
            print(f"[INFO]  Resuming MID-EPOCH {start_epoch} at step {global_step}, "
                  f"skipping {skip_batches} batches")
        else:
            start_epoch = ckpt_epoch + 1
            global_step = ckpt.get("global_step", 0)
            print(f"[INFO]  Resuming at epoch {start_epoch}/{config['epochs']}")
        optimizer_state = ckpt.get("optimizer_state_dict")
        wandb_id = ckpt.get("wandb_id")
        stats = ckpt.get("stats", stats)

    optimizer = torch.optim.Adam(
        [p for p in sami.parameters() if p.requires_grad], lr=config["lr"])
    if optimizer_state: optimizer.load_state_dict(optimizer_state)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, (config["epochs"] - start_epoch + 1) * len(loader)))
    if resume_path and os.path.isfile(resume_path):
        sched_state = torch.load(resume_path, map_location=device,
                                 weights_only=False).get("scheduler_state_dict")
        if sched_state:
            scheduler.load_state_dict(sched_state)

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-nsynth", config=config,
                   id=wandb_id, resume="allow" if wandb_id else None)
        if wandb_id is None: wandb_id = wandb.run.id

    amp_enabled = config.get("amp", False)
    probe_every = config.get("probe_every", 5)
    probe_size = config.get("probe_size", 1000)
    collapse_thresh = config.get("collapse_thresh", 0.05)
    early_stop_patience = config.get("early_stop_patience", 0)  # 0 = disabled
    r2_tol = config.get("r2_tol", 0.005)
    warmup_epochs = config.get("warmup_epochs", 0)
    ramp_epochs = config.get("ramp_epochs", 0)
    beta_target = config["beta"]
    print(f"[INFO]  SAMI NSynth β_target={beta_target}, D={config['latent_dim']}, T={config['T']}, "
          f"amp={amp_enabled}, probe_every={probe_every}, early_stop_patience={early_stop_patience}, "
          f"warmup_epochs={warmup_epochs}, ramp_epochs={ramp_epochs}")

    sami.train()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    probe_history: list[tuple[float, float]] = []
    best_r2_p = -1.0
    best_r2_f = -1.0

    # KL warm-up: beta = 0 per warmup_epochs, poi rampa lineare verso il
    # beta target su ramp_epochs, poi costante. Il beta per epoca si
    # ricalcola da `epoch` → riproducibile tra job con resume.
    def beta_at(epoch: int) -> float:
        if epoch <= warmup_epochs:
            return 0.0
        if ramp_epochs > 0 and epoch <= warmup_epochs + ramp_epochs:
            frac = (epoch - warmup_epochs) / ramp_epochs
            return beta_target * frac
        return beta_target

    final_epoch = start_epoch - 1
    last_x = None
    for epoch in range(start_epoch, config["epochs"] + 1):
        final_epoch = epoch
        beta_val = beta_at(epoch)
        sami.beta = beta_val
        epoch_loss = 0.0; epoch_lx = 0.0; epoch_lz = 0.0; epoch_lz_raw = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for x, _ in pbar:
            if skip_batches > 0:
                # Resume da metà epoca: salta i batch già processati
                skip_batches -= 1
                continue
            x = x.to(device)
            last_x = x
            optimizer.zero_grad()
            # AMP: solo il denoiser congelato gira in bf16 (dentro SAMI.forward,
            # context no_grad). Guidance e doppio-backward restano in fp32
            # per evitare NaN da autocast su autograd.grad(create_graph=True).
            loss, L_x, L_z, z, L_z_raw = sami(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sami.parameters(), config["grad_clip"])
            optimizer.step(); scheduler.step()

            epoch_loss += loss.item(); epoch_lx += L_x.item(); epoch_lz += L_z.item()
            epoch_lz_raw += L_z_raw.item()
            global_step += 1
            stats["loss"].append(loss.item())
            stats["L_x"].append(L_x.item()); stats["L_z"].append(L_z.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", Lx=f"{L_x.item():.4f}", Lz=f"{L_z.item():.4f}")

            # Checkpoint PER-STEP: garantisce che un job killato a metà epoca
            # non perda tutto — il resume riparte dallo step più recente,
            # non dall'inizio dell'epoca (epoca ~31 min > limite 29 min su
            # alcuni nodi: senza questo, il training non avanzerebbe MAI).
            ckpt_every_steps = config.get("ckpt_every_steps", 0)
            if ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                step_path = os.path.join(config["checkpoint_dir"],
                                         f"sami_step_{global_step:08d}.pt")
                torch.save({"epoch": epoch, "global_step": global_step,
                            "mid_epoch": True,
                            "model_state_dict": sami.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config, "stats": stats,
                            "mode": "sami-nsynth", "wandb_id": wandb_id}, step_path)
                print(f"  [CKPT-STEP {global_step}] {step_path}")

            if global_step == 1 and config.get("print_t_hist", False):
                t_vals = sami._last_t.float()
                bins = torch.histc(t_vals, bins=10, min=0, max=config["T"])
                frac_high = (t_vals > config["T"] * 0.5).float().mean().item()
                print(f"[T-HIST] t del primo batch: {[int(v) for v in bins.tolist()]}  "
                      f"(10 bin su {config['T']})  frac_t>0.5T={frac_high:.2f}  mean_t={t_vals.mean():.0f}")

            if use_wandb and global_step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/L_x": L_x.item(),
                           "train/L_z": L_z.item(), "train/beta": beta_val,
                           "train/lr": scheduler.get_last_lr()[0], "step": global_step})

        avg_loss = epoch_loss / len(loader); avg_lx = epoch_lx / len(loader); avg_lz = epoch_lz / len(loader)
        avg_lz_raw = epoch_lz_raw / len(loader)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}  L_x={avg_lx:.4f}  L_z={avg_lz:.4f}  "
              f"L_z_raw={avg_lz_raw:.4f}  β={beta_val}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"sami_epoch_{epoch:04d}.pt")
            torch.save({"epoch": epoch, "global_step": global_step, "mid_epoch": False,
                        "model_state_dict": sami.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config, "stats": stats, "mode": "sami-nsynth",
                        "wandb_id": wandb_id}, ckpt_path)
            print(f"  [CKPT] {ckpt_path}")

        if use_cache and probe_every > 0 and epoch % probe_every == 0:
            probe = _probe_r2(encoder, dataset.arr, dataset.metas, device, n=probe_size)
            print(f"  [PROBE] R²(mu→pitch)={probe['r2_pitch']:.4f}  "
                  f"R²(mu→family)={probe['r2_family']:.4f}  "
                  f"R²(control rumore)={probe['r2_control']:.4f}  "
                  f"KL/dim: >1nat={probe['n_dim_gt_1']}, >0.5nat={probe['n_dim_gt_0_5']}, "
                  f"<0.1nat={probe['n_dim_lt_0_1']}, top5={probe['top5_kl']}")

            if last_x is not None:
                # Appaiato: stessi t/ε dell'ultimo forward (che è su last_x).
                # uncond_lx usa gli stessi t/eps → il confronto è pulito.
                with torch.no_grad():
                    uncond = sami.uncond_lx(
                        last_x, sami._last_t.clone(), sami._last_eps.clone()).item()
                print(f"  [GUIDANCE] L_x guidato={avg_lx:.4f}  L_x senza guidance={uncond:.4f}  "
                      f"(guidato < uncond → la guidance aiuta)")
            else:
                uncond = float("nan")
            if use_wandb:
                wandb.log({"eval/r2_pitch": probe["r2_pitch"],
                           "eval/r2_family": probe["r2_family"],
                           "eval/lx_guided": avg_lx, "eval/lx_uncond": uncond,
                           "epoch": epoch})
            probe_history.append((probe["r2_pitch"], probe["r2_family"]))
            best_r2_p = max(best_r2_p, probe["r2_pitch"])
            best_r2_f = max(best_r2_f, probe["r2_family"])
            # Early stop sul MARGINE sopra il control (R²_pitch − R²_control):
            # con la metrica Ridge il valore assoluto è ininterpretabile senza
            # il riferimento del rumore. Si ferma solo se il margine è piatto
            # e sotto il best.
            if early_stop_patience > 0 and len(probe_history) >= early_stop_patience:
                margin = probe["r2_pitch"] - probe["r2_control"]
                best_margin = max(best_r2_p - probe["r2_control"] for _ in [0])
                window = probe_history[-early_stop_patience:]
                r2p_flat = max(p for p, _ in window) - min(p for p, _ in window) < r2_tol
                r2f_flat = max(f for _, f in window) - min(f for _, f in window) < r2_tol
                pitch_stalled = best_margin - margin > r2_tol
                if r2p_flat and r2f_flat and pitch_stalled and margin <= 0.05:
                    print(f"  [EARLY STOP] margine R²(pitch)−control={margin:.4f} "
                          f"stabile per {early_stop_patience} probe")
                    break

        # Guardia collasso sulla KL GREZZA (senza floor): con free bits attive
        # L_z post-floor ≈ 0 per costruzione anche da sano → la guardia sul
        # post-floor fermerebbe sempre il training (falso allarme).
        if avg_lz_raw < collapse_thresh and beta_val > 0:
            print(f"  [COLLAPSE] L_z_raw={avg_lz_raw:.4f} < {collapse_thresh} con β={beta_val:.1e}: "
                  f"posterior collassato, fermo training")
            break

    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({"epoch": final_epoch, "model_state_dict": sami.state_dict(),
                "config": config, "stats": stats, "mode": "sami-nsynth"}, final_path)
    print(f"[INFO]  Final model saved: {final_path}  (epoch {final_epoch})")
    if use_wandb: wandb.finish()
    return sami, stats


# ---------------------------------------------------------------------------
# Disks training (2D synthetic, 3 factors)
# ---------------------------------------------------------------------------

def train_disks(config: dict) -> tuple[SAMI, dict]:
    """
    Train SAMI on 2D synthetic disks to validate the full 2D stack.

    Uses MelUNet + MelEncoder on 32×32 images with 3 factors (cx, cy, I_bg).
    Joint training, no frozen denoiser (small model, fast convergence).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    resolution = config.get("resolution", 32)
    print(f"[INFO]  Generating disks dataset ({resolution}×{resolution})...")
    images, cx_labels, cy_labels, I_bg_labels = generate_disks_dataset(
        n_samples=config.get("n_disks", 10000),
        resolution=resolution,
        disk_radius=config.get("disk_radius", 5),
    )
    dataset = DisksDataset(images, cx_labels, cy_labels, I_bg_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    encoder = MelEncoder(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config.get("base_channels_vae", 64),
        channel_mult=config.get("encoder_mult", (1, 1)),
        input_size=(resolution, resolution),
    )
    denoiser = MelUNet(
        in_channels=1,
        base_channels=config.get("denoiser_channels", 64),
        channel_mult=config.get("denoiser_mult", (1, 1, 2)),
        time_dim=config.get("time_dim", 128),
    )
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
    sami = SAMI(encoder, denoiser, diffusion, beta=config["beta_start"],
                 oversample_t=False)
    sami.to(device)
    diffusion.to(device)

    total_params = sum(p.numel() for p in sami.parameters())
    print(f"[INFO]  Total parameters: {total_params:,}")

    start_epoch = 1; optimizer_state = None; wandb_id = None
    if config.get("resume"):
        ckpt = torch.load(config["resume"], map_location=device, weights_only=False)
        sami.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_state = ckpt.get("optimizer_state_dict")
        wandb_id = ckpt.get("wandb_id")

    optimizer = torch.optim.Adam(sami.parameters(), lr=config["lr"])
    if optimizer_state: optimizer.load_state_dict(optimizer_state)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, (config["epochs"] - start_epoch + 1) * len(loader)))

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-disks", config=config,
                   id=wandb_id, resume="allow" if wandb_id else None)
        if wandb_id is None: wandb_id = wandb.run.id

    beta_val = config.get("beta_fixed", config["beta_start"])
    sami.beta = beta_val
    print(f"[INFO]  SAMI Disks: β={beta_val}")

    sami.train(); global_step = 0
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    stats = {"loss": [], "L_x": [], "L_z": []}
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_loss = 0.0; epoch_lx = 0.0; epoch_lz = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for img, cx, cy, ibg in pbar:
            img = img.to(device)
            optimizer.zero_grad()
            loss, L_x, L_z, z, L_z_raw = sami(img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sami.parameters(), config["grad_clip"])
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); epoch_lx += L_x.item(); epoch_lz += L_z.item()
            global_step += 1
            stats["loss"].append(loss.item())
            stats["L_x"].append(L_x.item()); stats["L_z"].append(L_z.item())

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d} | loss={avg:.4f}  L_x={epoch_lx/len(loader):.4f}  L_z={epoch_lz/len(loader):.4f}  β={beta_val}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            cfg_save = dict(config)
            cfg_save["input_size"] = (resolution, resolution)
            cfg_save["encoder_mult"] = config.get("encoder_mult", (1, 1))
            ckpt_path = os.path.join(config["checkpoint_dir"], f"disks_epoch_{epoch:04d}.pt")
            torch.save({"epoch": epoch, "model_state_dict": sami.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg_save, "wandb_id": wandb_id,
                        "mode": "sami-disks"}, ckpt_path)
            print(f"  [CKPT] {ckpt_path}")

    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    config_to_save = dict(config)
    config_to_save["input_size"] = (resolution, resolution)
    config_to_save["encoder_mult"] = config.get("encoder_mult", (1, 1))
    torch.save({"epoch": config["epochs"], "model_state_dict": sami.state_dict(),
                "config": config_to_save, "stats": stats, "mode": "sami-disks"}, final_path)
    print(f"[INFO]  Saved: {final_path}")
    if use_wandb: wandb.finish()
    return sami, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_toy(config: dict) -> tuple[SAMI, dict]:
    """
    Train the SAMI toy model on synthetic sinusoids.

    Returns the trained model and a dict of training metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    # ---- Data ---------------------------------------------------------------
    print("[INFO]  Generating toy dataset...")
    signals, freq_labels, amp_labels = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, freq_labels, amp_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"[INFO]  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # ---- Models -------------------------------------------------------------
    signal_length = config["signal_length"]
    latent_dim = config["latent_dim"]

    encoder = ToyEncoder(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=config["base_channels"],
        signal_length=signal_length,
    )
    denoiser = ToyUNet(
        in_channels=1,
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    )
    diffusion = DiffusionSchedule(T=config["T"], s=config["s"])

    pretrained_path = config.get("pretrained_denoiser")
    if pretrained_path:
        print(f"[INFO]  Loading pretrained denoiser: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt["model_state_dict"])
        for p in denoiser.parameters():
            p.requires_grad = False
        frozen_params = sum(p.numel() for p in denoiser.parameters())
        print(f"[INFO]  Denoiser frozen ({frozen_params:,} params)")

    sami = SAMI(encoder, denoiser, diffusion, beta=config["beta_start"])

    encoder.to(device)
    denoiser.to(device)
    diffusion.to(device)
    sami.to(device)

    total_params = sum(p.numel() for p in sami.parameters())
    print(f"[INFO]  Total parameters: {total_params:,}")

    # ---- Optimizer ----------------------------------------------------------
    optimizer = torch.optim.Adam(sami.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(loader)
    )

    # ---- W&B ------------------------------------------------------------------
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="sami-audio-toy", config=config)
        print("[INFO]  wandb logging enabled")

    # ---- KL annealing / fixed beta ------------------------------------------------
    beta_fixed = config.get("beta_fixed")
    if beta_fixed is not None:
        sami.beta = beta_fixed
        print(f"[INFO]  KL weight: β = {beta_fixed} (fixed, no annealing)")
    else:
        beta_start = config["beta_start"]
        beta_end = config["beta_end"]
        beta_warmup = config["beta_warmup"]
        print(f"[INFO]  KL annealing: {beta_start:.1e} → {beta_end:.1e} over {beta_warmup} epochs (exponential)")

    # ---- Training loop ------------------------------------------------------
    sami.train()
    stats: dict[str, list[float]] = {"loss": [], "L_x": [], "L_z": []}
    global_step = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        if beta_fixed is None:
            if epoch <= beta_warmup:
                beta = beta_start * (beta_end / beta_start) ** (epoch / beta_warmup)
            else:
                beta = beta_end
            sami.beta = beta
        else:
            beta = beta_fixed

        epoch_loss = 0.0
        epoch_lx = 0.0
        epoch_lz = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{config['epochs']}", leave=False)
        for x, _, _ in pbar:
            x = x.to(device)

            optimizer.zero_grad()
            loss, L_x, L_z, z, L_z_raw = sami(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sami.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_lx += L_x.item()
            epoch_lz += L_z.item()
            global_step += 1

            stats["loss"].append(loss.item())
            stats["L_x"].append(L_x.item())
            stats["L_z"].append(L_z.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}", Lx=f"{L_x.item():.4f}", Lz=f"{L_z.item():.4f}")

            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/L_x": L_x.item(),
                    "train/L_z": L_z.item(),
                    "train/beta": beta,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(loader)
        avg_lx = epoch_lx / len(loader)
        avg_lz = epoch_lz / len(loader)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}  L_x={avg_lx:.4f}  L_z={avg_lz:.4f}  β={beta:.2e}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"toy_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": sami.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "stats": stats,
            }, ckpt_path)
            print(f"  [CKPT] Saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(config["checkpoint_dir"], "model_final.pt")
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": sami.state_dict(),
        "config": config,
        "stats": stats,
    }, final_path)
    print(f"[INFO]  Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()

    return sami, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAMI-Audio — Toy Model Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta-start", type=float, default=1e-6, help="Initial KL weight")
    parser.add_argument("--beta-end", type=float, default=1e-6, help="Target KL weight after warmup")
    parser.add_argument("--beta-warmup", type=int, default=10, help="Epochs of exponential KL annealing")
    parser.add_argument("--n-per-class", type=int, default=500, help="Samples per class (default 500)")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimension (keep 2 for toy)")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels")
    parser.add_argument("--time-dim", type=int, default=128, help="Time embedding dim")
    parser.add_argument("--T", type=int, default=200, help="Diffusion steps")
    parser.add_argument("--s", type=float, default=0.008, help="Cosine schedule offset")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/toy", help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--pretrain-denoiser-only", action="store_true",
                        help="Pre-train only the unconditional denoiser (no encoder)")
    parser.add_argument("--pretrained-denoiser", type=str, default=None,
                        help="Path to pretrained denoiser checkpoint (freezes it)")
    parser.add_argument("--denoiser-epochs", type=int, default=30,
                        help="Epochs for denoiser pre-training")
    parser.add_argument("--beta-fixed", type=float, default=None,
                        help="Fixed beta value (disables KL annealing)")
    parser.add_argument("--beta", type=float, default=2e-5,
                        help="Beta value for VAE mode (default 2e-5, tuned on disks sweep)")
    parser.add_argument("--mode", type=str, default="sami", choices=["sami", "vae", "vae-nsynth", "sami-nsynth"],
                        help="Training mode: sami, vae (toy), vae-nsynth, sami-nsynth")
    parser.add_argument("--vae-warmup", type=int, default=5,
                        help="VAE warmup epochs with beta=0")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    parser.add_argument("--nsynth-root", type=str, default="data/nsynth-train",
                        help="NSynth dataset root directory")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--pretrain-denoiser-2d", action="store_true",
                        help="Pre-train MelUNet on NSynth")
    parser.add_argument("--denoiser-checkpoint", type=str, default=None,
                        help="Path to pre-trained denoiser for SAMI encoder")
    parser.add_argument("--denoiser-channels", type=int, default=128,
                        help="Base channels for MelUNet")
    parser.add_argument("--oversample-t", action="store_true",
                        help="Oversample high timesteps (Beta(2,1)) for frozen denoiser")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision bf16 autocast for encoder+guidance")
    parser.add_argument("--probe-every", type=int, default=5,
                        help="Linear probe R²(mu→pitch) every N epochs (0=off)")
    parser.add_argument("--probe-size", type=int, default=1000,
                        help="Samples used by the linear probe")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop if R²(pitch) plateaus over N probes (0=off)")
    parser.add_argument("--collapse-thresh", type=float, default=0.05,
                        help="Stop if mean L_z drops below this (posterior collapse)")
    parser.add_argument("--print-t-hist", action="store_true",
                        help="Print t distribution of first batch (oversample check)")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="KL warm-up — beta=0 for N epochs, then ramp")
    parser.add_argument("--ramp-epochs", type=int, default=0,
                        help="Linear beta ramp duration after warmup")
    parser.add_argument("--free-bits", type=float, default=0.0,
                        help="Per-dimension KL free budget in nats (0=standard KL)")
    parser.add_argument("--ckpt-every-steps", type=int, default=0,
                        help="Save mid-epoch checkpoint every N steps (0=off). "
                             "Needed because an epoch (~31 min) can exceed the 29-min SLURM limit")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "nsynth", "disks"],
                        help="Dataset: toy (sinusoids), nsynth, or disks (2D, 3 factors)")

    args = parser.parse_args()

    config = {
        "n_freqs": 10,
        "n_amps": 5,
        "n_per_class": args.n_per_class,
        "signal_length": 256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_warmup": args.beta_warmup,
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "T": args.T,
        "s": args.s,
        "grad_clip": args.grad_clip,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_every": args.checkpoint_every,
        "use_wandb": args.use_wandb,
        "pretrained_denoiser": args.pretrained_denoiser,
        "denoiser_epochs": args.denoiser_epochs,
        "beta_fixed": args.beta_fixed,
        "beta": args.beta,
        "mode": args.mode,
        "vae_warmup": args.vae_warmup,
        "resume": args.resume,
        "nsynth_root": args.nsynth_root,
        "num_workers": args.num_workers,
        "pretrain_denoiser_2d": args.pretrain_denoiser_2d,
        "denoiser_checkpoint": args.denoiser_checkpoint,
        "denoiser_channels": args.denoiser_channels,
        "oversample_t": args.oversample_t,
        "amp": args.amp,
        "probe_every": args.probe_every,
        "probe_size": args.probe_size,
        "early_stop_patience": args.early_stop_patience,
        "collapse_thresh": args.collapse_thresh,
        "print_t_hist": args.print_t_hist,
        "warmup_epochs": args.warmup_epochs,
        "ramp_epochs": args.ramp_epochs,
        "free_bits": args.free_bits,
        "ckpt_every_steps": args.ckpt_every_steps,
    }

    if args.pretrain_denoiser_only:
        print("=" * 60)
        print("  SAMI-Audio — Denoiser Pre-Training")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20} = {v}")
        pretrain_denoiser(config)
        return

    if args.mode == "vae":
        print("=" * 60)
        print("  SAMI-Audio — β-VAE Toy Model Training")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20} = {v}")
        train_vae(config)
        return

    if args.mode == "vae-nsynth":
        print("=" * 60)
        print("  SAMI-Audio — β-VAE NSynth Training")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20} = {v}")
        train_vae_nsynth(config)
        return

    if args.mode == "sami-nsynth":
        if args.pretrain_denoiser_2d:
            print("=" * 60)
            print("  SAMI-Audio — Denoiser Pre-Training")
            print("=" * 60)
            for k, v in config.items():
                print(f"  {k:<20} = {v}")
            pretrain_denoiser_2d(config)
        else:
            print("=" * 60)
            print("  SAMI-Audio — SAMI Encoder Training")
            print("=" * 60)
            for k, v in config.items():
                print(f"  {k:<20} = {v}")
            train_sami_nsynth(config)
        return

    print("=" * 60)
    print("  SAMI-Audio — Toy Model Training")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<20} = {v}")

    if args.mode == "sami" and args.dataset == "disks":
        print("=" * 60)
        print("  SAMI-Audio — Disks Dataset (2D)")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:<20} = {v}")
        train_disks(config)
        return

    train_toy(config)


if __name__ == "__main__":
    main()
