"""
SAMI-Audio — VAE Module (β-VAE Baseline)
==========================================
Standard β-VAE with 1D encoder (Half-UNet) and 1D transposed-conv decoder.

Used to validate the encoder, training loop, MIG evaluation,
and disentanglement pipeline on synthetic sinusoids before scaling to NSynth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.disentanglement import kl_divergence


class VaeDecoder(nn.Module):
    """
    1D transposed-convolution decoder for the toy sinusoid model.

    Maps a latent vector z ∈ R^D back to a 1D signal (B, 1, 256).
    Architecture: Linear expansion → series of ConvTranspose1d with SiLU.

    Parameters
    ----------
    latent_dim : int
        Latent space dimension. Default 2.
    signal_length : int
        Output signal length. Default 256.
    """

    def __init__(self, latent_dim: int = 2, signal_length: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.signal_length = signal_length

        init_ch = 64
        init_len = 8

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, init_ch * init_len),
            nn.SiLU(),
        )
        self.init_ch = init_ch
        self.init_len = init_len

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(init_ch, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to a 1D signal.

        Parameters
        ----------
        z : (B, D) latent vector.

        Returns
        -------
        x_recon : (B, 1, L) reconstructed signal in [-1, 1].
        """
        B = z.shape[0]
        h = self.fc(z)
        h = h.view(B, self.init_ch, self.init_len)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        h = self.deconv4(h)
        h = self.deconv5(h)
        return h


class BetaVAE(nn.Module):
    """
    β-VAE combining a Half-UNet encoder and a transposed-conv decoder.

    Parameters
    ----------
    encoder : nn.Module
        Module mapping x → (mu, sigma2). Outputs (B, D) tensors.
    decoder : nn.Module
        Module mapping z → x_recon. Input (B, D), output (B, C, L).
    beta : float
        KL divergence weight. Default 4.0 (standard β-VAE).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, beta: float = 4.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Parameters
        ----------
        x : (B, C, L) input signal.

        Returns
        -------
        loss : scalar total loss = recon_loss + beta * KL.
        recon_loss : scalar reconstruction loss (MSE).
        kl_loss : scalar KL divergence loss (raw, not weighted).
        """
        mu, sigma2 = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma2.sqrt() * eps

        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = kl_divergence(mu, sigma2)
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss.detach(), kl_loss.detach()


class MelDecoder(nn.Module):
    """
    2D transposed-convolution decoder for mel-spectrogram outputs.

    Maps a latent vector z ∈ R^D back to a mel spectrogram (B, 1, 128, 256).
    Architecture: Linear expansion → series of ConvTranspose2d with GroupNorm + SiLU.

    Parameters
    ----------
    latent_dim : int
        Latent space dimension. Default 64.
    base_channels : int
        Largest channel count (at bottleneck). Default 128.
    """

    def __init__(self, latent_dim: int = 64, base_channels: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        init_h, init_w = 8, 16

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * init_h * init_w),
            nn.SiLU(),
        )
        self.base_channels = base_channels
        self.init_h = init_h
        self.init_w = init_w

        ch = base_channels

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, 2, 1, output_padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to a mel spectrogram.

        Parameters
        ----------
        z : (B, D) latent vector.

        Returns
        -------
        x_recon : (B, 1, 128, 256) reconstructed mel in [-1, 1].
        """
        B = z.shape[0]
        h = self.fc(z)
        h = h.view(B, self.base_channels, self.init_h, self.init_w)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        h = self.deconv4(h)
        return h
