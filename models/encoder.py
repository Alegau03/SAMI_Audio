"""
SAMI-Audio — Encoder Module
============================
Inference network (Half-UNet) that maps input data to a Gaussian
posterior q_φ(z|x) = N(μ_φ(x), Σ_φ(x)).

The Half-UNet uses only the Down + Mid blocks of a U-Net — no
upsampling path. Two linear heads produce μ and σ².

For the toy model: 1D convolutions on 1D signals.
For NSynth: 2D convolutions on mel spectrograms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Downsampling 1D convolution block: Conv → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class ConvBlock2D(nn.Module):
    """Downsampling 2D convolution block: Conv2d → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class ToyEncoder(nn.Module):
    """
    1D Half-UNet encoder for the toy sinusoid model.

    Architecture: 3 downsampling blocks + mid block → pooled → μ, σ² heads.
    Output σ² is guaranteed positive via Softplus + epsilon.

    Parameters
    ----------
    in_channels : int
        Input channels. Default 1.
    latent_dim : int
        Latent space dimension. Default 2 (for toy visualization).
    base_channels : int
        Base channel count. Default 32.
    signal_length : int
        Expected input signal length. Default 256.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 2,
        base_channels: int = 32,
        signal_length: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        ch = base_channels

        self.down1 = ConvBlock1D(in_channels, ch, stride=2)       # L → L/2
        self.down2 = ConvBlock1D(ch, ch * 2, stride=2)            # L/2 → L/4
        self.down3 = ConvBlock1D(ch * 2, ch * 4, stride=2)        # L/4 → L/8
        self.mid = ConvBlock1D(ch * 4, ch * 4, stride=1)          # L/8 → L/8

        mid_length = signal_length // 8
        self.pool = nn.AdaptiveAvgPool1d(1)
        flat_dim = ch * 4

        self.head_mu = nn.Linear(flat_dim, latent_dim)
        self.head_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input signal to Gaussian posterior parameters.

        Parameters
        ----------
        x : (B, C, L) input signal.

        Returns
        -------
        mu : (B, D) posterior mean.
        sigma2 : (B, D) posterior variance (strictly positive).
        """
        h = self.down1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.pool(h).squeeze(-1)                     # (B, 128)

        mu = self.head_mu(h)                            # (B, D)
        logvar = self.head_logvar(h)                      # (B, D)
        sigma2 = F.softplus(logvar) + 1e-6               # ensures σ² > 0

        return mu, sigma2


class MelEncoder(nn.Module):
    """
    2D Half-UNet encoder for mel-spectrogram inputs.

    Architecture: N downsampling blocks + mid block → flatten → μ, σ² heads.
    Preserves spatial layout: NO global pooling — position-dependent features
    (pitch on mel axis, disk position on toy) survive through flatten.

    Designed for (B, 1, H, W) inputs. H, W must be divisible by 2^n_down.

    Parameters
    ----------
    in_channels : int
        Input channels. Default 1 (mel spectrogram).
    latent_dim : int
        Latent space dimension. Default 64.
    base_channels : int
        Base channel count. Default 64.
    channel_mult : tuple
        Channel multipliers per level. Default (1, 1, 1, 1).
    input_size : tuple (H, W)
        Input spatial dimensions. Default (128, 256) for NSynth mel.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        base_channels: int = 64,
        channel_mult: tuple = (1, 1, 1, 1),
        input_size: tuple = (128, 256),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        chs = [base_channels * m for m in channel_mult]
        n_levels = len(chs)

        self.down_blocks = nn.ModuleList()
        for i, out_ch in enumerate(chs):
            in_ch = in_channels if i == 0 else chs[i - 1]
            self.down_blocks.append(ConvBlock2D(in_ch, out_ch, stride=2))

        mid_ch = base_channels * 2
        self.mid = nn.Sequential(
            ConvBlock2D(chs[-1], mid_ch, stride=1),
            ConvBlock2D(mid_ch, mid_ch, stride=1),
        )

        final_h = input_size[0] // (2 ** n_levels)
        final_w = input_size[1] // (2 ** n_levels)
        flat_dim = mid_ch * final_h * final_w
        self.flat_h, self.flat_w = final_h, final_w

        self.head_mu = nn.Linear(flat_dim, latent_dim)
        self.head_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to Gaussian posterior parameters.

        Parameters
        ----------
        x : (B, C, H, W) input (mel spectrogram or disk image).

        Returns
        -------
        mu : (B, D) posterior mean.
        sigma2 : (B, D) posterior variance (strictly positive).
        """
        h = x
        for block in self.down_blocks:
            h = block(h)
        h = self.mid(h)
        h = h.flatten(1)                               # (B, mid_ch * H_final * W_final)
        assert h.shape[1] == self.head_mu.in_features, \
            f"Flat dim mismatch: {h.shape[1]} vs head {self.head_mu.in_features}. Check input_size / channel_mult."

        mu = self.head_mu(h)
        logvar = self.head_logvar(h)
        sigma2 = F.softplus(logvar) + 1e-6

        return mu, sigma2
