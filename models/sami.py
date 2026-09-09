"""
SAMI-Audio — Core SAMI Module
===============================
Score-based Autoencoder for Multiscale Inference.

Implements the guidance-based formulation from Lyo, Simoncelli & Savin (2025):

    ε̂(x_t, t, z) = ε_θ(x_t, t) − γ_t · ∇_{x_t} log q_φ(z | x_t)

The guidance score g_t is computed via torch.autograd.grad, allowing
backpropagation through the guidance mechanism.

Reference:
    Lyo, Simoncelli & Savin (2025). SAMI: Score-based Autoencoder for
    Multiscale Inference. arXiv:2512.17127.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.diffusion import DiffusionSchedule
from models.losses.disentanglement import kl_divergence, mahalanobis_log_prob


class SAMI(nn.Module):
    """
    Core SAMI model: encoder + denoiser + score-based guidance.

    Architecture-agnostic — works with any encoder/denoiser pair that
    satisfies the expected interface.

    Parameters
    ----------
    encoder : nn.Module
        Inference network mapping input → (mu, sigma2).
        Forward: x → mu, sigma2   with shapes (B,C,L) → (B,D), (B,D).
    denoiser : nn.Module
        Unconditional denoiser predicting ε from x_t and t.
        Forward: (x, t) → eps  with shapes (B,C,L), (B,) → (B,C,L).
    diffusion : DiffusionSchedule
        Pre-computed diffusion schedule (T, betas, alphas, alphas_cumprod).
    beta : float
        Weight for the KL regularization term. Default 1.0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        denoiser: nn.Module,
        diffusion: DiffusionSchedule,
        beta: float = 1.0,
        frozen_denoiser: bool = False,
        oversample_t: bool = False,
        amp_denoiser: bool = False,
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.beta = beta
        self.free_bits = free_bits
        self.frozen_denoiser = frozen_denoiser
        self.oversample_t = oversample_t
        self.amp_denoiser = amp_denoiser

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode data to a latent vector via reparameterization.

        Parameters
        ----------
        x : (B, C, L) clean data.

        Returns
        -------
        z : (B, D) sampled latent.
        """
        mu, sigma2 = self.encoder(x)
        eps = torch.randn_like(mu)
        return mu + sigma2.sqrt() * eps

    def _log_q_and_grad(
        self, z: torch.Tensor, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log q_φ(z | x_t) and its gradient w.r.t. x_t.

        z carries gradient from the clean encoder path — this is the core of
        SAMI's representation learning. Unlike classifier guidance where y is
        an external fixed label, z is produced by the encoder and must
        propagate gradient for the encoder to learn useful representations.

        Parameters
        ----------
        z : (B, D) latent sample (carries gradient).
        x_t : (B, C, L) noisy data (requires grad).

        Returns
        -------
        log_q : scalar sum over batch.
        g_t : (B, C, L) gradient ∇_{x_t} log q(z | x_t).
        """
        mu_t, sigma2_t = self.encoder(x_t)
        log_q = mahalanobis_log_prob(z, mu_t, sigma2_t).sum()
        g_t = torch.autograd.grad(log_q, x_t, create_graph=True)[0]
        return log_q, g_t

    def forward(
        self, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass — Algorithm 2 from the SAMI paper.

        1. Sample t, ε. Compute x_t via forward diffusion.
        2. Encode clean x0 → z.
        3. Encode noisy x_t → (μ_t, σ²_t). Compute guidance g_t.
        4. ε_guided = ε_θ(x_t, t) − γ_t · g_t.
        5. L = L_x + β · L_z.

        Parameters
        ----------
        x0 : (B, C, L) clean data.

        Returns
        -------
        loss : scalar total loss.
        L_x : scalar reconstruction (score matching) loss.
        L_z : scalar KL regularization loss.
        z : (B, D) latent sample (detached, for logging).
        """
        B = x0.shape[0]
        device = x0.device
        T = self.diffusion.T

        if self.oversample_t:
            # Beta(4,1): media ~0.8 (t~800), coda su tutti i t. Spinge la
            # massa verso gli t alti dove la guidance ha leva (diagnosi
            # stratificata: Δ +408% a 750-1000), senza troncare la
            # distribuzione (mismatch col training del denoiser).
            u = torch.distributions.Beta(4.0, 1.0).sample((B,)).to(device)
            t = (u * T).long().clamp(0, T - 1)
        else:
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        self._last_t = t

        eps = torch.randn_like(x0)
        self._last_eps = eps
        xt = self.diffusion.q_sample(x0, t, eps)

        mu0, sigma2_0 = self.encoder(x0)
        eps_z = torch.randn_like(mu0)
        z = mu0 + sigma2_0.sqrt() * eps_z          # carries gradient to encoder via g_t

        xt.requires_grad_(True)
        log_q, g_t = self._log_q_and_grad(z, xt)

        gamma_t = self.diffusion.gamma(t)
        while gamma_t.dim() < g_t.dim():
            gamma_t = gamma_t.unsqueeze(-1)

        if self.frozen_denoiser:
            with torch.no_grad():
                if self.amp_denoiser:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        eps_pred = self.denoiser(xt.detach(), t)
                    eps_pred = eps_pred.float()
                else:
                    eps_pred = self.denoiser(xt.detach(), t)
        else:
            eps_pred = self.denoiser(xt, t)

        eps_guided = eps_pred - gamma_t * g_t

        L_x = F.mse_loss(eps_guided, eps)
        # L_z post-floor (usato nella loss); L_z_raw (senza floor) per il
        # monitoring del collasso: con free bits attive, L_z post-floor ≈ 0
        # anche quando l'encoder è sano (le dim usano il budget gratuito),
        # quindi la guardia di collasso DEVE guardare la KL grezza.
        L_z = kl_divergence(mu0, sigma2_0, free_bits=self.free_bits)
        L_z_raw = kl_divergence(mu0, sigma2_0, free_bits=0.0)
        loss = L_x + self.beta * L_z

        return loss, L_x.detach(), L_z.detach(), z.detach(), L_z_raw.detach()

    @torch.no_grad()
    def uncond_lx(
        self, x0: torch.Tensor, t: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        L_x of the denoiser WITHOUT guidance (g_t = 0).

        If t and eps are given, uses them (PAIRED comparison with the
        guided loss on the same batch/t/noise). Otherwise samples fresh
        from the same distribution as forward.

        Baseline for the "guidance helps" test during training: if the guided
        L_x is not below this, the encoder is decorative (case 2 in the
        diagnostics — the encoder receives no useful gradient from the
        guidance, so any beta crushes it toward the prior).
        """
        B = x0.shape[0]
        device = x0.device
        T = self.diffusion.T
        if t is None:
            if self.oversample_t:
                u = torch.distributions.Beta(4.0, 1.0).sample((B,)).to(device)
                t = (u * T).long().clamp(0, T - 1)
            else:
                t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(x0)
        xt = self.diffusion.q_sample(x0, t, eps)
        eps_pred = self.denoiser(xt, t)
        return F.mse_loss(eps_pred, eps)

    def sample(
        self,
        z: torch.Tensor,
        shape: tuple[int, int, int],
        n_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples via DDIM with guidance.

        Each DDIM step computes the guidance gradient g_t via autograd,
        then detaches it before the update. We use local enable_grad blocks
        instead of a global @torch.no_grad() decorator because the guidance
        computation itself requires gradient tracking.

        Parameters
        ----------
        z : (B, D) latent vector to condition on.
        shape : (C, H, W) or (C, L) desired output shape per sample.
        n_steps : int
            Number of DDIM steps. Default 50.
        guidance_scale : float
            Multiplicative scale on the guidance term γ_t·g_t (1.0 in
            training, 3-5 in the demo to make pitch control reliable).

        Returns
        -------
        x0 : (B, C, L) generated samples.
        """
        device = z.device
        B = z.shape[0]
        T = self.diffusion.T

        step_indices = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)

        xt = torch.randn(B, *shape, device=device)

        for i in range(n_steps - 1):
            t_curr = step_indices[i].expand(B)
            t_next = step_indices[i + 1].expand(B)

            with torch.enable_grad():
                xt.requires_grad_(True)
                _, g_t = self._log_q_and_grad(z, xt)
            g_t = g_t.detach()

            gamma_t = self.diffusion.gamma(t_curr)
            while gamma_t.dim() < g_t.dim():
                gamma_t = gamma_t.unsqueeze(-1)

            with torch.no_grad():
                eps_pred = self.denoiser(xt, t_curr)
                eps_guided = eps_pred - guidance_scale * gamma_t * g_t

                alpha_curr = self.diffusion.alphas_cumprod[t_curr]
                alpha_next = self.diffusion.alphas_cumprod[t_next]

                while alpha_curr.dim() < xt.dim():
                    alpha_curr = alpha_curr.unsqueeze(-1)
                    alpha_next = alpha_next.unsqueeze(-1)

                x0_pred = (xt - (1.0 - alpha_curr).sqrt() * eps_guided) / alpha_curr.sqrt()
                x0_pred = x0_pred.clamp(-1.0, 1.0)

                c_curr = (1.0 - alpha_curr).sqrt()
                c_next = (1.0 - alpha_next).sqrt()

                xt = alpha_next.sqrt() * x0_pred + c_next / c_curr * (xt - alpha_curr.sqrt() * x0_pred)

        return xt.detach()

    def sample_seeded(
        self,
        z: torch.Tensor,
        shape: tuple[int, int, int],
        n_steps: int = 50,
        generator: torch.Generator | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Come `sample` ma con x_T iniziale determinato dal generator.

        Serve per il confronto appaiato nel timbre transfer: condizioni
        diverse (A, B, transfer) che partono dallo STESSO rumore iniziale
        rendono la differenza di pitch attribuibile a z, non al rumore.
        """
        device = z.device
        B = z.shape[0]
        T = self.diffusion.T

        step_indices = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)

        if generator is not None:
            xt = torch.randn(B, *shape, device=device, generator=generator)
        else:
            xt = torch.randn(B, *shape, device=device)

        for i in range(n_steps - 1):
            t_curr = step_indices[i].expand(B)
            t_next = step_indices[i + 1].expand(B)

            with torch.enable_grad():
                xt.requires_grad_(True)
                _, g_t = self._log_q_and_grad(z, xt)
            g_t = g_t.detach()

            gamma_t = self.diffusion.gamma(t_curr)
            while gamma_t.dim() < g_t.dim():
                gamma_t = gamma_t.unsqueeze(-1)

            with torch.no_grad():
                eps_pred = self.denoiser(xt, t_curr)
                eps_guided = eps_pred - guidance_scale * gamma_t * g_t

                alpha_curr = self.diffusion.alphas_cumprod[t_curr]
                alpha_next = self.diffusion.alphas_cumprod[t_next]

                while alpha_curr.dim() < xt.dim():
                    alpha_curr = alpha_curr.unsqueeze(-1)
                    alpha_next = alpha_next.unsqueeze(-1)

                x0_pred = (xt - (1.0 - alpha_curr).sqrt() * eps_guided) / alpha_curr.sqrt()
                x0_pred = x0_pred.clamp(-1.0, 1.0)

                c_curr = (1.0 - alpha_curr).sqrt()
                c_next = (1.0 - alpha_next).sqrt()

                xt = alpha_next.sqrt() * x0_pred + c_next / c_curr * (xt - alpha_curr.sqrt() * x0_pred)

        return xt.detach()
