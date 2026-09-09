"""
SAMI-Audio — Evaluate
=====================
Toy model evaluation: MIG + scatter plot + disentanglement check.
NSynth evaluation: TAD, DCI, MIG, FactorVAE, FID, SI-SNR, log-mel L1.

Usage:
    .venv/bin/python evaluate.py --checkpoint checkpoints/toy/model_final.pt
    .venv/bin/python evaluate.py --checkpoint checkpoints/nsynth/beta_1/model_final.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mutual_info_score
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import ToyEncoder, MelEncoder
from models.unet import ToyUNet
from models.losses.diffusion import DiffusionSchedule
from models.sami import SAMI
from models.vae import VaeDecoder, BetaVAE, MelDecoder
from train import generate_toy_dataset, SinusoidDataset


# ---------------------------------------------------------------------------
# MIG computation (corrected: discrete-discrete MI, empirical entropy)
# ---------------------------------------------------------------------------

def compute_mig(
    z: np.ndarray, labels: np.ndarray, n_bins: int = 20
) -> float:
    """
    Compute Mutual Information Gap (MIG) for a single factor.

    Uses discrete-discrete mutual_info_score (exact, deterministic) after
    digitizing continuous z_j. Normalized by empirical entropy H(v) instead
    of log(n_classes) to handle class imbalance correctly.

    Parameters
    ----------
    z : (N, D) latent codes.
    labels : (N,) ground-truth factor labels (discrete).
    n_bins : int
        Number of bins for discretizing continuous z_j.

    Returns
    -------
    mig : float in [0, 1].
    """
    z = np.asarray(z)
    labels = np.asarray(labels)
    N, D = z.shape

    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    h_v = float(-(p * np.log(p)).sum())
    if h_v <= 0:
        return 0.0

    mi = np.zeros(D)
    for j in range(D):
        zj = z[:, j]
        if zj.std() == 0:
            continue
        edges = np.quantile(zj, np.linspace(0, 1, n_bins + 1)[1:-1])
        zj_disc = np.digitize(zj, edges)
        mi[j] = mutual_info_score(zj_disc, labels)

    order = np.argsort(mi)[::-1]
    return float(max(0.0, (mi[order[0]] - mi[order[1]]) / h_v))


# ---------------------------------------------------------------------------
# Positive control: verify the metric on a perfectly disentangled latent
# ---------------------------------------------------------------------------

def positive_control(freq_labels: np.ndarray, amp_labels: np.ndarray,
                     noise: float = 1e-3, seed: int = 0) -> bool:
    """
    Validate MIG metric on a perfectly disentangled latent space.

    Creates z where dim 0 = frequency, dim 1 = amplitude (+ tiny noise).
    MIG must be ~1.0 on both factors.
    """
    rng = np.random.default_rng(seed)
    f = np.asarray(freq_labels, float)
    a = np.asarray(amp_labels, float)
    z = np.stack([
        f + noise * rng.standard_normal(len(f)),
        a + noise * rng.standard_normal(len(a)),
    ], axis=1)
    mig_f = compute_mig(z, freq_labels)
    mig_a = compute_mig(z, amp_labels)
    print(f"[POSITIVE CONTROL] MIG_freq={mig_f:.3f}  MIG_amp={mig_a:.3f}  (attesi ~1.0)")
    if mig_f > 0.8 and mig_a > 0.8:
        print("[POSITIVE CONTROL] OK — metrica affidabile")
        return True
    print("[POSITIVE CONTROL] FAILED — metrica NON affidabile")
    return False


# ---------------------------------------------------------------------------
# Quality metrics (corrected)
# ---------------------------------------------------------------------------

def mel_features(mel: np.ndarray) -> np.ndarray:
    """(N,1,128,256) → (N,256): mean and std per mel-bin."""
    m = mel.reshape(mel.shape[0], mel.shape[-2], mel.shape[-1])
    return np.concatenate([m.mean(-1), m.std(-1)], axis=1)


def frechet_mel(real: np.ndarray, gen: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance on low-dim mel features (proxy for FID)."""
    fr = mel_features(real)
    fg = mel_features(gen)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cov_r = np.cov(fr, rowvar=False) + eps * np.eye(fr.shape[1])
    cov_g = np.cov(fg, rowvar=False) + eps * np.eye(fg.shape[1])
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


def mel_l1(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Mean absolute error in mel space (raw, not log)."""
    return float(np.mean(np.abs(x - x_hat)))


def knn_probe_accuracy(
    z_train: np.ndarray, labels_train: np.ndarray,
    z_test: np.ndarray, labels_test: np.ndarray, k: int = 5,
) -> float:
    """
    Majority-vote k-NN probe accuracy.

    For each factor v: select the latent dim with lowest k-NN error
    on training set, report accuracy of that dim on test set.
    """
    from sklearn.neighbors import KNeighborsClassifier
    D = z_train.shape[1]
    errors = np.zeros(D)
    for j in range(D):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(z_train[:, j:j + 1], labels_train)
        pred = knn.predict(z_test[:, j:j + 1])
        errors[j] = 1.0 - np.mean(pred == labels_test)
    best_j = np.argmin(errors)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(z_train[:, best_j:best_j + 1], labels_train)
    pred = knn.predict(z_test[:, best_j:best_j + 1])
    n_classes = len(np.unique(labels_train))
    acc = float(np.mean(pred == labels_test))
    baseline = 1.0 / n_classes
    print(f"  knn_probe: acc={acc:.4f} (baseline random = {baseline:.4f})")
    return acc


# ---------------------------------------------------------------------------
# Latent space visualization
# ---------------------------------------------------------------------------

def plot_latent_space(
    z: np.ndarray,
    freq_labels: np.ndarray,
    amp_labels: np.ndarray,
    save_dir: str = "plots",
) -> None:
    """
    Create scatter plots of the 2D latent space colored by ground-truth factors.

    Generates:
    - plots/toy_latent_freq.png  : colored by frequency class
    - plots/toy_latent_amp.png   : colored by amplitude class
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter_freq = axes[0].scatter(z[:, 0], z[:, 1], c=freq_labels, cmap="viridis", s=1, alpha=0.5)
    axes[0].set_xlabel("Latent dim 0")
    axes[0].set_ylabel("Latent dim 1")
    axes[0].set_title("Colored by Frequency")
    plt.colorbar(scatter_freq, ax=axes[0], label="Frequency class")

    scatter_amp = axes[1].scatter(z[:, 0], z[:, 1], c=amp_labels, cmap="plasma", s=1, alpha=0.5)
    axes[1].set_xlabel("Latent dim 0")
    axes[1].set_ylabel("Latent dim 1")
    axes[1].set_title("Colored by Amplitude")
    plt.colorbar(scatter_amp, ax=axes[1], label="Amplitude class")

    plt.tight_layout()
    path = os.path.join(save_dir, "toy_latent_space.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO]  Latent space plot saved: {path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_toy(checkpoint_path: str, output_dir: str = "plots") -> dict:
    """
    Load trained model, encode the full toy dataset, compute MIG and plot.

    Returns a dict with evaluation results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    # ---- Load checkpoint ----------------------------------------------------
    print(f"[INFO]  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mode = ckpt.get("mode", "sami")

    encoder = ToyEncoder(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config["base_channels"],
        signal_length=config["signal_length"],
    )

    if mode == "vae":
        decoder = VaeDecoder(
            latent_dim=config["latent_dim"],
            signal_length=config["signal_length"],
        )
        model = BetaVAE(encoder, decoder, beta=config.get("beta", 4.0))
    else:
        denoiser = ToyUNet(
            in_channels=1,
            base_channels=config["base_channels"],
            time_dim=config["time_dim"],
        )
        diffusion = DiffusionSchedule(T=config["T"], s=config["s"])
        model = SAMI(encoder, denoiser, diffusion, beta=config.get("beta", config.get("beta_end", 1.0)))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    mode_tag = "(VAE)" if mode == "vae" else ""
    print(f"[INFO]  Model loaded {mode_tag} (epoch {epoch}), params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Data ---------------------------------------------------------------
    signals, freq_labels, amp_labels = generate_toy_dataset(
        n_freqs=config["n_freqs"],
        n_amps=config["n_amps"],
        n_per_class=config["n_per_class"],
        signal_length=config["signal_length"],
    )
    dataset = SinusoidDataset(signals, freq_labels, amp_labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # ---- Encode all samples ------------------------------------------------
    all_z: list[np.ndarray] = []
    all_freq: list[int] = []
    all_amp: list[int] = []

    print("[INFO]  Encoding dataset...")
    with torch.no_grad():
        for x, f, a in tqdm(loader, desc="Encoding"):
            x = x.to(device)
            mu, _ = model.encoder(x)
            all_z.append(mu.cpu().numpy())
            all_freq.extend(f.tolist())
            all_amp.extend(a.tolist())

    z = np.concatenate(all_z, axis=0)
    freq_labels_arr = np.array(all_freq)
    amp_labels_arr = np.array(all_amp)
    print(f"[INFO]  Encoded {z.shape[0]} samples → latent shape {z.shape}")

    # ---- MIG ------------------------------------------------------------------
    print("[INFO]  Computing MIG...")
    mig_freq = compute_mig(z, freq_labels_arr)
    mig_amp = compute_mig(z, amp_labels_arr)
    print(f"  MIG (frequency)   = {mig_freq:.4f}")
    print(f"  MIG (amplitude)   = {mig_amp:.4f}")

    # ---- Plot -----------------------------------------------------------------
    if config["latent_dim"] == 2:
        plot_latent_space(z, freq_labels_arr, amp_labels_arr, output_dir)
    else:
        print(f"[WARN]  Latent dim = {config['latent_dim']}, skipping 2D scatter (only works for D=2)")

    # ---- Disentanglement check -----------------------------------------------
    results = {"mig_freq": mig_freq, "mig_amp": mig_amp}

    print()
    print("=" * 60)
    if mig_freq > 0.5 and mig_amp > 0.5:
        print("  DISENTANGLEMENT: PASSED")
        print(f"  MIG freq = {mig_freq:.3f} > 0.5, MIG amp = {mig_amp:.3f} > 0.5")
        print("  Disentanglement funziona. Pronto per il β-VAE su NSynth.")
    else:
        print("  DISENTANGLEMENT: FAILED")
        print(f"  MIG freq = {mig_freq:.3f}, MIG amp = {mig_amp:.3f}")
        if mode == "vae":
            print("  Controlla l'encoder (Half-UNet) e il valore di β")
        else:
            print("  Controlla il meccanismo di guidance (sami.py: _log_q_and_grad)")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# NSynth metrics
# ---------------------------------------------------------------------------

def compute_tad(
    z: np.ndarray, labels: np.ndarray, n_bins: int = 20
) -> float:
    """
    Total AUROC Difference — how exclusively each factor is captured by one axis.

    For each latent dimension j:
      1. Compute AUROC for predicting label v from z_j alone.
      2. TAD = mean over v of (AUROC_best - max_{j≠best} AUROC_j).

    Parameters
    ----------
    z : (N, D) latent codes.
    labels : (N,) discrete labels.
    n_bins : int

    Returns
    -------
    tad : float in [0, 1].
    """
    from sklearn.metrics import roc_auc_score
    N, D = z.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    auroc = np.zeros((D, n_classes))
    for j in range(D):
        z_j = z[:, j]
        if z_j.std() == 0:
            continue
        for vi, v in enumerate(unique_labels):
            y_bin = (labels == v).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == N:
                auroc[j, vi] = 0.5
            else:
                try:
                    auc = roc_auc_score(y_bin, z_j)
                    auroc[j, vi] = max(auc, 1 - auc)
                except ValueError:
                    auroc[j, vi] = 0.5

    tad = 0.0
    for vi in range(n_classes):
        col = auroc[:, vi]
        sorted_idx = np.argsort(col)[::-1]
        if len(sorted_idx) >= 2:
            tad += col[sorted_idx[0]] - col[sorted_idx[1]]

    return tad / n_classes


def compute_factor_vae_metric(
    z_train: np.ndarray, labels_train: np.ndarray,
    z_test: np.ndarray, labels_test: np.ndarray,
    k: int = 5,
) -> float:
    """
    FactorVAE metric: majority-vote k-NN classifier accuracy.

    For each factor v:
      1. Select the latent dim j with lowest k-NN error on training set.
      2. Report accuracy of that dim on test set via majority vote.
    """
    from sklearn.neighbors import KNeighborsClassifier
    D = z_train.shape[1]
    unique_labels = np.unique(labels_train)
    n_classes = len(unique_labels)

    errors = np.zeros(D)
    for j in range(D):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(z_train[:, j:j + 1], labels_train)
        pred = knn.predict(z_test[:, j:j + 1])
        errors[j] = 1.0 - np.mean(pred == labels_test)

    best_j = np.argmin(errors)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(z_train[:, best_j:best_j + 1], labels_train)
    pred = knn.predict(z_test[:, best_j:best_j + 1])
    return float(np.mean(pred == labels_test))


def compute_fid(
    real: np.ndarray, generated: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Fréchet Inception Distance between real and generated mel features.

    Uses per-frame statistics as a simple feature space.
    For full FID, a pretrained CNN should replace this summary.
    """
    real_f = real.reshape(real.shape[0], -1)
    gen_f = generated.reshape(generated.shape[0], -1)

    mu_r = real_f.mean(0)
    sigma_r = np.cov(real_f, rowvar=False) + eps * np.eye(real_f.shape[1])
    mu_g = gen_f.mean(0)
    sigma_g = np.cov(gen_f, rowvar=False) + eps * np.eye(gen_f.shape[1])

    diff = mu_r - mu_g
    covmean = np.real(
        (sigma_r @ sigma_g) ** 0.5
    ) if sigma_r.shape[0] < 5000 else np.real(
        np.asarray((sigma_r @ sigma_g) ** 0.5)
    )

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


def compute_si_snr(x: np.ndarray, x_hat: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio.

    SI-SNR = 10·log10(||s_target||² / ||e_noise||²)
    """
    s_target_num = np.sum(x_hat * x, axis=-1, keepdims=True)
    s_target_den = np.sum(x * x, axis=-1, keepdims=True) + 1e-8
    s_target = s_target_num * x / s_target_den
    e_noise = x_hat - s_target
    val = np.mean(
        10 * np.log10(
            (np.sum(s_target ** 2, axis=-1) + 1e-8)
            / (np.sum(e_noise ** 2, axis=-1) + 1e-8)
        )
    )
    return float(val)


def compute_log_mel_l1(x: np.ndarray, x_hat: np.ndarray, eps: float = 1e-5) -> float:
    """Mean absolute error in log-mel space."""
    x_log = np.log(np.maximum(x, eps))
    x_hat_log = np.log(np.maximum(x_hat, eps))
    return float(np.mean(np.abs(x_log - x_hat_log)))


# ---------------------------------------------------------------------------
# NSynth evaluation
# ---------------------------------------------------------------------------

def evaluate_nsynth(checkpoint_path: str, output_dir: str = "plots", max_samples: int = 0) -> dict:
    """
    Load a trained VAE (NSynth), encode the dataset, compute full metrics.

    Parameters
    ----------
    max_samples : int
        If > 0, evaluate only a random subset of this size. Default 0 (all).
    """
    from data.nsynth import NSynthDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]  Device: {device}")

    print(f"[INFO]  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mode = ckpt.get("mode", "vae-nsynth")

    encoder = MelEncoder(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config.get("base_channels_vae", 64),
        input_size=config.get("input_size", (128, 256)),
    )
    decoder = MelDecoder(
        latent_dim=config["latent_dim"],
        base_channels=config.get("decoder_base_channels", 128),
    )
    model = BetaVAE(encoder, decoder, beta=config.get("beta", 4.0))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    print(f"[INFO]  Model loaded ({mode}, epoch {epoch}), params: {sum(p.numel() for p in model.parameters()):,}")

    print("[INFO]  Loading NSynth dataset...")
    dataset = NSynthDataset(root=config.get("nsynth_root", "data/nsynth-train"))
    if max_samples > 0 and max_samples < len(dataset):
        idx = np.random.default_rng(42).choice(len(dataset), size=max_samples, replace=False)
        from torch.utils.data import Subset
        dataset = Subset(dataset, idx)
        print(f"[INFO]  Evaluating on subset: {max_samples} samples")
    else:
        print(f"[INFO]  Evaluating on full dataset: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    all_z: list[np.ndarray] = []
    all_pitch: list[int] = []
    all_family: list[str] = []
    all_mel: list[np.ndarray] = []
    all_recon: list[np.ndarray] = []

    print("[INFO]  Encoding dataset...")
    with torch.no_grad():
        for x, meta in tqdm(loader, desc="Encoding"):
            x = x.to(device)
            mu, _ = model.encoder(x)
            z_sample = mu + model.encoder(x)[1].sqrt() * torch.randn_like(mu)
            x_recon = model.decoder(z_sample)

            all_z.append(mu.cpu().numpy())
            all_mel.append(x.cpu().numpy())
            all_recon.append(x_recon.cpu().numpy())

            pitch_vals = meta["pitch"]
            family_vals = meta["instrument_family"]
            if isinstance(pitch_vals, torch.Tensor):
                all_pitch.extend(pitch_vals.cpu().tolist())
            else:
                all_pitch.extend(pitch_vals)
            if isinstance(family_vals, torch.Tensor):
                all_family.extend(family_vals.cpu().tolist())
            else:
                all_family.extend(family_vals)

    z = np.concatenate(all_z, axis=0)
    mel_real = np.concatenate(all_mel, axis=0)
    mel_recon = np.concatenate(all_recon, axis=0)
    pitch_arr = np.array(all_pitch)
    family_arr = np.array(all_family)

    print(f"[INFO]  Encoded {z.shape[0]} samples → latent shape {z.shape}")
    print()

    # MIG
    print("[INFO]  Computing MIG...")
    mig_pitch = compute_mig(z, pitch_arr)
    import pandas as pd
    mig_family = compute_mig(z, pd.Categorical(family_arr).codes)
    print(f"  MIG (pitch)    = {mig_pitch:.4f}")
    print(f"  MIG (family)   = {mig_family:.4f}")

    # TAD
    print("[INFO]  Computing TAD...")
    tad_pitch = compute_tad(z, pitch_arr)
    tad_family = compute_tad(z, family_arr_to_int(family_arr))
    print(f"  TAD (pitch)    = {tad_pitch:.4f}")
    print(f"  TAD (family)   = {tad_family:.4f}")

    # k-NN probe
    print("[INFO]  Computing k-NN probe accuracy...")
    n_test = min(10000, len(z) // 2)
    idx_p = np.random.default_rng(42).permutation(len(z))
    z_train, z_test = z[idx_p[n_test:]], z[idx_p[:n_test]]
    p_train, p_test = pitch_arr[idx_p[n_test:]], pitch_arr[idx_p[:n_test]]
    f_train, f_test = family_arr[idx_p[n_test:]], family_arr[idx_p[:n_test]]
    f_train_int = family_arr_to_int(f_train)
    f_test_int = family_arr_to_int(f_test)

    knn_pitch = knn_probe_accuracy(z_train, p_train, z_test, p_test)
    knn_family = knn_probe_accuracy(z_train, f_train_int, z_test, f_test_int)

    # Fréchet mel distance
    print("[INFO]  Computing Fréchet mel distance...")
    fd_val = frechet_mel(mel_real[:n_test], mel_recon[:n_test])
    print(f"  Fréchet mel = {fd_val:.2f}")

    # Mel L1
    print("[INFO]  Computing mel L1...")
    l1_val = mel_l1(mel_real[:n_test], mel_recon[:n_test])
    print(f"  mel L1 = {l1_val:.4f}")

    results = {
        "mig_pitch": mig_pitch, "mig_family": mig_family,
        "tad_pitch": tad_pitch, "tad_family": tad_family,
        "knn_pitch": knn_pitch, "knn_family": knn_family,
        "frechet_mel": fd_val, "mel_l1": l1_val,
    }

    print()
    print("=" * 60)
    print("  RESULTS SUMMARY (β-VAE NSynth)")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<18} = {v:.4f}" if isinstance(v, float) else f"  {k:<18} = {v}")
    print("=" * 60)

    return results


def family_arr_to_int(arr: np.ndarray) -> np.ndarray:
    """Convert string family array to integer labels."""
    import pandas as pd
    return pd.Categorical(arr).codes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAMI-Audio — Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mode = ckpt.get("mode", "sami")

    if mode in ("vae-nsynth",):
        evaluate_nsynth(args.checkpoint, args.output_dir, args.max_samples)
    else:
        evaluate_toy(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
