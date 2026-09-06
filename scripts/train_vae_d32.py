#!/usr/bin/env python3
"""
SAMI-Audio — Train a clean D=32 beta-VAE on the global mel cache
==================================================================
Protocol option 1 (docs/SAMI_Audio_Metriche_Confronto.md Sec. 5): the
head-to-head with SAMI is valid only if the beta-VAE sees the SAME data
(global-normalized mels), the SAME encoder architecture (current MelEncoder,
flatten — no legacy pooling) and the SAME latent dimension (D=32).

This script trains exactly that baseline:
  - data:      CachedMelDataset (global normalization, same cache as SAMI)
  - encoder:   MelEncoder D=32 (same class/weights layout as SAMI's encoder)
  - decoder:   MelDecoder D=32
  - loss:      MSE reconstruction + beta * KL  (beta fixed, no warm-up)
  - hyper:     batch 128, lr 1e-4, cosine annealing (Phase-2 values)
  - resume:    auto-resume from the latest checkpoint in CKPT_DIR
               (per-epoch saves; one epoch is ~1-2 min on GPU, well below
               the 29-min SLURM limit)

USAGE:  python scripts/train_vae_d32.py [--epochs 40] [--beta 1e-3] [--batch-size 128]
OUTPUT: checkpoints/nsynth/vae_d32_global/{nsynth_epoch_*.pt, model_final.pt}
"""
import argparse, os, sys, glob, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nsynth import CachedMelDataset
from models.encoder import MelEncoder
from models.vae import MelDecoder, BetaVAE
from models.losses.disentanglement import kl_divergence


def find_latest(ckpt_dir):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "nsynth_epoch_*.pt")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--ckpt-dir", default="checkpoints/nsynth/vae_d32_global")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}  |  D={args.latent_dim}  beta={args.beta}  "
          f"epochs={args.epochs}  batch={args.batch_size}")

    ds = CachedMelDataset()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4,
                                         pin_memory=True, drop_last=True)
    print(f"[INFO] Dataset: {len(ds)} cached mels, {len(loader)} batches/epoch")

    encoder = MelEncoder(in_channels=1, latent_dim=args.latent_dim,
                         base_channels=64, input_size=(128, 256))
    decoder = MelDecoder(latent_dim=args.latent_dim, base_channels=128)
    vae = BetaVAE(encoder, decoder, beta=args.beta).to(device)
    print(f"[INFO] Params: {sum(p.numel() for p in vae.parameters()):,}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    start_epoch = 1
    latest = find_latest(args.ckpt_dir)
    if latest:
        ck = torch.load(latest, map_location=device, weights_only=False)
        vae.load_state_dict(ck["model_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[INFO] Resuming from {os.path.basename(latest)} at epoch {start_epoch}")

    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader))

    vae.train()
    for epoch in range(start_epoch, args.epochs + 1):
        tot_loss = tot_recon = tot_kl = 0.0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad()
            loss, recon, kl = vae(x)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item()
            tot_recon += recon.item()
            tot_kl += kl.item()
        avg = tuple(v / len(loader) for v in (tot_loss, tot_recon, tot_kl))
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg[0]:.4f}  "
              f"recon={avg[1]:.4f}  KL={avg[2]:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        path = os.path.join(args.ckpt_dir, f"nsynth_epoch_{epoch:04d}.pt")
        torch.save({"epoch": epoch, "model_state_dict": vae.state_dict(),
                    "config": vars(args), "mode": "vae-nsynth-global"}, path)
        print(f"  [CKPT] {path}")

    final = os.path.join(args.ckpt_dir, "model_final.pt")
    torch.save({"epoch": args.epochs, "model_state_dict": vae.state_dict(),
                "config": vars(args), "mode": "vae-nsynth-global"}, final)
    print(f"[DONE] {final}")


if __name__ == "__main__":
    main()
