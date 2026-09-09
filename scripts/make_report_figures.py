#!/usr/bin/env python3
"""
Genera tutte le figure del report in plots/finals/.

Estrae i dati dai log (senza GPU) e produce le 7 figure:
  fig1_denoiser_gate.png   — spettrogrammi DDIM del denoiser (gate 3a)
  fig2_collapse_betas.png  — L_z vs epoche (collasso a tutti i β + warm-up)
  fig3_guidance_by_t.png   — Δ% guidato-vs-uncond per fascia di t
  fig4_kl_perdim.png       — distribuzione KL per-dim (D=128 vs D=32)
  fig5_convergence.png     — margine R²−control vs epoche (train 3b)
  fig6_transfer_demo.png   — spettrogrammi A|T|B dei 3 casi demo
  fig7_demo_grid.png       — bar chart successi X/Y per (s, α)

USO: python make_report_figures.py  [dir_output]
"""
import os, re, sys, glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "plots/finals"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

STATS_JSON = "data/norm_stats.json"
LOGMIN, LOGMAX = -11.5129, 4.4849  # norm_stats.json (fallback)

def load_norm_stats():
    global LOGMIN, LOGMAX
    try:
        import json
        s = json.load(open(STATS_JSON))
        LOGMIN, LOGMAX = s["log_mel_min"], s["log_mel_max"]
    except Exception:
        pass

def norm_to_log(mel):
    """[-1,1] -> log-mel grezzo (per visualizzazione)."""
    return (mel + 1.0) / 2.0 * (LOGMAX - LOGMIN) + LOGMIN

# ---------------------------------------------------------------- fig1
def fig1_denoiser_gate():
    # 3 rows: real NSynth mel (input) | DDIM sample (output)
    path = "plots/finals/samples.npy"
    if not os.path.isfile(path):
        print("  [fig1] SKIP: samples.npy non trovato")
        return
    from data.nsynth import CachedMelDataset
    ds = CachedMelDataset()
    metas = ds.metas

    def find(family, pitch):
        for i, m in enumerate(metas):
            if m["instrument_family"] == family and m["pitch"] == pitch:
                return i

    REFS = [("guitar", 60), ("brass", 67), ("keyboard", 60)]
    reals = []
    for fam, pit in REFS:
        i = find(fam, pit)
        reals.append(np.asarray(ds.arr[i]).copy())   # (1,128,256) in [-1,1]

    x0 = np.load(path)  # (10,1,128,256) DDIM samples in [-1,1]
    n = min(3, len(x0))
    fig, axes = plt.subplots(n, 2, figsize=(9, 9))
    for r in range(n):
        axes[r][0].imshow(norm_to_log(reals[r][0]), aspect="auto",
                          origin="lower", cmap="magma")
        axes[r][1].imshow(norm_to_log(x0[r, 0]), aspect="auto",
                          origin="lower", cmap="magma")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[r][1].set_xticks([]); axes[r][1].set_yticks([])
        axes[r][0].set_ylabel(f"{REFS[r][0]} MIDI {REFS[r][1]}",
                              fontsize=9, rotation=90, labelpad=30)
    axes[0][0].set_title("input: real NSynth mel")
    axes[0][1].set_title("output: DDIM sample")
    fig.suptitle("DDIM sampling from the denoiser — real vs generated")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_denoiser_gate.png"), dpi=130)
    plt.close(fig)
    print("  [fig1] OK")

# ---------------------------------------------------------------- fig2
def parse_epoch_lz(pattern, epoch_re, lz_re, label_re=None):
    """Estrae (epoch, L_z) da un insieme di log con resume."""
    data = {}
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f, errors="ignore") as fh:
                text = fh.read().replace("\r", "\n")
        except Exception:
            continue
        for line in text.split("\n"):
            m = re.search(epoch_re, line)
            if not m:
                continue
            epoch = int(m.group(1))
            ml = re.search(lz_re, line)
            if not ml:
                continue
            data[epoch] = float(ml.group(1))
    return data

def fig2_collapse_betas():
    # (a) warm-up (Test #2): L_z per epoca su 15 epoche
    wz = parse_epoch_lz(
        "logs/t2beta_*.out",
        r"^Epoch\s+(\d+)\s+\|",
        r"L_z=([\d.eE+-]+)",
    )
    # (b) free bits (Test #3): L_z_raw per epoca (vivo)
    fb = parse_epoch_lz(
        "logs/t3fb_*.out",
        r"^Epoch\s+(\d+)\s+\|",
        r"L_z_raw=([\d.eE+-]+)",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if wz:
        eps = sorted(wz.keys())
        vals = [wz[e] for e in eps]
        axes[0].plot(eps, vals, "o-", color="tab:red")
        axes[0].axhline(0.05, ls="--", color="gray", lw=1, label="soglia collasso")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("L_z (nat)")
        axes[0].set_title("Warm-up KL: posterior degenere (β=0) → collasso (rampa)")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "log warm-up assenti", ha="center")
    if fb:
        eps = sorted(fb.keys())
        vals = [fb[e] for e in eps]
        axes[1].plot(eps, vals, "o-", color="tab:green")
        axes[1].axhline(0.05, ls="--", color="gray", lw=1, label="soglia collasso")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("L_z_raw (nat)")
        axes[1].set_title("Free bits: KL grezza viva (collasso risolto)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "log free-bits assenti", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_collapse_betas.png"))
    plt.close(fig)
    print(f"  [fig2] OK (warm-up {len(wz)} pts, free-bits {len(fb)} pts)")

# ---------------------------------------------------------------- fig3
def fig3_guidance_by_t():
    # Valori della diagnosi stratificata su encoder fresco
    bands = ["0-250", "250-500", "500-750", "750-1000"]
    delta = [0.4, 21.4, 116.1, 407.7]  # Δ% guidato vs uncond
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(bands, delta, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("Δ% (L_x guidato vs uncond)")
    ax.set_xlabel("fascia di timestep t")
    ax.set_title("La guida ha leva solo agli alti livelli di rumore")
    for b, d in zip(bars, delta):
        ax.text(b.get_x() + b.get_width()/2, d*1.2, f"{d:.0f}%",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_guidance_by_t.png"))
    plt.close(fig)
    print("  [fig3] OK")

# ---------------------------------------------------------------- fig4
def fig4_kl_perdim():
    # D=32: usa kl_d32.npy pre-calcolato (script GPU) se esiste; altrimenti calcola
    npy_path = os.path.join("plots", "finals", "kl_d32.npy")
    if os.path.isfile(npy_path):
        kl_sorted = np.sort(np.load(npy_path))[::-1]
    else:
        # fallback: calcolo diretto (GPU o CPU — può essere lento su CPU)
        import torch
        from models.encoder import MelEncoder
        from data.nsynth import CachedMelDataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = "checkpoints/nsynth/sami_d32/model_final.pt"
        if not os.path.isfile(ckpt_path):
            print("  [fig4] SKIP: checkpoint assente")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc = MelEncoder(in_channels=1, latent_dim=32, base_channels=64,
                         input_size=(128, 256)).to(device)
        sd = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items()
              if k.startswith("encoder.")}
        enc.load_state_dict(sd, strict=True)
        enc.eval()
        ds = CachedMelDataset()
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ds), 2000, replace=False)
        mu_all, var_all = [], []
        with torch.no_grad():
            for b in range(0, 2000, 256):
                x = torch.from_numpy(np.asarray(ds.arr[idx[b:b+256]]).copy()).to(device)
                mu, s2 = enc(x)
                mu_all.append(mu.cpu().numpy())
                var_all.append(s2.cpu().numpy())
        mu = np.concatenate(mu_all); var = np.concatenate(var_all)
        kl = 0.5 * np.mean(mu**2 + var - np.log(var) - 1, axis=0)
        kl_sorted = np.sort(kl)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 33), kl_sorted, color="tab:purple")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="free bits (0.5 nat)")
    ax.set_xlabel("dimensione (ordine decrescente)")
    ax.set_ylabel("KL (nat)")
    ax.set_title("KL per-dimensione a D=32 — codifica concentrata")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_kl_perdim.png"))
    plt.close(fig)
    print(f"  [fig4] OK (KL top5={[round(v,3) for v in kl_sorted[:5]]})")

# ---------------------------------------------------------------- fig5
def fig5_convergence():
    # margine R²_pitch − R²_control vs epoca, dai probe del train 3b
    probes = {}   # epoch -> margin
    cur_epoch = None
    for f in sorted(glob.glob("logs/sami_enc_9747*.out")):
        try:
            with open(f, errors="ignore") as fh:
                text = fh.read().replace("\r", "\n")
        except Exception:
            continue
        for line in text.split("\n"):
            me = re.search(r"^Epoch\s+(\d+)\s+\|", line)
            if me:
                cur_epoch = int(me.group(1))
            mp = re.search(r"\[PROBE\] R²\(mu→pitch\)=([\d.-]+).*?R²\(control rumore\)=([\d.-]+)", line)
            if mp and cur_epoch is not None:
                probes[cur_epoch] = float(mp.group(1)) - float(mp.group(2))
    if not probes:
        print("  [fig5] SKIP: probe assenti")
        return
    eps = sorted(probes.keys()); vals = [probes[e] for e in eps]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, vals, "o-", color="tab:green")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="soglia attesa pitch")
    ax.set_xlabel("epoch"); ax.set_ylabel("margine R²(pitch) − R²(control)")
    ax.set_title("Convergenza del training (D=32, β=1e-5) — margine piatto a 31")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_convergence.png"))
    plt.close(fig)
    print(f"  [fig5] OK ({len(eps)} probe, margine finale {vals[-1]:.3f})")

# ---------------------------------------------------------------- fig6
def fig6_transfer_demo():
    base = "plots/demo"
    cases = [
        ("Transfer perfetto (seed 1006)", "s5.0_a0.3_seed1006"),
        ("Pitch ok + timbro guitar (seed 1001)", "s5.0_a0.3_seed1001"),
        ("Fallimento: T=A (seed 1000)", "s5.0_a0.3_seed1000"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for r, (title, tag) in enumerate(cases):
        for c, (suf, lab) in enumerate([("_A", "A (guitar)"), ("_T", "TRANSFER"), ("_B", "B (brass)")]):
            p = os.path.join(base, f"{tag}{suf}.wav")
            ax = axes[r][c]
            # Carica WAV e calcola mel
            try:
                import torchaudio, librosa
                wav, sr = torchaudio.load(p)
                y = wav[0].numpy()
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
            except Exception as e:
                ax.text(0.5, 0.5, f"err: {e}", ha="center")
            ax.set_title(lab)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r][0].set_ylabel(title, fontsize=9, rotation=90, labelpad=30)
    fig.suptitle("Timbre transfer demo (s=5, α=0.3) — pitch verso B, timbro resta A")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_transfer_demo.png"))
    plt.close(fig)
    print("  [fig6] OK")

# ---------------------------------------------------------------- fig7
def fig7_demo_grid():
    # Dati onesti dalla tabella demo (X/Y semi validi)
    labels = ["s3 a0.5", "s5 a0.5", "s7 a0.5", "s5 a0.3", "s5 a0.7", "s5 a1.0"]
    ok = [1, 2, 1, 3, 1, 1]
    tot = [6, 7, 3, 7, 7, 6]
    fig, ax = plt.subplots(figsize=(8, 4))
    frac = [o/t for o, t in zip(ok, tot)]
    bars = ax.bar(labels, frac, color=["tab:orange"]*3 + ["tab:green"] + ["tab:orange"]*2)
    ax.axhline(6/8, ls="--", color="gray", lw=1, label="soglia successo (6/8)")
    ax.set_ylabel("frazione semi con transfer completo")
    ax.set_title("Esito demo per (s, α) — best: s=5, α=0.3 (3/7)")
    for b, o, t in zip(bars, ok, tot):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{o}/{t}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_demo_grid.png"))
    plt.close(fig)
    print("  [fig7] OK")

# ----------------------------------------------------------------
def main():
    load_norm_stats()
    print(f"Genero figure in {OUT}")
    fig1_denoiser_gate()
    fig2_collapse_betas()
    fig3_guidance_by_t()
    fig4_kl_perdim()
    fig5_convergence()
    fig6_transfer_demo()
    fig7_demo_grid()
    print("DONE — file in", OUT)

if __name__ == "__main__":
    main()
