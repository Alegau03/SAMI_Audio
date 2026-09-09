#!/usr/bin/env python3
"""
SAMI vs beta-VAE — paired metric comparison

One function per metric, applied to BOTH encoders on the SAME samples
(fixed index set, seed 0). Metrics operate on mu (posterior mean):

  probe R^2 (pitch)  — Ridge(alpha=1) + train/test split, margin over noise control
  kNN accuracy (timbre) — k=5, majority vote, chance baseline 25%
  MIG (pitch, family) — discrete-discrete MI, empirical entropy (evaluate.compute_mig)
  cos(w_pitch, w_family) — intervention directions in standardized latent space
  random cosine baseline (per latent dimension D)

Clean comparison: the beta-VAE baseline was retrained on
the SAME global-normalized mel cache as SAMI, with the SAME encoder class
(current MelEncoder, flatten) and the SAME latent dimension D=32
(scripts/train_vae_d32.py). Both checkpoints below load with the same
load_encoder() — no legacy code paths.

Evaluation on the OFFICIAL HELD-OUT (NSynth test split,
scripts/precompute_heldout.py): filtered and normalized identically to the
train set, data unseen during training. Since the splits are
instrument-disjoint, timbre scores also test generalization to unseen
instruments. All of the (small) held-out set is used — no subsampling.

USAGE: python scripts/metrics_comparison.py
OUTPUT: metrics_comparison.json (raw) — metrics_comparison.md is the
        human-readable report of the same numbers.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nsynth import CachedMelDataset
from sklearn.preprocessing import LabelEncoder

SAMICKPT = "checkpoints/nsynth/sami_d32/model_final.pt"
VAECKPT = "checkpoints/nsynth/vae_d32_global/model_final.pt"
N = 5000
SEED = 0
# Held-out ufficiale (test split). Precompute con scripts/precompute_heldout.py.
EVAL_CACHE = "data/mel_cache_test.npy"
EVAL_META = "data/mel_meta_test.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(ckpt_path, latent_dim, device):
    from models.encoder import MelEncoder
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = MelEncoder(in_channels=1, latent_dim=latent_dim, base_channels=64,
                     input_size=(128, 256)).to(device)
    sd = {k[len("encoder."):]: v for k, v in ck["model_state_dict"].items()
          if k.startswith("encoder.")}
    enc.load_state_dict(sd, strict=True)
    enc.eval()
    return enc


def extract_mu(encoder, idx, device):
    """mu + labels on a FIXED index set — identical for every model."""
    ds = CachedMelDataset(cache=EVAL_CACHE, meta_path=EVAL_META)
    metas = ds.metas
    mu_all = []
    with torch.no_grad():
        for b in range(0, len(idx), 256):
            sl = idx[b:b + 256]
            x = torch.from_numpy(np.asarray(ds.arr[sl]).copy()).to(device)
            mu, _ = encoder(x)
            mu_all.append(mu.cpu().numpy())
    mu = np.concatenate(mu_all)
    pitch = np.array([metas[i]["pitch"] for i in idx], dtype=float)
    family = LabelEncoder().fit_transform([metas[i]["instrument_family"] for i in idx])
    return mu, pitch, family


def participation_ratio(mu):
    """Effective dimensionality of the latent from the covariance spectrum.
    PR in [1, D]: ~1 = all info on one axis, ~D = spread over all axes."""
    C = np.cov(mu, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    pr = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))
    return {"PR": pr, "PR_norm": pr / mu.shape[1]}


def probe_r2_margin(X, y, seed=0, alpha=1.0):
    """Ridge R^2 (50/50 STRATIFIED split) and its margin over a
    Gaussian-noise control. Same procedure as train._probe_r2:
    control = Ridge on pure noise. Stratify on y (pitch is discrete:
    37 values) to avoid a lucky/unlucky split on the small held-out set."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)
    r2 = float(Ridge(alpha=alpha).fit(X_tr, y_tr).score(X_te, y_te))
    r2_ctrl = float(Ridge(alpha=alpha).fit(
        rng.standard_normal(X_tr.shape), y_tr).score(
        rng.standard_normal(X_te.shape), y_te))
    return {"r2": r2, "r2_control": r2_ctrl, "margin": r2 - r2_ctrl}


def knn_accuracy(X, y, k=5, seed=0):
    """k-NN majority-vote accuracy (50/50 STRATIFIED split). Chance = 1/4."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def mean_std_over_seeds(fn, X, y, n_seeds=5):
    """Runs fn(X, y, seed) over seeds 0..n_seeds-1 and aggregates.
    Reports per-seed values, mean, std and seed-0 (reproducible reference)."""
    per_seed = {s: fn(X, y, seed=s) for s in range(n_seeds)}
    vals = np.array([per_seed[s] for s in range(n_seeds)])
    return {"per_seed": per_seed,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "seed0": float(per_seed[0])}


def mig_scores(z, pitch, family, n_bins_pitch=10, n_bins_mig=20):
    """MIG on discretized pitch (10 quantile bins) and on family."""
    from evaluate import compute_mig
    pit_d = np.digitize(pitch, np.quantile(pitch, np.linspace(0, 1, 11)[1:-1]))
    out = {"mig_pitch": compute_mig(z, pit_d, n_bins=n_bins_mig),
           "mig_family": compute_mig(z, family, n_bins=n_bins_mig)}
    # positive control with THIS discretization (must be ~1 before trusting)
    rng = np.random.default_rng(SEED)
    z_gt = np.stack([pit_d + 1e-3 * rng.standard_normal(len(pit_d)),
                     family + 1e-3 * rng.standard_normal(len(family))], axis=1)
    ctrl_p = compute_mig(z_gt, pit_d, n_bins=n_bins_mig)
    ctrl_f = compute_mig(z_gt, family, n_bins=n_bins_mig)
    out["positive_control"] = {"pitch": ctrl_p, "family": ctrl_f}
    return out


def direction(mu_s, labels, lo_q=33, hi_q=67):
    lo, hi = np.percentile(labels, [lo_q, hi_q])
    w = mu_s[labels >= hi].mean(0) - mu_s[labels <= lo].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def direction_from_classes(mu_s, family, pos=1, neg=0):
    """w = mean(mu_s | class pos) - mean(mu_s | class neg), normalized."""
    w = mu_s[family == pos].mean(0) - mu_s[family == neg].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    rng = np.random.default_rng(SEED)
    ds_len = len(CachedMelDataset(cache=EVAL_CACHE, meta_path=EVAL_META))
    # the held-out set is small: use it ALL (no subsampling to N)
    IDX = np.arange(ds_len)
    print(f"Device: {DEVICE}  |  held-out samples: {ds_len} (all of them)")

    enc_sami = load_encoder(SAMICKPT, latent_dim=32, device=DEVICE)
    enc_vae = load_encoder(VAECKPT, latent_dim=32, device=DEVICE)

    print("Extracting mu (SAMI)...")
    mu_sami, pitch, family = extract_mu(enc_sami, IDX, DEVICE)
    print("Extracting mu (beta-VAE)...")
    mu_vae, pitch_v, family_v = extract_mu(enc_vae, IDX, DEVICE)
    assert np.array_equal(pitch, pitch_v) and np.array_equal(family, family_v)

    results = {"sami": {}, "bvae": {}}
    for tag, mu in (("sami", mu_sami), ("bvae", mu_vae)):
        # probe R^2 and kNN: mean/std over 5 stratified seeds (robustness on
        # the small held-out set). MIG/cos/PR are split-free (deterministic).
        def _margin(X, y, seed):
            return probe_r2_margin(X, y, seed=seed)["margin"]

        def _ctrl(X, y, seed):
            return probe_r2_margin(X, y, seed=seed)["r2_control"]

        margins = mean_std_over_seeds(_margin, mu, pitch)
        ctrls = mean_std_over_seeds(_ctrl, mu, pitch)
        accs = mean_std_over_seeds(knn_accuracy, mu, family)
        mig = mig_scores(mu, pitch, family)
        # family direction: guitar(1) minus brass(0), standardized
        m, s = mu.mean(0), mu.std(0) + 1e-8
        mu_s = (mu - m) / s
        wp = direction(mu_s, pitch)
        wf = direction_from_classes(mu_s, family, pos=1, neg=0)
        results[tag] = {
            "D": int(mu.shape[1]),
            "probe_r2_pitch": {
                "margin_mean_5seeds": float(margins["mean"]),
                "margin_std_5seeds": float(margins["std"]),
                "control_mean_5seeds": float(ctrls["mean"]),
                "per_seed": margins["per_seed"],
            },
            "knn_timbre": {
                "accuracy_mean_5seeds": float(accs["mean"]),
                "accuracy_std_5seeds": float(accs["std"]),
                "k": 5, "chance": 0.25,
                "per_seed": accs["per_seed"],
            },
            "mig": mig,
            "participation_ratio": participation_ratio(mu),
            "cos_sep": {
                "cos(wp,wf)": float(np.dot(wp, wf)),
                "abs_cos(wp,wf)": float(abs(np.dot(wp, wf))),
            },
        }

    # random cosine baseline (same D for both models in this comparison)
    baselines = {}
    for D in (32,):
        R = rng.standard_normal((100000, D))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        c = np.abs(R[:50000] @ R[50000:].T).diagonal()
        baselines[f"D={D}"] = {"mean": float(c.mean()),
                               "std": float(c.std()),
                               "mean3std": float(c.mean() + 3 * c.std())}
    results["cos_random_baseline"] = baselines

    json.dump(results, open("metrics_comparison.json", "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("Saved metrics_comparison.json")


if __name__ == "__main__":
    main()
