"""
SAMI-Audio — NSynth Dataset Module
===================================
PyTorch Dataset for loading, preprocessing, and filtering the NSynth dataset.

Provides mel-spectrogram extraction from raw .wav files with fixed preprocessing
parameters.

Usage:
    dataset = NSynthDataset(root="data/nsynth-train")
    mel, meta = dataset[0]        # mel: (1, 128, 256) in [-1, 1]
    audio = mel_to_audio(mel)     # Griffin-Lim reconstruction

Dataset sanity check:
    python -m data.nsynth         # runs from project root
    pdm run gate0                 # via PDM script alias
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants: instrument family mapping
# ---------------------------------------------------------------------------

NSYNTH_FAMILY_MAP: Dict[int, str] = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth_lead",
    10: "vocal",
}

FILTER_FAMILIES: List[str] = ["guitar", "keyboard", "string", "brass"]
FILTER_PITCH_MIN: int = 48
FILTER_PITCH_MAX: int = 84
NSYNTH_SAMPLE_RATE: int = 16000


# ---------------------------------------------------------------------------
# Mel spectrogram configuration (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MelConfig:
    """
    Immutable configuration for mel-spectrogram preprocessing.

    All parameters are fixed.
    Shape invariant: output is always (1, n_mels, n_frames) = (1, 128, 256).
    """

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 8000.0
    n_frames: int = 256          # target time dimension (crop or pad)
    power: float = 1.0           # magnitude spectrogram (not power)

    # Pre-constructed transforms (created lazily per MelConfig instance)
    mel_transform: torchaudio.transforms.MelSpectrogram = field(init=False)
    inverse_mel: torchaudio.transforms.InverseMelScale = field(init=False)
    griffin_lim: torchaudio.transforms.GriffinLim = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mel_transform",
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                power=self.power,
                normalized=False,
            ),
        )
        object.__setattr__(
            self,
            "inverse_mel",
            torchaudio.transforms.InverseMelScale(
                n_stft=self.n_fft // 2 + 1,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                f_min=self.f_min,
                f_max=self.f_max,
            ),
        )
        object.__setattr__(
            self,
            "griffin_lim",
            torchaudio.transforms.GriffinLim(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                power=self.power,
                n_iter=32,
            ),
        )

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return (1, self.n_mels, self.n_frames)


# Global default config
MEL_CONFIG = MelConfig()


# ---------------------------------------------------------------------------
# NSynth Dataset
# ---------------------------------------------------------------------------

class NSynthDataset(Dataset):
    """
    PyTorch Dataset for the NSynth subset.

    Loads raw .wav files, computes log-mel spectrograms, normalizes to [-1, 1],
    and crops/pads to a fixed time dimension.

    Parameters
    ----------
    root : str
        Path to the NSynth directory containing `examples.json` and `audio/`.

    mel_config : MelConfig, optional
        Fixed mel-spectrogram configuration. Defaults to global MEL_CONFIG.

    Attributes
    ----------
    labels : List[Dict[str, object]]
        Per-sample metadata dictionaries. Available for evaluation only.
        Contains: 'pitch', 'instrument_family', 'instrument_family_str',
        'instrument_source', 'qualities', 'velocity', 'sample_rate', 'note_str'.

    family_counts : Counter
        Number of samples per instrument family in the filtered set.

    pitch_counts : Counter
        Number of samples per MIDI pitch in the filtered set.
    """

    def __init__(
        self,
        root: str = "data/nsynth-train",
        mel_config: MelConfig = MEL_CONFIG,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.mel_config = mel_config
        self.normalize = normalize

        # Load global normalization stats if available
        self._norm_stats = None
        stats_path = os.path.join(os.path.dirname(root), "norm_stats.json")
        if os.path.isfile(stats_path):
            import json as _json
            self._norm_stats = _json.load(open(stats_path))

        json_path = os.path.join(root, "examples.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"examples.json not found at {json_path}. "
                f"Run 'bash data/download.sh' first to download the NSynth dataset."
            )

        with open(json_path, "r") as f:
            self._raw_data: Dict[str, dict] = json.load(f)

        self._index: List[str] = []        # filtered entry IDs
        self._family_map: Dict[str, int] = {v: k for k, v in NSYNTH_FAMILY_MAP.items()}

        self._build_index()

    # ---- index construction ---------------------------------------------------

    def _build_index(self) -> None:
        """
        Scan all entries in examples.json and keep only those matching:
          - instrument_family in FILTER_FAMILIES
          - pitch in [FILTER_PITCH_MIN, FILTER_PITCH_MAX]
        """
        self.labels: List[Dict[str, object]] = []
        self.family_counts: Counter[str] = Counter()
        self.pitch_counts: Counter[int] = Counter()

        for entry_id, entry in self._raw_data.items():
            family = self._resolve_family(entry)
            pitch = self._resolve_pitch(entry)

            if family is None or pitch is None:
                continue
            if family not in FILTER_FAMILIES:
                continue
            if not (FILTER_PITCH_MIN <= pitch <= FILTER_PITCH_MAX):
                continue

            self._index.append(entry_id)

            label_dict: Dict[str, object] = {
                "entry_id": entry_id,
                "pitch": pitch,
                "instrument_family": family,
                "instrument_family_int": self._family_map.get(family, -1),
                "instrument_source": entry.get("instrument_source", -1),
                "qualities": entry.get("qualities", []),
                "velocity": entry.get("velocity", -1),
                "sample_rate": entry.get("sample_rate", NSYNTH_SAMPLE_RATE),
                "note_str": entry.get("note_str", ""),
            }
            self.labels.append(label_dict)
            self.family_counts[family] += 1
            self.pitch_counts[pitch] += 1

    # ---- data access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, object]]:
        """
        Returns a tuple (mel_tensor, metadata_dict).

        mel_tensor: shape (1, 128, 256), values in [-1, 1]
        metadata_dict: labels for this sample (for evaluation only)
        """
        entry_id = self._index[idx]
        meta = self.labels[idx]

        wav_path = os.path.join(self.root, "audio", f"{entry_id}.wav")
        waveform, sr = sf.read(wav_path)
        waveform = torch.from_numpy(waveform.T).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.mel_config.sample_rate:
            waveform = F.resample(waveform, sr, self.mel_config.sample_rate)

        mel = self._waveform_to_mel(waveform)
        mel = self._crop_or_pad_time(mel)
        if self.normalize:
            mel = self._normalize_global(mel) if self._norm_stats else self._normalize_per_sample(mel)

        return mel, meta

    # ---- preprocessing helpers ------------------------------------------------

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Input:  (1, num_samples)
        Output: (1, n_mels, num_frames)
        """
        mel = self.mel_config.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def _crop_or_pad_time(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Crop or pad the time dimension to exactly n_frames.

        If longer than n_frames: center-crop.
        If shorter than n_frames: zero-pad both sides symmetrically.
        """
        target = self.mel_config.n_frames
        current = mel.shape[-1]

        if current > target:
            start = (current - target) // 2
            mel = mel[..., start : start + target]
        elif current < target:
            pad_total = target - current
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            mel = torch.nn.functional.pad(
                mel, (pad_left, pad_right), mode="constant",
                value=math.log(1e-5)  # silence floor in log-mel space (~ -11.5)
            )

        return mel

    @staticmethod
    def _normalize_per_sample(mel: torch.Tensor) -> torch.Tensor:
        """
        Per-sample min-max normalization to [-1, 1].

        If all values are equal (zero variance), returns a zero tensor.
        """
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max > mel_min:
            mel = 2.0 * (mel - mel_min) / (mel_max - mel_min) - 1.0
        else:
            mel = torch.zeros_like(mel)
        return mel

    def _normalize_global(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Global min-max normalization to [-1, 1] using pre-computed constants.

        Fixed percentiles (robust to outliers) loaded from data/norm_stats.json.
        Invertible — preserves dynamics and compatible with Tanh decoder.
        """
        lo = self._norm_stats["log_mel_min"]
        hi = self._norm_stats["log_mel_max"]
        mel = torch.clamp(mel, lo, hi)
        return 2.0 * (mel - lo) / (hi - lo) - 1.0

    # ---- metadata resolution --------------------------------------------------

    @staticmethod
    def _resolve_family(entry: dict) -> Optional[str]:
        """Extract instrument family as a lowercase string, or None on failure."""
        family = entry.get("instrument_family_str", entry.get("instrument_family", ""))
        if isinstance(family, int):
            family = NSYNTH_FAMILY_MAP.get(family, str(family))
        if not family:
            return None
        return str(family).lower().strip()

    @staticmethod
    def _resolve_pitch(entry: dict) -> Optional[int]:
        """Extract MIDI pitch as int, or None on failure."""
        pitch = entry.get("pitch", entry.get("note", -1))
        if isinstance(pitch, str):
            try:
                pitch = int(pitch)
            except (ValueError, TypeError):
                return None
        if not isinstance(pitch, (int, float)):
            return None
        return int(pitch)

    # ---- utilities ------------------------------------------------------------

    def class_distribution(self) -> Dict[str, Counter]:
        """
        Return per-family and per-pitch sample counts.

        Returns a dict with keys 'family' and 'pitch'.
        """
        return {
            "family": self.family_counts,
            "pitch": self.pitch_counts,
        }

    def __repr__(self) -> str:
        return (
            f"NSynthDataset(root='{self.root}', "
            f"samples={len(self)}, "
            f"families={dict(self.family_counts)}, "
            f"pitch_range=[{min(self.pitch_counts.keys())}, {max(self.pitch_counts.keys())}], "
            f"shape={self.mel_config.output_shape})"
        )


# ---------------------------------------------------------------------------
# Griffin-Lim reconstruction (for sanity checks and demo baseline)
# ---------------------------------------------------------------------------

def mel_to_audio(
    mel: torch.Tensor,
    mel_config: MelConfig = MEL_CONFIG,
) -> torch.Tensor:
    """
    Convert a log-mel spectrogram back to an audio waveform via Griffin-Lim.

    Parameters
    ----------
    mel : torch.Tensor
        Log-mel spectrogram of shape (1, n_mels, n_frames) in [-1, 1].
    mel_config : MelConfig
        Configuration with matching mel parameters.

    Returns
    -------
    torch.Tensor
        Reconstructed waveform of shape (1, num_samples).
    """
    import json as _json
    import os as _os
    stats_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "norm_stats.json")
    if _os.path.isfile(stats_path):
        stats = _json.load(open(stats_path))
        lo, hi = stats["log_mel_min"], stats["log_mel_max"]
        mel = (mel + 1.0) / 2.0 * (hi - lo) + lo  # invert global normalization
    mel_linear = torch.exp(torch.clamp(mel, min=-10.0, max=10.0))

    lin_spec = mel_config.inverse_mel(mel_linear)
    audio = mel_config.griffin_lim(lin_spec)
    return audio


# ===========================================================================
# Dataset sanity check
# ===========================================================================

if __name__ == "__main__":
    """
    Sanity check: validates the full preprocessing pipeline.

    Checks:
      1. Dataset loads without errors
      2. Every sample has shape (1, 128, 256) and values in [-1, 1]
      3. Griffin-Lim reconstruction produces audible output
      4. Class distribution is printed for manual inspection
    """
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/nsynth-train"

    print("=" * 60)
    print("  SAMI-Audio — Dataset Sanity Check")
    print("=" * 60)
    print()

    if not os.path.isdir(root):
        print(f"[ERROR] Directory not found: {root}")
        print(f"        Run 'bash data/download.sh' first to download NSynth.")
        sys.exit(1)

    print(f"[INFO]  Loading dataset from: {root}")
    dataset = NSynthDataset(root=root)
    print(f"[INFO]  {dataset}")
    print()

    # Distribution
    dist = dataset.class_distribution()
    print("[CHECK] Class distribution:")
    print(f"  Families: {dict(dist['family'])}")
    print(f"  Pitch range: {min(dist['pitch'].keys())} - {max(dist['pitch'].keys())}")
    print(f"  Unique pitches: {len(dist['pitch'])}")
    print()

    # Shape and range check (random subset — full check would be too slow)
    NUM_SAMPLES_CHECK = 200
    print(f"[CHECK] Validating shape and value range on {NUM_SAMPLES_CHECK} random samples...")
    shape_errors = 0
    range_errors = 0
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(NUM_SAMPLES_CHECK, len(dataset)), replace=False)

    for i in indices:
        mel, meta = dataset[int(i)]
        if mel.shape != (1, 128, 256):
            shape_errors += 1
            print(f"  [ERROR] Sample {i}: expected shape (1,128,256), got {mel.shape}")
        if mel.min() < -1.0 or mel.max() > 1.0:
            range_errors += 1
            print(f"  [ERROR] Sample {i}: values out of [-1, 1] "
                  f"(min={mel.min():.4f}, max={mel.max():.4f})")

    if shape_errors == 0 and range_errors == 0:
        print(f"  [PASS]  All {NUM_SAMPLES_CHECK} samples: shape (1, 128, 256), values in [-1, 1]")
    else:
        print(f"  [FAIL]  {shape_errors} shape errors, {range_errors} range errors")
    print()

    # Audio reconstruction
    print("[CHECK] Griffin-Lim reconstruction test...")
    mel, meta = dataset[np.random.default_rng(42).integers(0, len(dataset))]
    recon = mel_to_audio(mel, mel_config=MEL_CONFIG)

    output_path = "data/gate0_test.wav"
    sf.write(output_path, recon.squeeze(0).numpy().T, MEL_CONFIG.sample_rate)
    print(f"  [INFO]  Original: pitch={meta['pitch']}, "
          f"instrument={meta['instrument_family']}")
    print(f"  [INFO]  Reconstructed audio saved to: {output_path}")
    print()

    # Final verdict
    if shape_errors == 0 and range_errors == 0:
        print("=" * 60)
        print("  SANITY CHECK: PASSED")
        print("=" * 60)
        print()
        print("  Dataset is ready.")
    else:
        print("=" * 60)
        print("  SANITY CHECK: FAILED")
        print("=" * 60)
        sys.exit(1)


# ===========================================================================
# Cached Mel Dataset (for fast GPU training)
# ===========================================================================

class CachedMelDataset(torch.utils.data.Dataset):
    """
    Dataset that reads pre-computed mel spectrograms from a memory-mapped
    .npy file, eliminating the .wav → mel bottleneck (~14 min/epoch).

    Use after running scripts/precompute_mels.py.

    Parameters
    ----------
    cache : str
        Path to mel_cache.npy (memmap).
    meta_path : str
        Path to mel_meta.pkl (pickled metadata list).
    """

    def __init__(self, cache: str = "data/mel_cache.npy",
                 meta_path: str = "data/mel_meta.pkl") -> None:
        import pickle as _pickle
        self.arr = np.load(cache, mmap_mode="r")
        self.metas = _pickle.load(open(meta_path, "rb"))

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.arr[idx].copy()), self.metas[idx]
