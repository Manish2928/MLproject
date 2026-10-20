"""
Acoustic feature extraction for DementiaVoiceAI.

Extracts a comprehensive set of audio features from a speech recording:
- MFCCs (13 coefficients) with delta and delta-delta
- Spectral features: centroid, bandwidth, rolloff, contrast, flatness
- Chroma features (12 pitch classes)
- Energy / RMS statistics
- Zero-crossing rate
- Pitch (pYIN) statistics
- Pause features derived from Voice Activity Detection

Inspired by the demo notebook's extract_features() function, adapted to
integrate with the project's AudioPreprocessor and EnergyVAD modules.

Usage:
    from src.features.audio_features import AudioFeatureExtractor
    extractor = AudioFeatureExtractor(cfg)
    features = extractor.extract("path/to/audio.wav")
    # features is a dict of {feature_name: float}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.config import DotDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean_std(arr: np.ndarray, prefix: str, out: dict) -> None:
    """Compute mean and std for each row of a 2D array (or 1D → single row).

    For a matrix with shape (n_features, n_frames), produces:
        {prefix}_1_mean, {prefix}_1_std, {prefix}_2_mean, ...
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for i, row in enumerate(arr):
        out[f"{prefix}_{i+1}_mean"] = float(np.nanmean(row))
        out[f"{prefix}_{i+1}_std"] = float(np.nanstd(row))


def _safe_scalar(value, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class AudioFeatureExtractor:
    """Extracts acoustic features from audio files.

    Parameters
    ----------
    cfg : DotDict
        Project configuration. Uses ``cfg.audio.*`` and ``cfg.features.*``.
    """

    def __init__(self, cfg: DotDict) -> None:
        self.cfg = cfg
        self.sr = cfg.audio.target_sample_rate  # 16000
        self.max_duration = cfg.audio.get("max_duration_sec", 600)

        # Feature config
        feat_cfg = cfg.features.acoustic
        self.n_mfcc = feat_cfg.mfcc.get("n_mfcc", 13)
        self.n_fft = feat_cfg.mfcc.get("n_fft", 2048)
        self.hop_length = feat_cfg.mfcc.get("hop_length", 512)
        self.include_delta = feat_cfg.mfcc.get("include_delta", True)
        self.include_delta2 = feat_cfg.mfcc.get("include_delta2", True)
        self.n_mels = feat_cfg.mel_spectrogram.get("n_mels", 128)

        # VAD config for pause features
        self.vad_enabled = cfg.audio.vad.get("enabled", True)

    def extract(self, audio_path: str | Path) -> Dict[str, float]:
        """Extract all acoustic features from a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file (WAV, MP3, FLAC, etc.).

        Returns
        -------
        dict[str, float]
            Flat dictionary mapping feature names to scalar values.
            Typically ~130+ features.

        Raises
        ------
        ValueError
            If the audio is shorter than 0.25 seconds.
        """
        import librosa

        path = str(Path(audio_path).resolve())
        y, sr = librosa.load(path, sr=self.sr, mono=True,
                             duration=self.max_duration)

        if y.size < int(0.25 * sr):
            raise ValueError(
                f"Audio is shorter than 0.25 seconds: {Path(audio_path).name}"
            )

        # Clean the signal
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        out: Dict[str, float] = {}

        # ----- Signal-level statistics -----
        out["duration_sec"] = float(len(y) / sr)
        out["rms_global"] = float(np.sqrt(np.mean(y ** 2)))
        out["peak_amplitude"] = float(np.max(np.abs(y)))
        out["near_silence_fraction"] = float(np.mean(np.abs(y) < 1e-4))

        # ----- Zero Crossing Rate -----
        zcr = librosa.feature.zero_crossing_rate(y=y)
        out["zero_crossing_rate_mean"] = float(np.mean(zcr))
        out["zero_crossing_rate_std"] = float(np.std(zcr))

        # ----- Spectral features -----
        for name, func in [
            ("spectral_centroid", librosa.feature.spectral_centroid),
            ("spectral_bandwidth", librosa.feature.spectral_bandwidth),
            ("spectral_rolloff", librosa.feature.spectral_rolloff),
        ]:
            arr = func(y=y, sr=sr)
            out[f"{name}_mean"] = float(np.mean(arr))
            out[f"{name}_std"] = float(np.std(arr))

        # Spectral contrast (7 frequency bands)
        try:
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_bands=6
            )
            _safe_mean_std(contrast, "spectral_contrast", out)
        except Exception:
            for i in range(7):
                out[f"spectral_contrast_{i+1}_mean"] = 0.0
                out[f"spectral_contrast_{i+1}_std"] = 0.0

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        out["spectral_flatness_mean"] = float(np.mean(flatness))
        out["spectral_flatness_std"] = float(np.std(flatness))

        # ----- RMS frame-level -----
        rms_frame = librosa.feature.rms(y=y)
        out["rms_frame_mean"] = float(np.mean(rms_frame))
        out["rms_frame_std"] = float(np.std(rms_frame))

        # ----- MFCCs -----
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        _safe_mean_std(mfcc, "mfcc", out)

        # MFCC deltas
        if self.include_delta:
            delta = librosa.feature.delta(mfcc, order=1)
            _safe_mean_std(delta, "mfcc_delta", out)

        # MFCC delta-deltas
        if self.include_delta2:
            delta2 = librosa.feature.delta(mfcc, order=2)
            _safe_mean_std(delta2, "mfcc_delta2", out)

        # ----- Chroma -----
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        _safe_mean_std(chroma, "chroma", out)

        # ----- Pitch (pYIN) -----
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                out["pitch_mean"] = float(np.mean(f0_valid))
                out["pitch_std"] = float(np.std(f0_valid))
                out["pitch_min"] = float(np.min(f0_valid))
                out["pitch_max"] = float(np.max(f0_valid))
                out["pitch_range"] = out["pitch_max"] - out["pitch_min"]
            else:
                for k in ["pitch_mean", "pitch_std", "pitch_min",
                          "pitch_max", "pitch_range"]:
                    out[k] = 0.0
            out["voiced_fraction"] = float(
                np.sum(voiced_flag) / len(voiced_flag)
            ) if len(voiced_flag) > 0 else 0.0
        except Exception:
            for k in ["pitch_mean", "pitch_std", "pitch_min",
                      "pitch_max", "pitch_range", "voiced_fraction"]:
                out[k] = 0.0

        # ----- Pause features (from VAD) -----
        if self.vad_enabled:
            try:
                from src.preprocessing.vad import EnergyVAD
                vad = EnergyVAD(self.cfg)
                vad_result = vad.detect(y, sr)
                out["speech_ratio"] = _safe_scalar(vad_result.speech_ratio)
                out["num_pauses"] = _safe_scalar(vad_result.num_pauses)
                out["mean_pause_duration"] = _safe_scalar(
                    vad_result.mean_pause_duration
                )
                out["speech_duration_sec"] = _safe_scalar(
                    vad_result.speech_duration_sec
                )
                out["silence_duration_sec"] = _safe_scalar(
                    vad_result.silence_duration_sec
                )
            except Exception as e:
                logger.debug(f"VAD pause features failed: {e}")
                for k in ["speech_ratio", "num_pauses",
                          "mean_pause_duration", "speech_duration_sec",
                          "silence_duration_sec"]:
                    out[k] = 0.0

        return out

    def extract_batch(
        self,
        audio_paths: List[str | Path],
        labels: Optional[List[int]] = None,
        speaker_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract features from multiple audio files.

        Parameters
        ----------
        audio_paths : list[str | Path]
            Paths to audio files.
        labels : list[int] | None
            Optional binary labels (1=dementia, 0=control).
        speaker_ids : list[str] | None
            Optional speaker IDs (for grouping in splits).

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per audio file.
            Columns prefixed with 'feat_' for model features.
            Also includes 'audio_path', 'label', 'speaker_id' metadata.
        """
        feature_rows = []
        errors = []

        for idx, path in enumerate(
            tqdm(audio_paths, desc="Extracting features")
        ):
            try:
                feats = self.extract(path)
                row = {f"feat_{k}": v for k, v in feats.items()}
                row["audio_path"] = str(path)
                if labels is not None:
                    row["label"] = labels[idx]
                if speaker_ids is not None:
                    row["speaker_id"] = speaker_ids[idx]
                feature_rows.append(row)
            except Exception as e:
                errors.append({"audio_path": str(path), "error": str(e)[:250]})
                logger.warning(f"Feature extraction failed for {Path(path).name}: {e}")

        if errors:
            logger.warning(
                f"Feature extraction failed for {len(errors)} / "
                f"{len(audio_paths)} files"
            )

        df = pd.DataFrame(feature_rows)
        logger.info(
            f"Feature extraction complete: {len(df)} rows, "
            f"{len([c for c in df.columns if c.startswith('feat_')])} features"
        )
        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Return the list of feature columns (prefixed with 'feat_').

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame from extract_batch().

        Returns
        -------
        list[str]
            Sorted list of feature column names.
        """
        return sorted([
            c for c in df.columns
            if c.startswith("feat_") and df[c].dtype in [np.float64, np.int64, float, int]
        ])
