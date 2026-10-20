"""
Audio preprocessing pipeline for DementiaVoiceAI.

Standardizes all audio to a consistent format:
- Resample to target sample rate (16 kHz)
- Convert to mono
- Normalize amplitude
- Optional noise reduction
- Return clean numpy array ready for feature extraction

Usage:
    from src.preprocessing.audio_preprocessor import AudioPreprocessor
    preprocessor = AudioPreprocessor(cfg)
    audio, sr = preprocessor.process("path/to/audio.wav")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.config import DotDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioPreprocessor:
    """Preprocesses raw audio into a standardized format.

    Parameters
    ----------
    cfg : DotDict
        Project configuration. Uses ``cfg.audio.*`` settings.
    """

    def __init__(self, cfg: DotDict) -> None:
        self.target_sr = cfg.audio.target_sample_rate
        self.target_channels = cfg.audio.target_channels
        self.normalize = cfg.audio.normalize
        self.normalize_method = cfg.audio.get("normalize_method", "peak")
        self.noise_reduction_cfg = cfg.audio.get("noise_reduction", {})

    def process(
        self,
        audio_path: str | Path,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load and preprocess an audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.
        target_sr : int | None
            Override target sample rate.

        Returns
        -------
        tuple[np.ndarray, int]
            (audio_array, sample_rate) — mono, normalized, resampled.
        """
        import librosa

        sr = target_sr or self.target_sr
        path = Path(audio_path)

        logger.debug(f"Processing: {path.name}")

        # 1. Load and resample to target SR, force mono
        audio, sr = librosa.load(str(path), sr=sr, mono=True)

        # 2. Remove DC offset
        audio = audio - np.mean(audio)

        # 3. Normalize
        if self.normalize:
            audio = self._normalize(audio)

        # 4. Optional noise reduction
        if (
            isinstance(self.noise_reduction_cfg, dict)
            and self.noise_reduction_cfg.get("enabled", False)
        ):
            audio = self._reduce_noise(audio, sr)

        logger.debug(
            f"Preprocessed: {path.name} → "
            f"{len(audio)} samples, {sr}Hz, "
            f"duration={len(audio)/sr:.2f}s"
        )

        return audio, sr

    def process_array(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Preprocess an in-memory audio array.

        Parameters
        ----------
        audio : np.ndarray
            Raw audio signal.
        sr : int
            Current sample rate.
        target_sr : int | None
            Override target sample rate.

        Returns
        -------
        tuple[np.ndarray, int]
            (processed_audio, new_sample_rate)
        """
        import librosa

        new_sr = target_sr or self.target_sr

        # Resample if needed
        if sr != new_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
            sr = new_sr

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # DC offset
        audio = audio - np.mean(audio)

        # Normalize
        if self.normalize:
            audio = self._normalize(audio)

        # Noise reduction
        if (
            isinstance(self.noise_reduction_cfg, dict)
            and self.noise_reduction_cfg.get("enabled", False)
        ):
            audio = self._reduce_noise(audio, sr)

        return audio, sr

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.

        Returns
        -------
        np.ndarray
            Normalized audio in [-1, 1].
        """
        if self.normalize_method == "peak":
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
        elif self.normalize_method == "rms":
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1
                audio = audio * (target_rms / rms)
                # Clip to prevent clipping
                audio = np.clip(audio, -1.0, 1.0)
        return audio

    def _reduce_noise(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Apply spectral gating noise reduction.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            Denoised audio.
        """
        try:
            import noisereduce as nr

            prop = self.noise_reduction_cfg.get("prop_decrease", 0.8)
            audio = nr.reduce_noise(
                y=audio, sr=sr, prop_decrease=prop
            )
        except ImportError:
            logger.warning(
                "noisereduce not installed — skipping noise reduction. "
                "Install with: pip install noisereduce"
            )
        return audio
