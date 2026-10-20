"""
Audio validation module for DementiaVoiceAI.

Validates audio files before processing to catch issues early:
- File existence and readability
- Format support
- Duration within acceptable range
- Sample rate adequacy
- Corruption detection (can the file be loaded?)

Usage:
    from src.preprocessing.audio_validator import AudioValidator
    validator = AudioValidator(cfg)
    result = validator.validate("path/to/audio.wav")
    if result.is_valid:
        print("OK")
    else:
        print(result.issues)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.utils.config import DotDict
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class ValidationResult:
    """Result of validating a single audio file."""

    path: str
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    num_channels: Optional[int] = None
    num_samples: Optional[int] = None

    def add_issue(self, issue: str) -> None:
        """Add a validation issue and mark as invalid."""
        self.issues.append(issue)
        self.is_valid = False


class AudioValidator:
    """Validates audio files for the DementiaVoiceAI pipeline.

    Parameters
    ----------
    cfg : DotDict
        Project configuration. Uses ``cfg.audio.*`` settings.
    """

    def __init__(self, cfg: DotDict) -> None:
        self.min_duration = cfg.audio.min_duration_sec
        self.max_duration = cfg.audio.max_duration_sec
        self.min_sample_rate = 8000  # Minimum acceptable SR

    def validate(self, audio_path: str | Path) -> ValidationResult:
        """Run all validation checks on an audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        ValidationResult
            Contains validity flag, issues list, and audio metadata.
        """
        path = Path(audio_path)
        result = ValidationResult(path=str(path))

        # 1. File existence
        if not path.exists():
            result.add_issue(f"File not found: {path}")
            return result

        # 2. File is not empty
        if path.stat().st_size == 0:
            result.add_issue("File is empty (0 bytes)")
            return result

        # 3. Format check
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            result.add_issue(
                f"Unsupported format '{suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )
            return result

        # 4. Try to load the file
        try:
            import librosa
            audio, sr = librosa.load(str(path), sr=None, mono=False)
        except Exception as e:
            result.add_issue(f"Failed to load audio: {e}")
            return result

        # 5. Populate metadata
        if audio.ndim == 1:
            result.num_channels = 1
            result.num_samples = len(audio)
        else:
            result.num_channels = audio.shape[0]
            result.num_samples = audio.shape[1]

        result.sample_rate = sr
        result.duration_sec = result.num_samples / sr

        # 6. Duration check
        if result.duration_sec < self.min_duration:
            result.add_issue(
                f"Audio too short: {result.duration_sec:.2f}s "
                f"(minimum: {self.min_duration}s)"
            )

        if result.duration_sec > self.max_duration:
            result.add_issue(
                f"Audio too long: {result.duration_sec:.2f}s "
                f"(maximum: {self.max_duration}s)"
            )

        # 7. Sample rate check
        if sr < self.min_sample_rate:
            result.add_issue(
                f"Sample rate too low: {sr} Hz "
                f"(minimum: {self.min_sample_rate} Hz)"
            )

        # 8. Check for silence / all-zeros
        if np.max(np.abs(audio)) < 1e-6:
            result.add_issue("Audio appears to be silent (all zeros)")

        # 9. Check for NaN/Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            result.add_issue("Audio contains NaN or Inf values")

        if result.is_valid:
            logger.debug(
                f"✓ Valid: {path.name} "
                f"({result.duration_sec:.1f}s, {sr}Hz, "
                f"{result.num_channels}ch)"
            )
        else:
            logger.warning(
                f"✗ Invalid: {path.name} — {result.issues}"
            )

        return result

    def validate_batch(
        self, audio_paths: List[str | Path]
    ) -> List[ValidationResult]:
        """Validate multiple audio files.

        Parameters
        ----------
        audio_paths : list
            List of paths to validate.

        Returns
        -------
        list[ValidationResult]
            One result per file.
        """
        results = []
        valid_count = 0

        for path in audio_paths:
            result = self.validate(path)
            results.append(result)
            if result.is_valid:
                valid_count += 1

        total = len(results)
        logger.info(
            f"Batch validation: {valid_count}/{total} files valid "
            f"({total - valid_count} issues)"
        )
        return results
