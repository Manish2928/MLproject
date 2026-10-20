
"""
Voice Activity Detection (VAD) for DementiaVoiceAI.

Detects speech vs. silence segments in audio. Used for:
1. Extracting pause features (count, duration, ratio)
2. Trimming leading/trailing silence
3. Computing speech-to-silence ratio

Usage:
    from src.preprocessing.vad import EnergyVAD
    vad = EnergyVAD(cfg)
    segments = vad.detect(audio, sr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.utils.config import DotDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech or silence segment."""
    start_sec: float
    end_sec: float
    is_speech: bool

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class VADResult:
    """Full VAD analysis result."""
    segments: List[SpeechSegment]
    total_duration_sec: float

    @property
    def speech_segments(self) -> List[SpeechSegment]:
        return [s for s in self.segments if s.is_speech]

    @property
    def silence_segments(self) -> List[SpeechSegment]:
        return [s for s in self.segments if not s.is_speech]

    @property
    def speech_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.speech_segments)

    @property
    def silence_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.silence_segments)

    @property
    def speech_ratio(self) -> float:
        if self.total_duration_sec == 0:
            return 0.0
        return self.speech_duration_sec / self.total_duration_sec

    @property
    def num_pauses(self) -> int:
        return len(self.silence_segments)

    @property
    def mean_pause_duration(self) -> float:
        pauses = self.silence_segments
        if not pauses:
            return 0.0
        return np.mean([p.duration_sec for p in pauses])


class EnergyVAD:
    """Energy-based Voice Activity Detection.

    Classifies frames as speech or silence based on short-time energy.

    Parameters
    ----------
    cfg : DotDict
        Project configuration. Uses ``cfg.audio.vad.*`` settings.
    """

    def __init__(self, cfg: DotDict) -> None:
        vad_cfg = cfg.audio.vad
        self.energy_threshold = vad_cfg.get("energy_threshold", 0.02)
        self.min_speech_duration = vad_cfg.get(
            "min_speech_duration_sec", 0.25
        )
        self.frame_length_ms = 25   # 25ms frames
        self.hop_length_ms = 10     # 10ms hop

    def detect(self, audio: np.ndarray, sr: int) -> VADResult:
        """Run VAD on an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal (mono).
        sr : int
            Sample rate.

        Returns
        -------
        VADResult
            Contains list of speech/silence segments and statistics.
        """
        frame_length = int(sr * self.frame_length_ms / 1000)
        hop_length = int(sr * self.hop_length_ms / 1000)

        # Compute short-time energy
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        if n_frames <= 0:
            return VADResult(segments=[], total_duration_sec=len(audio) / sr)

        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + frame_length]
            energies[i] = np.sqrt(np.mean(frame**2))

        # Classify frames
        is_speech = energies > self.energy_threshold

        # Smooth: apply minimum speech duration
        min_frames = int(
            self.min_speech_duration * 1000 / self.hop_length_ms
        )
        is_speech = self._smooth_decisions(is_speech, min_frames)

        # Convert to segments
        segments = self._frames_to_segments(
            is_speech, hop_length, sr, len(audio)
        )

        total_duration = len(audio) / sr
        result = VADResult(
            segments=segments, total_duration_sec=total_duration
        )

        logger.debug(
            f"VAD: {result.num_pauses} pauses, "
            f"speech ratio={result.speech_ratio:.2%}"
        )

        return result

    @staticmethod
    def _smooth_decisions(
        decisions: np.ndarray, min_length: int
    ) -> np.ndarray:
        """Remove segments shorter than min_length frames."""
        if min_length <= 1:
            return decisions

        smoothed = decisions.copy()
        i = 0
        while i < len(smoothed):
            j = i
            while j < len(smoothed) and smoothed[j] == smoothed[i]:
                j += 1
            segment_len = j - i
            if segment_len < min_length:
                # Flip short segments
                smoothed[i:j] = not smoothed[i]
            i = j

        return smoothed

    @staticmethod
    def _frames_to_segments(
        is_speech: np.ndarray,
        hop_length: int,
        sr: int,
        total_samples: int,
    ) -> List[SpeechSegment]:
        """Convert frame-level decisions to time-based segments."""
        segments = []
        if len(is_speech) == 0:
            return segments

        current_state = is_speech[0]
        start_frame = 0

        for i in range(1, len(is_speech)):
            if is_speech[i] != current_state:
                start_sec = start_frame * hop_length / sr
                end_sec = i * hop_length / sr
                segments.append(
                    SpeechSegment(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        is_speech=bool(current_state),
                    )
                )
                current_state = is_speech[i]
                start_frame = i

        # Final segment
        start_sec = start_frame * hop_length / sr
        end_sec = total_samples / sr
        segments.append(
            SpeechSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                is_speech=bool(current_state),
            )
        )

        return segments

    def trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: float = 30.0,
    ) -> np.ndarray:
        """Trim leading and trailing silence from audio.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal.
        sr : int
            Sample rate.
        top_db : float
            Threshold in dB below peak to consider as silence.

        Returns
        -------
        np.ndarray
            Trimmed audio.
        """
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
