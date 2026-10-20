"""Tests for the audio preprocessor and VAD."""

import numpy as np

from src.preprocessing.vad import EnergyVAD


class TestEnergyVAD:
    """Test suite for energy-based VAD."""

    def test_detect_speech_in_audio(self, cfg, sample_audio):
        """VAD should detect speech in a non-silent signal."""
        audio, sr = sample_audio
        vad = EnergyVAD(cfg)
        result = vad.detect(audio, sr)

        assert result.total_duration_sec > 0
        assert result.speech_ratio > 0

    def test_detect_silence(self, cfg, silent_audio):
        """VAD should detect silence in a silent signal."""
        audio, sr = silent_audio
        vad = EnergyVAD(cfg)
        result = vad.detect(audio, sr)

        # Silent audio should have very low speech ratio
        assert result.speech_ratio < 0.1
