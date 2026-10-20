"""
Shared test fixtures for DementiaVoiceAI tests.

Provides reusable fixtures: config, mock audio, mock datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.config import load_config, DotDict


@pytest.fixture
def cfg() -> DotDict:
    """Load the default project configuration."""
    return load_config()


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate a short synthetic audio signal for testing.

    Returns a 2-second sine wave at 440 Hz, sampled at 16 kHz.
    """
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 440 Hz sine + some noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio += 0.05 * np.random.randn(len(audio))

    return audio.astype(np.float32), sr


@pytest.fixture
def silent_audio() -> tuple[np.ndarray, int]:
    """Generate 1 second of silence."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    return audio, sr


@pytest.fixture
def sample_participants_df():
    """Create a small mock participants DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "participant_id": [f"SPK_{i:03d}" for i in range(1, 11)],
            "group": (
                ["dementia"] * 5 + ["control"] * 5
            ),
            "num_recordings": [3, 2, 1, 2, 3, 2, 1, 3, 2, 1],
        }
    )
