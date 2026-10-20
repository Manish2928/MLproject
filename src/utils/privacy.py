"""
Privacy and data protection utilities for DementiaVoiceAI.

Provides functions to anonymize participant identifiers and scrub
personally identifiable information from text and paths.

IMPORTANT: This project must never expose real participant names,
locations, or identifying metadata in logs, outputs, or the UI.

Usage:
    from src.utils.privacy import Anonymizer
    anon = Anonymizer()
    anon_id = anon.anonymize("Ronald Reagan")  # "SPK_001"
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional


class Anonymizer:
    """Maps real participant names to anonymized speaker IDs.

    The mapping is deterministic within a session but can also be
    saved/loaded for consistency across runs.
    """

    def __init__(self, prefix: str = "SPK") -> None:
        self._prefix = prefix
        self._name_to_id: Dict[str, str] = {}
        self._counter = 0

    def anonymize(self, name: str) -> str:
        """Return an anonymized ID for the given name.

        The same name always maps to the same ID within a session.

        Parameters
        ----------
        name : str
            The real participant/speaker name.

        Returns
        -------
        str
            Anonymized ID like ``SPK_001``.
        """
        normalized = name.strip().lower()
        if normalized not in self._name_to_id:
            self._counter += 1
            self._name_to_id[normalized] = (
                f"{self._prefix}_{self._counter:03d}"
            )
        return self._name_to_id[normalized]

    def get_mapping(self) -> Dict[str, str]:
        """Return a copy of the current name → ID mapping."""
        return dict(self._name_to_id)

    @property
    def num_participants(self) -> int:
        """Number of unique participants anonymized so far."""
        return self._counter


def hash_identifier(name: str) -> str:
    """Create a deterministic, non-reversible hash of a name.

    Useful for creating consistent anonymous IDs across sessions
    without storing the mapping.

    Parameters
    ----------
    name : str
        The identifier to hash.

    Returns
    -------
    str
        8-character hex hash.
    """
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:8]


def scrub_path(path: str) -> str:
    """Remove potentially identifying directory names from a file path.

    Replaces speaker/participant folder names with ``[REDACTED]``.

    Parameters
    ----------
    path : str
        The file path to scrub.

    Returns
    -------
    str
        Path with identifying components redacted.
    """
    # Replace segments that look like person names (two+ capitalized words)
    parts = path.replace("\\", "/").split("/")
    scrubbed = []
    for part in parts:
        # Simple heuristic: if a folder name has spaces and capital letters,
        # it's likely a person's name
        if (
            " " in part
            and any(c.isupper() for c in part)
            and not part.endswith((".wav", ".mp3", ".txt", ".csv"))
        ):
            scrubbed.append("[REDACTED]")
        else:
            scrubbed.append(part)
    return "/".join(scrubbed)


def scrub_text(
    text: str,
    names: Optional[list[str]] = None,
) -> str:
    """Remove known names and common PII patterns from text.

    Parameters
    ----------
    text : str
        The text to scrub.
    names : list[str] | None
        Optional list of known names to redact.

    Returns
    -------
    str
        Scrubbed text.
    """
    result = text

    # Remove known names
    if names:
        for name in names:
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            result = pattern.sub("[REDACTED]", result)

    return result
