"""
Dataset adapter for DementiaVoiceAI.

Provides an abstract DatasetAdapter interface and a concrete implementation
for the existing dataset structure (celebrity speech recordings organized
by speaker folder with train/valid CSV manifests).

The adapter converts the raw data layout into the project's standardized
internal format (participants DataFrame, recordings DataFrame, splits dict).

Usage:
    from src.data.dataset_adapter import CelebritySpeechAdapter
    adapter = CelebritySpeechAdapter(cfg)
    participants, recordings = adapter.load()
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import DotDict, resolve_path
from src.utils.logger import get_logger
from src.utils.privacy import Anonymizer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class DatasetAdapter(ABC):
    """Abstract interface for loading any dementia-speech dataset.

    Subclass this to support a new data source (e.g., DementiaBank Pitt,
    ADReSS, or the current celebrity speech dataset).
    """

    @abstractmethod
    def validate_source(self) -> bool:
        """Check that the raw data source exists and is valid."""
        ...

    @abstractmethod
    def load_participants(self) -> pd.DataFrame:
        """Load participant/speaker metadata.

        Returns a DataFrame with columns:
            participant_id, group, num_sessions
        """
        ...

    @abstractmethod
    def load_recordings(self) -> pd.DataFrame:
        """Load per-recording metadata.

        Returns a DataFrame with columns:
            recording_id, participant_id, audio_path, group, duration_sec
        """
        ...

    @abstractmethod
    def get_splits(self) -> Dict[str, List[str]]:
        """Return participant-level train/val/test split.

        Returns a dict like:
            {"train": ["SPK_001", ...], "val": [...], "test": [...]}
        """
        ...


# ---------------------------------------------------------------------------
# Concrete adapter for the existing celebrity speech dataset
# ---------------------------------------------------------------------------

class CelebritySpeechAdapter(DatasetAdapter):
    """Adapter for the existing dataset structure.

    Expected layout:
        data/
        ├── dementia/
        │   ├── <SpeakerName>/
        │   │   ├── file1.wav
        │   │   └── file2.wav
        │   └── ...
        ├── non_dementia/
        │   └── n1/nodementia/
        │       ├── <SpeakerName>/
        │       │   └── file.wav
        │       └── ...
        ├── train_dm.csv      (TSV: file, label, path)
        └── valid_dm.csv      (TSV: file, label, path)

    Parameters
    ----------
    cfg : DotDict
        Project configuration (from load_config()).
    anonymize : bool
        If True, speaker names are replaced with anonymous IDs.
    """

    def __init__(self, cfg: DotDict, anonymize: bool = True) -> None:
        self.cfg = cfg
        self.anonymize = anonymize
        self._anonymizer = Anonymizer() if anonymize else None
        self._project_root = resolve_path(".")

    # ----- helpers -----

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a relative path against project root."""
        return self._project_root / rel_path

    def _read_manifest_csv(self, csv_path: Path) -> pd.DataFrame:
        """Read a tab-separated manifest CSV."""
        if not csv_path.exists():
            raise FileNotFoundError(f"Manifest not found: {csv_path}")
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8")
        required_cols = {"file", "label", "path"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Manifest {csv_path.name} missing columns: "
                f"{required_cols - set(df.columns)}"
            )
        return df

    def _extract_speaker_name(self, path_str: str) -> str:
        """Extract speaker name from the Google Drive-style path.

        The path has the form:
            .../data/dementia/SpeakerName/file.wav
        or  .../data/nodementia/SpeakerName/file.wav
        """
        parts = path_str.replace("\\", "/").split("/")
        # Speaker name is the second-to-last component
        if len(parts) >= 2:
            return parts[-2]
        return "Unknown"

    def _normalize_label(self, label: str) -> str:
        """Normalize label to 'dementia' or 'control'."""
        label = label.strip().lower()
        if label in ("dementia",):
            return "dementia"
        elif label in ("nodementia", "no_dementia", "control", "healthy"):
            return "control"
        else:
            logger.warning(f"Unknown label '{label}', treating as 'control'")
            return "control"

    def _find_local_audio(
        self, file_name: str, label: str, speaker_name: str
    ) -> Optional[Path]:
        """Try to find the actual audio file on disk.

        Searches in the local data directories rather than relying on
        Google Drive paths from the CSV.
        """
        # Ensure .wav extension
        if not file_name.endswith(".wav"):
            file_name = file_name + ".wav"

        if label == "dementia":
            base = self._resolve(self.cfg.paths.dementia_audio_dir)
            candidate = base / speaker_name / file_name
            if candidate.exists():
                return candidate
        else:
            # Non-dementia is nested: non_dementia/n1/nodementia/SpeakerName/
            base = self._resolve(self.cfg.paths.non_dementia_audio_dir)
            for sub in base.rglob(file_name):
                return sub

        return None

    # ----- public interface -----

    def validate_source(self) -> bool:
        """Check that the data directories and CSVs exist."""
        issues = []

        train_csv = self._resolve(self.cfg.paths.train_csv)
        valid_csv = self._resolve(self.cfg.paths.valid_csv)
        dem_dir = self._resolve(self.cfg.paths.dementia_audio_dir)
        nodem_dir = self._resolve(self.cfg.paths.non_dementia_audio_dir)

        if not train_csv.exists():
            issues.append(f"Training CSV not found: {train_csv}")
        if not valid_csv.exists():
            issues.append(f"Validation CSV not found: {valid_csv}")
        if not dem_dir.is_dir():
            issues.append(f"Dementia audio dir not found: {dem_dir}")
        if not nodem_dir.is_dir():
            issues.append(f"Non-dementia audio dir not found: {nodem_dir}")

        if issues:
            for issue in issues:
                logger.error(issue)
            return False

        logger.info("Dataset source validation passed.")
        return True

    def load_participants(self) -> pd.DataFrame:
        """Build participant table from the CSV manifests.

        Returns
        -------
        pd.DataFrame
            Columns: participant_id, original_name (if not anonymized),
                     group, num_recordings
        """
        train_df = self._read_manifest_csv(
            self._resolve(self.cfg.paths.train_csv)
        )
        valid_df = self._read_manifest_csv(
            self._resolve(self.cfg.paths.valid_csv)
        )
        all_df = pd.concat([train_df, valid_df], ignore_index=True)

        speakers: Dict[str, Dict] = {}
        for _, row in all_df.iterrows():
            name = self._extract_speaker_name(row["path"])
            if name not in speakers:
                speakers[name] = {
                    "original_name": name,
                    "group": self._normalize_label(row["label"]),
                    "num_recordings": 0,
                }
            speakers[name]["num_recordings"] += 1

        records = []
        for name, info in speakers.items():
            pid = (
                self._anonymizer.anonymize(name)
                if self._anonymizer
                else name
            )
            records.append(
                {
                    "participant_id": pid,
                    "original_name": info["original_name"],
                    "group": info["group"],
                    "num_recordings": info["num_recordings"],
                }
            )

        df = pd.DataFrame(records)

        if self.anonymize:
            # Don't expose original names in the returned DataFrame
            df = df.drop(columns=["original_name"])

        logger.info(
            f"Loaded {len(df)} participants: "
            f"{(df['group'] == 'dementia').sum()} dementia, "
            f"{(df['group'] == 'control').sum()} control"
        )
        return df

    def load_recordings(self) -> pd.DataFrame:
        """Build recordings table from the CSV manifests.

        Returns
        -------
        pd.DataFrame
            Columns: recording_id, participant_id, audio_path, group,
                     split, file_found
        """
        records = []

        for split_name, csv_rel_path in [
            ("train", self.cfg.paths.train_csv),
            ("val", self.cfg.paths.valid_csv),
        ]:
            csv_path = self._resolve(csv_rel_path)
            df = self._read_manifest_csv(csv_path)

            for _, row in df.iterrows():
                speaker = self._extract_speaker_name(row["path"])
                label = self._normalize_label(row["label"])
                pid = (
                    self._anonymizer.anonymize(speaker)
                    if self._anonymizer
                    else speaker
                )
                local_path = self._find_local_audio(
                    row["file"], label, speaker
                )

                records.append(
                    {
                        "recording_id": row["file"],
                        "participant_id": pid,
                        "audio_path": (
                            str(local_path) if local_path else None
                        ),
                        "group": label,
                        "split": split_name,
                        "file_found": local_path is not None,
                    }
                )

        df = pd.DataFrame(records)

        found = df["file_found"].sum()
        total = len(df)
        logger.info(
            f"Loaded {total} recordings, {found} found on disk "
            f"({total - found} missing)"
        )
        return df

    def get_splits(self) -> Dict[str, List[str]]:
        """Return participant-level splits derived from the CSVs.

        The existing CSVs define train and validation sets.
        We use those as-is and note that no test set is yet defined.
        """
        recordings = self.load_recordings()

        splits: Dict[str, List[str]] = {}
        for split_name in recordings["split"].unique():
            mask = recordings["split"] == split_name
            participants = recordings.loc[mask, "participant_id"].unique()
            splits[split_name] = sorted(participants.tolist())

        # Check for participant overlap (leakage!)
        if "train" in splits and "val" in splits:
            overlap = set(splits["train"]) & set(splits["val"])
            if overlap:
                logger.warning(
                    f"⚠️  PARTICIPANT OVERLAP detected between train and val: "
                    f"{len(overlap)} participants appear in both sets!"
                )
            else:
                logger.info(
                    "✓ No participant overlap between train and val."
                )

        for name, pids in splits.items():
            logger.info(f"Split '{name}': {len(pids)} participants")

        return splits
