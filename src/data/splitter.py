"""
Participant-level stratified data splitter for DementiaVoiceAI.

Ensures that ALL recordings from a single participant stay in the same
split (train / val / test) to prevent data leakage.

Usage:
    from src.data.splitter import ParticipantSplitter
    splitter = ParticipantSplitter(seed=42)
    splits = splitter.split(participants_df, ratios=(0.7, 0.15, 0.15))
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParticipantSplitter:
    """Stratified participant-level splitter.

    Groups participants by class label, then splits each group according
    to the specified ratios, so that class proportions are preserved
    across train/val/test.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split(
        self,
        participants: pd.DataFrame,
        ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        group_col: str = "group",
        id_col: str = "participant_id",
    ) -> Dict[str, List[str]]:
        """Split participants into train/val/test sets.

        Parameters
        ----------
        participants : pd.DataFrame
            Must contain at least ``id_col`` and ``group_col``.
        ratios : tuple of 3 floats
            (train_ratio, val_ratio, test_ratio). Must sum to ~1.0.
        group_col : str
            Column name for the class label.
        id_col : str
            Column name for the participant identifier.

        Returns
        -------
        dict
            {"train": [...], "val": [...], "test": [...]}
        """
        train_r, val_r, test_r = ratios
        assert abs(train_r + val_r + test_r - 1.0) < 1e-6, (
            f"Ratios must sum to 1.0, got {train_r + val_r + test_r}"
        )

        rng = np.random.RandomState(self.seed)

        train_ids, val_ids, test_ids = [], [], []

        for group_name, group_df in participants.groupby(group_col):
            ids = group_df[id_col].values.copy()
            rng.shuffle(ids)

            n = len(ids)
            n_train = max(1, int(round(n * train_r)))
            n_val = max(1, int(round(n * val_r)))
            # Test gets the remainder
            n_test = n - n_train - n_val

            if n_test < 0:
                # Very small group — put everything in train
                n_train = n
                n_val = 0
                n_test = 0

            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train : n_train + n_val])
            test_ids.extend(ids[n_train + n_val :])

            logger.info(
                f"Group '{group_name}': {n} participants → "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        splits = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }

        # Verify no overlap
        self._verify_no_overlap(splits)

        return splits

    @staticmethod
    def _verify_no_overlap(splits: Dict[str, List[str]]) -> None:
        """Raise an error if any participant appears in multiple splits."""
        names = list(splits.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = set(splits[names[i]]) & set(splits[names[j]])
                if overlap:
                    raise ValueError(
                        f"DATA LEAKAGE: {len(overlap)} participants appear "
                        f"in both '{names[i]}' and '{names[j]}': "
                        f"{list(overlap)[:5]}..."
                    )
        logger.info("✓ Participant split verification passed — no overlap.")

    def split_from_recordings(
        self,
        recordings: pd.DataFrame,
        ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        group_col: str = "group",
        participant_col: str = "participant_id",
    ) -> pd.DataFrame:
        """Add a 'split' column to a recordings DataFrame.

        First builds a participant table, splits it, then maps the split
        assignment back to each recording.

        Parameters
        ----------
        recordings : pd.DataFrame
            Must have ``participant_col`` and ``group_col``.
        ratios : tuple
            (train, val, test) ratios.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with an added 'split' column.
        """
        # Build participant table
        participants = (
            recordings
            .groupby(participant_col)[group_col]
            .first()
            .reset_index()
        )
        participants.columns = ["participant_id", "group"]

        # Get splits
        splits = self.split(participants, ratios=ratios)

        # Create mapping
        pid_to_split = {}
        for split_name, pids in splits.items():
            for pid in pids:
                pid_to_split[pid] = split_name

        # Map to recordings
        recordings = recordings.copy()
        recordings["split"] = recordings[participant_col].map(pid_to_split)

        # Log distribution
        for split_name in ["train", "val", "test"]:
            mask = recordings["split"] == split_name
            n = mask.sum()
            if n > 0:
                dist = recordings.loc[mask, group_col].value_counts().to_dict()
                logger.info(f"Split '{split_name}': {n} recordings — {dist}")

        return recordings
