"""Tests for the participant-level splitter."""

from src.data.splitter import ParticipantSplitter


class TestParticipantSplitter:
    """Test suite for ParticipantSplitter."""

    def test_split_no_overlap(self, sample_participants_df):
        """Verify no participant appears in multiple splits."""
        splitter = ParticipantSplitter(seed=42)
        splits = splitter.split(
            sample_participants_df, ratios=(0.7, 0.15, 0.15)
        )

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_covers_all_participants(self, sample_participants_df):
        """Verify all participants are assigned to exactly one split."""
        splitter = ParticipantSplitter(seed=42)
        splits = splitter.split(
            sample_participants_df, ratios=(0.7, 0.15, 0.15)
        )

        all_split_ids = (
            set(splits["train"])
            | set(splits["val"])
            | set(splits["test"])
        )
        all_ids = set(sample_participants_df["participant_id"])

        assert all_split_ids == all_ids

    def test_split_preserves_class_balance(self, sample_participants_df):
        """Verify that both classes appear in the training set."""
        splitter = ParticipantSplitter(seed=42)
        splits = splitter.split(
            sample_participants_df, ratios=(0.7, 0.15, 0.15)
        )

        train_groups = (
            sample_participants_df[
                sample_participants_df["participant_id"].isin(
                    splits["train"]
                )
            ]["group"]
            .unique()
        )
        assert "dementia" in train_groups
        assert "control" in train_groups

    def test_split_reproducible(self, sample_participants_df):
        """Verify same seed produces same split."""
        splitter1 = ParticipantSplitter(seed=123)
        splitter2 = ParticipantSplitter(seed=123)

        splits1 = splitter1.split(sample_participants_df)
        splits2 = splitter2.split(sample_participants_df)

        assert splits1 == splits2
