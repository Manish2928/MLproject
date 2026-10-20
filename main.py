"""
DementiaVoiceAI — Main Entry Point

Quick-start script to verify the project setup and configuration.

Usage:
    python main.py              # Verify setup
    python main.py --validate   # Validate dataset
"""

from __future__ import annotations

import argparse
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def verify_setup() -> None:
    """Verify that the project is correctly set up."""
    print("=" * 60)
    print("  DementiaVoiceAI - Setup Verification")
    print("=" * 60)
    print()

    # 1. Config
    print("[1/4] Loading configuration...")
    try:
        from src.utils.config import load_config, PROJECT_ROOT
        cfg = load_config()
        print(f"  [OK] Config loaded from: {PROJECT_ROOT / 'configs' / 'default.yaml'}")
        print(f"  [OK] Project: {cfg.project.name} v{cfg.project.version}")
    except Exception as e:
        print(f"  [FAIL] Config failed: {e}")
        return

    # 2. Seed
    print("\n[2/4] Setting reproducibility seed...")
    from src.utils.seed import set_global_seed
    set_global_seed(cfg.project.seed)
    print(f"  [OK] Global seed set to {cfg.project.seed}")

    # 3. Dependencies
    print("\n[3/4] Checking core dependencies...")
    deps = {
        "numpy": "numpy",
        "pandas": "pandas",
        "librosa": "librosa",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "matplotlib": "matplotlib",
    }
    for import_name, display_name in deps.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [FAIL] {display_name}: NOT INSTALLED")

    # Optional deps
    optional_deps = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "streamlit": "Streamlit",
        "xgboost": "XGBoost",
    }
    print("\n  Optional:")
    for import_name, display_name in optional_deps.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [--] {display_name}: not installed (optional)")

    # 4. Dataset
    print("\n[4/4] Checking data directories...")
    from src.utils.config import resolve_path

    checks = [
        ("Train CSV", cfg.paths.train_csv),
        ("Valid CSV", cfg.paths.valid_csv),
        ("Dementia audio", cfg.paths.dementia_audio_dir),
        ("Non-dementia audio", cfg.paths.non_dementia_audio_dir),
    ]
    for name, rel_path in checks:
        p = resolve_path(rel_path)
        if p.exists():
            print(f"  [OK] {name}: {p}")
        else:
            print(f"  [FAIL] {name}: NOT FOUND at {p}")

    print("\n" + "=" * 60)
    print("  Setup verification complete!")
    print("=" * 60)


def validate_dataset() -> None:
    """Run full dataset validation."""
    from src.utils.config import load_config
    from src.utils.seed import set_global_seed
    from src.data.dataset_adapter import CelebritySpeechAdapter

    cfg = load_config()
    set_global_seed(cfg.project.seed)

    print("=" * 60)
    print("  DementiaVoiceAI - Dataset Validation")
    print("=" * 60)
    print()

    adapter = CelebritySpeechAdapter(cfg, anonymize=True)

    # Validate source
    print("[1/3] Validating source directories...")
    if not adapter.validate_source():
        print("  [FAIL] Source validation failed!")
        return

    # Load participants
    print("\n[2/3] Loading participants...")
    participants = adapter.load_participants()
    print(f"  Total participants: {len(participants)}")
    print(f"  Class distribution:")
    for group, count in participants["group"].value_counts().items():
        print(f"    {group}: {count}")

    # Load recordings
    print("\n[3/3] Loading recordings...")
    recordings = adapter.load_recordings()
    print(f"  Total recordings: {len(recordings)}")
    print(f"  Files found on disk: {recordings['file_found'].sum()}")
    print(f"  Files missing: {(~recordings['file_found']).sum()}")

    # Check splits
    print("\n  Checking for participant leakage...")
    splits = adapter.get_splits()

    print("\n" + "=" * 60)
    print("  Dataset validation complete!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DementiaVoiceAI - AI-Based Alzheimer's Detection"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full dataset validation",
    )
    args = parser.parse_args()

    if args.validate:
        validate_dataset()
    else:
        verify_setup()


if __name__ == "__main__":
    main()
