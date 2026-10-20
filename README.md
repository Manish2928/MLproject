# DementiaVoiceAI

**AI-Based Alzheimer's Disease Detection Using Speech**

> ⚠️ **Disclaimer:** This is an academic AI/ML research project. It is **NOT** a medical diagnostic tool and does not claim to diagnose Alzheimer's disease.

## Team

| Name | Registration |
|------|-------------|
| Kashyap Manish | 26MAG0047 |
| Pravin Kumar | 26MAG0012 |

## Overview

DementiaVoiceAI is an end-to-end machine learning system that analyzes speech recordings to identify patterns associated with cognitive impairment. The system extracts acoustic features from audio recordings and uses trained classifiers to produce a binary classification (Cognitive Impairment vs. Control) with confidence scores.

## Project Structure

```
DementiaVoiceAI/
├── configs/              # YAML configuration files
│   └── default.yaml      # Master configuration
├── data/                 # Audio data (NOT in version control)
│   ├── dementia/         # Dementia speaker recordings
│   ├── non_dementia/     # Control speaker recordings
│   ├── train_dm.csv      # Training manifest
│   └── valid_dm.csv      # Validation manifest
├── src/                  # Source code
│   ├── data/             # Dataset adapters and splitting
│   ├── preprocessing/    # Audio validation, preprocessing, VAD
│   ├── features/         # Feature extraction (acoustic, embeddings)
│   ├── models/           # Model definitions (baseline, neural)
│   ├── training/         # Training loops and callbacks
│   ├── evaluation/       # Metrics, plots, reporting
│   ├── inference/        # End-to-end prediction pipeline
│   └── utils/            # Config, logging, seeding, privacy
├── app/                  # Streamlit web application
├── notebooks/            # Jupyter notebooks for EDA
├── saved_models/         # Trained model checkpoints
├── results/              # Evaluation results and plots
├── tests/                # Unit and integration tests
├── scripts/              # CLI scripts for training/evaluation
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project metadata
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: install deep learning dependencies
pip install torch torchaudio transformers
```

### 2. Verify Setup

```bash
python main.py
```

### 3. Validate Dataset

```bash
python main.py --validate
```

## Configuration

All settings are in `configs/default.yaml`. Key sections:
- **audio**: Sample rate, normalization, noise reduction
- **features**: Which features to extract (acoustic, embeddings, linguistic)
- **training**: Model types, hyperparameters, split ratios
- **evaluation**: Metrics, plotting options

## Privacy & Data Handling

- All participant/speaker identifiers are anonymized in logs and outputs
- Raw audio data is excluded from version control via `.gitignore`
- The system never displays real participant names in the UI
- No personally identifiable information is stored in model outputs

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Setup | ✅ | Project structure, config, utilities |
| 2. Data Adapter | ✅ | Dataset loading, splitting, validation |
| 3. Preprocessing | ✅ | Audio validation, normalization, VAD |
| 4. Features | 🔲 | Acoustic feature extraction |
| 5. Baselines | 🔲 | LR, SVM, RF, XGBoost |
| 6. Evaluation | 🔲 | Metrics, plots, reporting |
| 7. Embeddings | 🔲 | Wav2Vec2/HuBERT features |
| 8. Multimodal | 🔲 | Audio + transcript fusion |
| 9. Inference | 🔲 | End-to-end prediction |
| 10. UI | 🔲 | Streamlit application |
| 11. Testing | 🔲 | Unit & integration tests |
| 12. Docs | 🔲 | Final report |

## License

Academic project — not for clinical use.
