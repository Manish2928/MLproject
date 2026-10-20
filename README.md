# 🧠 DementiaVoiceAI

**AI-Based Alzheimer's Disease Detection Using Speech**

> ⚠️ **Disclaimer:** This is an academic AI/ML research project. It is **NOT** a medical diagnostic tool and does not claim to diagnose Alzheimer's disease.

---

## 👥 Team

| Name | Registration | Programme |
|------|-------------|-----------|
| Kashyap Manish | 26MAG0047 | M.Sc. AIML |
| Pravin Kumar | 26MAG0012 | M.Sc. AIML |

**Faculty Guide:** Dr. Durai Raj Vincent P.M  
**Institution:** VIT Vellore University

---

## 📖 Overview

DementiaVoiceAI is an end-to-end machine learning system that analyses speech recordings to identify acoustic patterns associated with cognitive impairment. The system:

1. **Preprocesses** raw audio (resampling, normalization, noise reduction, VAD)
2. **Extracts** 143 acoustic features (MFCCs, spectral, chroma, pitch, pause features)
3. **Trains** 8 classical ML models (Logistic Regression, SVM, Random Forest, XGBoost, etc.)
4. **Evaluates** with comprehensive metrics (balanced accuracy, ROC-AUC, F1, confusion matrix)
5. **Serves** predictions via an interactive Streamlit web application

### Current Results (Best Model: Gradient Boosting)

| Metric | Test Score |
|--------|-----------|
| Balanced Accuracy | 60.83% |
| ROC AUC | 60.06% |
| F1 Score (Dementia) | 54.55% |
| Specificity | 64.52% |
| Sensitivity / Recall | 57.14% |

---

## 📁 Project Structure

```
DementiaVoiceAI/
├── app/                        # Streamlit web application
│   └── streamlit_app.py        # Interactive demo UI
├── configs/                    # YAML configuration files
│   └── default.yaml            # Master configuration (single source of truth)
├── data/                       # Audio data (NOT in version control)
│   ├── dementia/               # Dementia speaker recordings (.wav)
│   ├── non_dementia/           # Control speaker recordings (.wav)
│   ├── train_dm.csv            # Training manifest
│   └── valid_dm.csv            # Validation manifest
├── notebooks/                  # Jupyter notebooks for EDA
│   └── 01_exploration.ipynb    # Data exploration notebook
├── results/                    # Generated evaluation results
│   ├── metrics/                # CSV metrics files
│   └── plots/                  # Confusion matrix, ROC, PR curves
├── saved_models/               # Trained model checkpoints
│   ├── best_model.joblib       # Serialized best model
│   ├── feature_columns.json    # Feature schema
│   └── run_config.json         # Training run metadata
├── scripts/                    # CLI automation scripts
│   └── run_pipeline.py         # Master pipeline (train → evaluate → save)
├── src/                        # Source code
│   ├── data/                   # Dataset adapters and loading
│   │   ├── dataset_adapter.py  # Celebrity speech dataset adapter
│   │   └── splitter.py         # Participant-level stratified splitting
│   ├── evaluation/             # Metrics, plots, reporting
│   │   └── evaluator.py        # Model evaluation and visualization
│   ├── features/               # Feature extraction
│   │   └── audio_features.py   # 143 acoustic features extractor
│   ├── inference/              # End-to-end prediction
│   │   └── pipeline.py         # Audio → features → prediction pipeline
│   ├── models/                 # Model definitions (placeholder for neural)
│   ├── preprocessing/          # Audio preprocessing
│   │   ├── audio_preprocessor.py  # Resampling, normalization
│   │   ├── audio_validator.py     # Audio file validation
│   │   └── vad.py                 # Energy-based Voice Activity Detection
│   ├── training/               # Model training
│   │   └── trainer.py          # Multi-model trainer with comparison
│   └── utils/                  # Shared utilities
│       ├── config.py           # YAML config loader with DotDict
│       ├── logger.py           # Structured logging setup
│       ├── privacy.py          # Participant anonymization
│       └── seed.py             # Reproducibility seeding
├── tests/                      # Unit tests
│   ├── conftest.py             # Pytest fixtures
│   ├── test_preprocessing.py   # Preprocessing tests
│   └── test_splitter.py        # Data splitting tests
├── main.py                     # Entry point (setup verification)
├── pyproject.toml              # Project metadata
├── requirements.txt            # Python dependencies
└── .gitignore                  # Version control exclusions
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (tested with CPython 3.13)
- **pip** package manager
- **Git** for version control

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DementiaVoiceAI.git
cd DementiaVoiceAI
```

### 2. Create & Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
.venv\Scripts\activate.bat

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Optional (Deep Learning):** If you plan to use Wav2Vec2/HuBERT embeddings:
> ```bash
> pip install torch torchaudio transformers
> ```

### 4. Verify Installation

```bash
python main.py
```

This runs a setup check that verifies:
- ✅ Configuration loads correctly
- ✅ Reproducibility seed is set
- ✅ All core dependencies are installed
- ✅ Data directories and CSV files exist

### 5. Prepare the Dataset

Place your audio data in the following structure:

```
data/
├── dementia/         # .wav files of dementia speakers
├── non_dementia/     # .wav files of control speakers
├── train_dm.csv      # Training manifest with columns: participant_id, file, group
└── valid_dm.csv      # Validation manifest with same columns
```

### 6. Validate the Dataset

```bash
python main.py --validate
```

---

## 🔬 Running the ML Pipeline

The master pipeline script handles the entire workflow:

```bash
python scripts/run_pipeline.py
```

This will:
1. Load and validate the dataset (275 recordings, 141 speakers)
2. Extract 143 acoustic features from all audio files
3. Perform speaker-level stratified train/val/test split (no data leakage)
4. Train 8 ML models and compare on validation set
5. Evaluate the best model on the held-out test set
6. Generate evaluation plots (confusion matrix, ROC, PR curves)
7. Compute permutation feature importance
8. Save the best model and feature schema to `saved_models/`

**Options:**

```bash
# Force re-extraction of features (ignores cache)
python scripts/run_pipeline.py --no-cache

# Use a custom config file
python scripts/run_pipeline.py --config path/to/custom.yaml
```

---

## 🌐 Streamlit Web Application

Launch the interactive demo after training a model:

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- Upload WAV/MP3/FLAC/OGG audio files
- Real-time acoustic feature extraction
- Binary prediction with confidence score and bands (High / Medium / Low)
- Audio playback in browser
- Model performance overview

---

## ⚙️ Configuration

All project settings are centralized in [`configs/default.yaml`](configs/default.yaml):

| Section | Key Settings |
|---------|-------------|
| `project` | Name, version, seed (42), device |
| `audio` | Sample rate (16kHz), normalization, VAD settings |
| `features` | MFCC params, spectral features, pitch, chroma |
| `splitting` | Train/val/test ratios, stratification method |
| `training` | Model types, scaler, class weighting, CV folds |
| `evaluation` | Metrics, threshold (0.5), plot formats |
| `inference` | Confidence bands, disclaimer text |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_preprocessing.py -v
```

---

## 🔒 Privacy & Data Handling

- All participant/speaker identifiers are **anonymized** in logs and outputs
- Raw audio data is **excluded from version control** via `.gitignore`
- The system **never displays** real participant names in the UI
- No personally identifiable information is stored in model outputs
- Audio files uploaded via the Streamlit app are **deleted after prediction**

---

## 📊 ML Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Raw Audio   │───▶│ Preprocessing │───▶│ Feature Extraction│
│  (.wav files)│    │ Resample 16kHz│    │  143 features     │
│              │    │ Normalize     │    │  MFCCs, spectral  │
│              │    │ VAD           │    │  pitch, pauses    │
└─────────────┘    └──────────────┘    └──────────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Streamlit   │◀───│  Inference   │◀───│    Training       │
│  Web App     │    │  Pipeline    │    │  8 ML models      │
│  Upload →    │    │  Audio →     │    │  Speaker-level    │
│  Predict     │    │  Prediction  │    │  stratified split │
└─────────────┘    └──────────────┘    └──────────────────┘
```

---

## 📦 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12+ |
| **Audio** | librosa, soundfile, soxr |
| **ML** | scikit-learn, XGBoost |
| **Visualization** | matplotlib, seaborn |
| **Config** | PyYAML |
| **Web App** | Streamlit |
| **Testing** | pytest, pytest-cov |
| **DL (Optional)** | PyTorch, Hugging Face Transformers |

---

## 📈 Development Phases

| Phase | Status | Description |
|-------|:------:|-------------|
| 1. Setup | ✅ | Project structure, config, utilities |
| 2. Data Adapter | ✅ | Dataset loading, splitting, validation |
| 3. Preprocessing | ✅ | Audio validation, normalization, VAD |
| 4. Features | ✅ | 143 acoustic features extraction |
| 5. Baselines | ✅ | LR, SVM, RF, XGBoost + 4 more models |
| 6. Evaluation | ✅ | Metrics, plots, permutation importance |
| 7. Inference | ✅ | End-to-end prediction pipeline |
| 8. UI | ✅ | Streamlit web application |
| 9. Embeddings | 🔲 | Wav2Vec2/HuBERT features (planned) |
| 10. Multimodal | 🔲 | Audio + transcript fusion (planned) |
| 11. Testing | 🔶 | Unit tests (partial coverage) |
| 12. Docs | ✅ | README, Phase 2 report |

---

## 📜 License

Academic project — **not for clinical use**.

---

*Built with ❤️ at VIT Vellore University*
