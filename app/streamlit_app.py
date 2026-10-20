"""
DementiaVoiceAI — Streamlit Web Application.

Interactive demo for speech-based dementia detection.
Upload a WAV/MP3 file → get a prediction with confidence score.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DementiaVoiceAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-dementia {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .result-control {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .confidence-high { color: #059669; font-weight: bold; }
    .confidence-medium { color: #d97706; font-weight: bold; }
    .confidence-low { color: #dc2626; font-weight: bold; }

    /* Disclaimer */
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    /* Upload area */
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Load the inference pipeline with the saved model."""
    from src.utils.config import load_config, resolve_path
    from src.inference.pipeline import InferencePipeline

    cfg = load_config()
    models_dir = resolve_path(cfg.paths.saved_models)
    model_path = models_dir / "best_model.joblib"
    feature_cols_path = models_dir / "feature_columns.json"

    if not model_path.exists():
        return None, cfg

    pipeline = InferencePipeline.from_saved(
        cfg, model_path, feature_cols_path
    )
    return pipeline, cfg


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<p class="main-header">🧠 DementiaVoiceAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Based Speech Analysis for Cognitive Impairment Research</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain--v2.png", width=80)
        st.markdown("## About")
        st.markdown(
            "This tool analyses speech recordings to identify acoustic "
            "patterns that may be associated with cognitive impairment."
        )
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown(
            "1. **Upload** a WAV or MP3 audio file\n"
            "2. **Features** are extracted automatically\n"
            "3. A trained **ML model** makes a prediction\n"
            "4. Results show the **confidence level**"
        )
        st.markdown("---")
        st.markdown(
            "⚠️ **Research Only** — This is an academic project, "
            "not a medical diagnostic tool."
        )

    # Load pipeline
    pipeline, cfg = load_pipeline()

    if pipeline is None:
        st.error(
            "⚠️ No trained model found! "
            "Run `python scripts/run_pipeline.py` first to train a model."
        )
        st.info(
            "The pipeline script will:\n"
            "1. Extract audio features from the dataset\n"
            "2. Train multiple ML models\n"
            "3. Save the best model for inference"
        )
        st.stop()

    # File upload
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Upload a speech recording (max 10 minutes)",
    )

    if uploaded_file is not None:
        # Show audio player
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        # Save temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run prediction
        with st.spinner("🔬 Analysing audio features..."):
            start = time.time()
            try:
                result = pipeline.predict(str(temp_path))
                elapsed = time.time() - start
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
                st.stop()

        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            pred_class = result["predicted_class"]
            if pred_class == "Dementia":
                st.metric(
                    label="Prediction",
                    value="⚠️ Dementia-like Pattern",
                )
            else:
                st.metric(
                    label="Prediction",
                    value="✅ Control / Healthy Pattern",
                )

        with col2:
            confidence_pct = f"{result['confidence'] * 100:.1f}%"
            band = result["confidence_band"]
            band_color = {
                "High": "🟢", "Medium": "🟡", "Low": "🔴"
            }.get(band, "⚪")
            st.metric(
                label="Confidence",
                value=confidence_pct,
                delta=f"{band_color} {band}",
            )

        with col3:
            st.metric(
                label="Processing Time",
                value=f"{elapsed:.1f}s",
            )

        # Detailed results table
        with st.expander("📋 Detailed Results"):
            details = {
                "File": result["file"],
                "Predicted Class": result["predicted_class"],
                "Confidence": f"{result['confidence']:.4f}",
                "Confidence Band": result["confidence_band"],
                "Raw Probability (Dementia)": f"{result['raw_probability_dementia']:.4f}",
                "Threshold": f"{result['threshold']:.2f}",
            }
            st.table(pd.DataFrame(details.items(), columns=["Field", "Value"]))

        # Disclaimer
        st.markdown(
            f'<div class="disclaimer">'
            f"⚠️ <strong>Important Disclaimer:</strong> {result['disclaimer']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    else:
        # Show placeholder when no file uploaded
        st.markdown(
            '<div class="upload-section">'
            "<h4>👆 Upload an audio file to get started</h4>"
            "<p>Supported formats: WAV, MP3, FLAC, OGG</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Show model info
        st.markdown("---")
        st.markdown("### 📈 Model Information")

        import json
        from src.utils.config import resolve_path

        config_path = resolve_path(cfg.paths.saved_models) / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                run_config = json.load(f)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", run_config.get("selected_model", "N/A"))
            with col2:
                st.metric(
                    "Test Balanced Accuracy",
                    f"{run_config.get('test_balanced_accuracy', 0):.2%}",
                )
            with col3:
                st.metric(
                    "Test ROC AUC",
                    f"{run_config.get('test_roc_auc', 0):.2%}",
                )
            with col4:
                st.metric(
                    "Features Used",
                    run_config.get("n_features", "N/A"),
                )


if __name__ == "__main__":
    main()
