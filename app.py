"""
audiovec -- Streamlit frontend
==============================
Upload an audio file, and we'll predict the emotion and extract a 256-d
embedding vector using the trained CRNN-Transformer model.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import traceback
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from audiovec.config import EMOTION_MAPPING, MAX_PAD_LEN, SAMPLE_RATE
from audiovec.data import load_and_process_audio
from audiovec.predict import ensure_model, extract_embedding, load_trained_model, predict_emotion

# -- Page config --------------------------------------------------------------

st.set_page_config(
    page_title="audiovec -- emotion embedding",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -- Custom CSS -- dark, polished, audio-wave vibe ----------------------------

st.markdown(
    """
<style>
    /* -- Base ----------------------------------------------------------- */
    .stApp {
        background: #0B0E14;
    }
    /* main block container */
    .main > div {
        background: transparent !important;
    }
    /* remove default padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* -- Header ---------------------------------------------------------- */
    .app-header {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .app-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #A78BFA 0%, #EC4899 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .app-header .subtitle {
        color: #8B95A8;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
    }

    /* -- Metric cards ---------------------------------------------------- */
    div[data-testid="metric-container"] {
        background: #141A24;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #8B95A8 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #F1F5F9 !important;
        font-size: 1.3rem !important;
        font-weight: 600;
    }

    /* -- File uploader --------------------------------------------------- */
    section[data-testid="stFileUploader"] {
        background: #141A24;
        border: 2px dashed #2A3A4E;
        border-radius: 16px;
        padding: 2rem 1rem;
        transition: border-color 0.3s, background 0.3s;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: #A78BFA;
        background: #1A212D;
    }
    section[data-testid="stFileUploader"] button {
        background: #A78BFA !important;
        color: #0B0E14 !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.2rem !important;
    }
    section[data-testid="stFileUploader"] button:hover {
        background: #C4B5FD !important;
    }

    /* -- Progress bars --------------------------------------------------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #A78BFA, #EC4899) !important;
    }

    /* -- Section headers ------------------------------------------------- */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1E293B;
        letter-spacing: -0.01em;
    }

    /* -- Emotion bar labels ---------------------------------------------- */
    .emotion-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.3rem;
    }
    .emotion-label {
        width: 6rem;
        font-size: 0.85rem;
        color: #CBD5E1;
        text-align: right;
        font-weight: 500;
    }
    .emotion-bar-bg {
        flex: 1;
        height: 1.2rem;
        background: #1E293B;
        border-radius: 6px;
        overflow: hidden;
    }
    .emotion-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    .emotion-pct {
        width: 3rem;
        font-size: 0.78rem;
        color: #8B95A8;
        font-variant-numeric: tabular-nums;
    }
    .emotion-pct.active {
        color: #A78BFA;
        font-weight: 600;
    }

    /* -- Embedding sparkline --------------------------------------------- */
    .embedding-bar {
        background: #141A24;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .embedding-bar .bar-title {
        color: #8B95A8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }

    /* -- Audio player ---------------------------------------------------- */
    [data-testid="stAudio"] {
        background: #141A24;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stAudio"] audio {
        width: 100%;
        height: 40px;
    }
    [data-testid="stAudio"] audio::-webkit-media-controls-panel {
        background: #1A212D;
    }

    /* -- Footer ---------------------------------------------------------- */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #1E293B;
    }

    /* -- Hide Streamlit branding ----------------------------------------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# -- Header -------------------------------------------------------------------

st.markdown(
    '<div class="app-header">'
    "<h1>audiovec</h1>"
    '<div class="subtitle">upload speech audio &middot; predict emotion &middot; extract 256-d embedding</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# -- Check for trained model --------------------------------------------------

MODEL_PATH = Path("models/audiovec_model.pt")
try:
    MODEL_PATH = ensure_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def _load_model():
    return load_trained_model(MODEL_PATH)


model = _load_model()

# -- Upload -------------------------------------------------------------------

uploaded = st.file_uploader(
    "Drop your audio file here",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    label_visibility="collapsed",
)

if not uploaded:
    # Show a placeholder prompt
    st.info(
        "Upload a **.wav** or **.mp3** audio file to get started.\n\n"
        "Supported formats: WAV, MP3, M4A, OGG, FLAC",
        icon="🎤",
    )
    st.stop()

# -- Process audio ------------------------------------------------------------

with st.spinner("Processing audio..."):
    try:
        # Read the uploaded file into a bytes buffer
        audio_bytes = uploaded.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio with librosa for the waveform & spectrogram
        audio, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)

        # Reset buffer position for our processing function
        audio_buffer.seek(0)

        # Process through our pipeline -- save to temp file since
        # load_and_process_audio expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name


        try:
            spectrogram = load_and_process_audio(tmp_path, max_pad_len=MAX_PAD_LEN)
            emotion, confidence, probs = predict_emotion(model, spectrogram)
            embedding = extract_embedding(model, spectrogram)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        st.error(f"Failed to process audio: {e}")
        st.exception(e)
        st.stop()

# -- Results layout -----------------------------------------------------------

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Emotion", emotion.upper())
with col2:
    st.metric("Confidence", f"{confidence:.1%}")
with col3:
    top3 = np.argsort(probs)[-3:][::-1]
    top3_labels = " . ".join(
        EMOTION_MAPPING.get(int(i) + 1, "?").title() for i in top3
    )
    st.metric("Top 3", top3_labels)

# -- Audio playback -----------------------------------------------------------

st.markdown('<div class="section-title">Audio Playback</div>', unsafe_allow_html=True)

# Determine the MIME type based on the file extension
audio_ext = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac"}
audio_mime = mime_map.get(audio_ext, "audio/wav")

st.audio(audio_bytes, format=audio_mime)

st.caption(f"File: {uploaded.name}  .  {len(audio)/sr:.1f}s  .  {audio_ext.upper()}")

# -- Emotion probability bars -------------------------------------------------

st.markdown('<div class="section-title">Emotion Probabilities</div>', unsafe_allow_html=True)

# Map indices 0-7 to emotion codes 1-8
emotion_codes = sorted(EMOTION_MAPPING.keys())
colors = [
    "#A78BFA", "#60A5FA", "#34D399", "#FBBF24",
    "#F87171", "#FB923C", "#E879F9", "#22D3EE",
]

html_bars = ""
for i, code in enumerate(emotion_codes):
    label = EMOTION_MAPPING[code].title()
    prob = probs[code - 1]
    pct = f"{prob:.1%}"
    bar_color = colors[i % len(colors)]
    is_active = code - 1 == np.argmax(probs)
    active_cls = "active" if is_active else ""
    html_bars += f"""
    <div class="emotion-row">
        <div class="emotion-label">{label}</div>
        <div class="emotion-bar-bg">
            <div class="emotion-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color};"></div>
        </div>
        <div class="emotion-pct {active_cls}">{pct}</div>
    </div>
    """
st.markdown(html_bars, unsafe_allow_html=True)

# -- Embedding + Spectrogram --------------------------------------------------

st.markdown('<div class="section-title">Embedding Vector</div>', unsafe_allow_html=True)

# Show a compact "sparkline" of the full 256-d vector using matplotlib
fig_embed, ax_embed = plt.subplots(figsize=(10, 1.3))
fig_embed.patch.set_facecolor("#0B0E14")
ax_embed.set_facecolor("#0B0E14")

# Create a custom colormap from purple to pink
cmap = LinearSegmentedColormap.from_list(
    "embed_cmap", ["#1E1B4B", "#A78BFA", "#EC4899", "#FBBF24"], N=256
)
embed_norm = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)
ax_embed.bar(
    range(len(embedding)),
    embed_norm,
    width=1.0,
    color=cmap(embed_norm),
    edgecolor="none",
)
ax_embed.set_xlim(0, len(embedding))
ax_embed.set_ylim(0, 1.05)
ax_embed.set_xticks([])
ax_embed.set_yticks([])
ax_embed.spines[:].set_visible(False)
fig_embed.tight_layout(pad=0)

st.pyplot(fig_embed, width='stretch')

# Download button for embedding
embedding_json = json.dumps(
    {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "embedding": embedding.tolist(),
    },
    indent=2,
)
st.download_button(
    label="Download Embedding as JSON",
    data=embedding_json,
    file_name="audiovec_embedding.json",
    mime="application/json",
    width='stretch',
)

# -- Spectrogram visualization ------------------------------------------------

st.markdown('<div class="section-title">Mel-Spectrogram</div>', unsafe_allow_html=True)

fig_spec, ax_spec = plt.subplots(figsize=(10, 3.5))
fig_spec.patch.set_facecolor("#0B0E14")
ax_spec.set_facecolor("#0B0E14")

S_dB = spectrogram.squeeze(-1)  # (128, 228)
img = ax_spec.imshow(
    S_dB,
    aspect="auto",
    origin="lower",
    cmap="magma",
    interpolation="bilinear",
)
ax_spec.set_xlabel("Time frames", color="#8B95A8", fontsize=8)
ax_spec.set_ylabel("Mel bands", color="#8B95A8", fontsize=8)
ax_spec.tick_params(colors="#8B95A8", labelsize=7)
ax_spec.set_title(
    f"Spectrogram -- {uploaded.name} ({len(audio)/sr:.1f}s @ {sr} Hz)",
    color="#CBD5E1",
    fontsize=10,
    pad=8,
)
cbar = fig_spec.colorbar(img, ax=ax_spec, pad=0.02, shrink=0.85)
cbar.ax.yaxis.set_tick_params(color="#8B95A8", labelsize=7)
fig_spec.tight_layout()

st.pyplot(fig_spec, width='stretch')

# -- Waveform -----------------------------------------------------------------

st.markdown('<div class="section-title">Waveform</div>', unsafe_allow_html=True)

fig_wav, ax_wav = plt.subplots(figsize=(10, 1.8))
fig_wav.patch.set_facecolor("#0B0E14")
ax_wav.set_facecolor("#0B0E14")

time = np.linspace(0, len(audio) / sr, len(audio))
ax_wav.fill_between(time, audio, 0, color="#A78BFA", alpha=0.5, lw=0)
ax_wav.fill_between(time, audio, 0, color="#EC4899", alpha=0.2, lw=0)
ax_wav.set_xlim(0, time[-1])
ax_wav.set_ylim(-1, 1)
ax_wav.set_yticks([])
ax_wav.set_xlabel("Time (s)", color="#8B95A8", fontsize=8)
ax_wav.tick_params(colors="#8B95A8", labelsize=7)
ax_wav.spines[:].set_visible(False)
ax_wav.set_facecolor("#0B0E14")
fig_wav.tight_layout(pad=0.5)

st.pyplot(fig_wav, width='stretch')

# -- Footer -------------------------------------------------------------------

st.markdown(
    '<div class="footer">'
    "audiovec &middot; 256-dimensional audio sentiment embedding &middot; RAVDESS dataset"
    "</div>",
    unsafe_allow_html=True,
)
