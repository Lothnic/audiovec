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
import subprocess
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
from audiovec.search import embeddings_cached, find_similar, load_ravdess_embeddings
from audiovec.sliding import compute_sliding_predictions, plot_emotion_curves

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
    /* Upload column */
    [data-testid="column"]:first-child section[data-testid="stFileUploader"] {
        background: #141A24;
        border: 2px dashed #2A3A4E;
        border-radius: 16px;
        padding: 2rem 1rem;
        transition: border-color 0.3s, background 0.3s;
    }
    [data-testid="column"]:first-child section[data-testid="stFileUploader"]:hover {
        border-color: #A78BFA;
        background: #1A212D;
    }
    [data-testid="column"]:first-child section[data-testid="stFileUploader"] button {
        background: #A78BFA !important;
        color: #0B0E14 !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.2rem !important;
    }
    [data-testid="column"]:first-child section[data-testid="stFileUploader"] button:hover {
        background: #C4B5FD !important;
    }

    /* -- Mic widget ------------------------------------------------------ */
    [data-testid="column"]:last-child [data-testid="stAudioInput"] {
        background: #141A24;
        border: 2px dashed #2A3A4E;
        border-radius: 16px;
        padding: 0.5rem 1rem;
        transition: border-color 0.3s, background 0.3s;
    }
    [data-testid="column"]:last-child [data-testid="stAudioInput"]:hover {
        border-color: #A78BFA;
        background: #1A212D;
    }
    [data-testid="column"]:last-child [data-testid="stAudioInput"] button {
        background: #A78BFA !important;
        color: #0B0E14 !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.2rem !important;
    }
    [data-testid="column"]:last-child [data-testid="stAudioInput"] button:hover {
        background: #C4B5FD !important;
    }
    [data-testid="column"]:last-child [data-testid="stAudioInput"] svg {
        color: #C4B5FD !important;
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

    /* -- Similarity cards ------------------------------------------------ */
    .similar-card {
        background: #141A24;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .similar-card:hover {
        border-color: #A78BFA;
        box-shadow: 0 0 12px rgba(167,139,250,0.12);
    }
    .similar-card .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .similar-card .info {
        flex: 1;
        min-width: 0;
    }
    .similar-card .info .emotion {
        font-size: 0.85rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .similar-card .info .meta {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 0.1rem;
    }
    .similar-card .score {
        font-size: 0.85rem;
        font-weight: 600;
        color: #A78BFA;
        flex-shrink: 0;
        font-variant-numeric: tabular-nums;
    }
    .similar-card .audio-wrap {
        flex-shrink: 0;
        width: 180px;
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

    /* -- Emotion flow dots --------------------------------------------- */
    .emotion-flow {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .emotion-flow .flow-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 3px;
        vertical-align: middle;
    }
    .emotion-flow .flow-arrow {
        color: #475569;
        margin: 0 4px;
    }
    .emotion-flow .flow-count {
        color: #64748B;
        font-size: 0.78rem;
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


@st.cache_resource(show_spinner="Loading RAVDESS reference embeddings…")
def _load_ravdess_embeddings():
    return load_ravdess_embeddings()


# -- Upload / Record ----------------------------------------------------------

col_upload, col_mic = st.columns(2, gap="large")

with col_upload:
    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "mp4"],
        label_visibility="collapsed",
    )

with col_mic:
    recorded = st.audio_input(
        "Record from microphone",
        sample_rate=SAMPLE_RATE,
        label_visibility="collapsed",
    )

if not uploaded and not recorded:
    st.info(
        "Upload an **audio** or **video** file, or record directly with your microphone.\n\n"
        "Supported formats: WAV, MP3, M4A, OGG, FLAC, MP4 (audio extracted via ffmpeg)",
        icon="🎤",
    )
    st.stop()

# Determine which source to use
is_mic = recorded is not None
source = recorded if is_mic else uploaded

# -- Process audio ------------------------------------------------------------

specific_error_shown = False
try:
    audio_bytes = source.read()
    is_mp4 = not is_mic and source.name.lower().endswith(".mp4")
    audio_wav_path = None

    if is_mp4:
        # Save MP4 to temp file for probing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(audio_bytes)
            video_path = tmp.name

        # Check for audio stream before showing any progress widget
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             video_path],
            capture_output=True,
            text=True,
        )
        if not probe.stdout.strip():
            os.unlink(video_path)
            st.error(
                "This video file contains **no audio track**."
                " Please upload a video with an audible speech signal, or upload an audio file directly "
                "(WAV, MP3, M4A, OGG, FLAC)."
            )
            specific_error_shown = True
            st.stop()

        # Phase 1 — ffmpeg audio extraction (now we know audio exists)
        with st.status("Extracting audio from video…", expanded=False) as status:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio_wav_path = tmp.name

                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", video_path,
                            "-vn", "-acodec", "pcm_s16le",
                            "-ar", str(SAMPLE_RATE), "-ac", "1",
                            audio_wav_path,
                        ],
                        capture_output=True,
                        check=True,
                    )
                finally:
                    os.unlink(video_path)

                status.update(label="Audio extracted ✓", state="complete")
            except Exception:
                status.update(state="error")
                raise

        # Load audio from extracted WAV
        audio, sr = librosa.load(audio_wav_path, sr=SAMPLE_RATE)
        tmp_path = audio_wav_path
    else:
        # Phase 1 — load audio file
        with st.status("Loading audio…", expanded=False) as status:
            try:
                audio_buffer = io.BytesIO(audio_bytes)
                audio, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)

                # Save to temp WAV for load_and_process_audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                source_label = "Microphone" if is_mic else uploaded.name
                status.update(
                    label=f"Loaded {len(audio)/sr:.1f}s audio ✓  ({source_label})",
                    state="complete",
                )
            except Exception:
                status.update(state="error")
                raise

    # Phase 2 — model inference (simple spinner since it's fast)
    with st.spinner("Running inference…"):
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
    # Clean up audio_wav_path if it was created but never reached the spinner's cleanup
    if audio_wav_path is not None:
        try:
            os.unlink(audio_wav_path)
        except Exception:
            pass
    if not specific_error_shown:
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

# Play the waveform array directly -- works for all formats including MP4 (audio extracted via ffmpeg)
st.audio(audio, sample_rate=sr)

if is_mic:
    file_ext = "wav"
    source_label = "Microphone recording"
else:
    file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
    source_label = uploaded.name
edit_suffix = " (audio extracted)" if file_ext == "mp4" else ""
st.caption(f"Source: {source_label}  .  {len(audio)/sr:.1f}s  .  {file_ext.upper()}{edit_suffix}")

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

# -- Emotion probability curves (sliding window over long audio) --------------

DURATION_THRESHOLD = 8.0  # seconds — minimum for meaningful sliding windows

window_len_s = MAX_PAD_LEN * 512 / sr  # seconds per window
duration = len(audio) / sr
if duration > DURATION_THRESHOLD:
    st.markdown('<div class="section-title">Emotion Over Time</div>', unsafe_allow_html=True)
    st.caption(
        f"Sliding-window analysis over {duration:.1f}s of audio "
        f"(window {window_len_s:.1f}s, 50% overlap)."
    )

    with st.spinner("Computing time-resolved predictions…"):
        progress_bar = st.progress(0, text="Processing windows…")
        try:
            result = compute_sliding_predictions(
                model,
                audio,
                sr=sr,
                progress_callback=lambda cur, tot: progress_bar.progress(
                    cur / tot, text=f"Window {cur}/{tot}…"
                ),
            )
        finally:
            progress_bar.empty()

    if not result["single"]:
        fig_curves = plot_emotion_curves(
            result["timeline"],
            result["probs"],
            result["emotion_labels"],
        )
        st.pyplot(fig_curves, width="stretch")

        # Emotion flow — collapse consecutive duplicates with colored dots
        dominant = result["probs"].argmax(axis=1)
        segments = []
        i = 0
        while i < len(dominant):
            idx = int(dominant[i])
            label = result["emotion_labels"][idx]
            count = 1
            while i + count < len(dominant) and int(dominant[i + count]) == idx:
                count += 1
            dot = f"<span class=\"flow-dot\" style=\"background:{colors[idx]};\"></span>"
            seg = f"{dot}{label}"
            if count > 1:
                seg += f" <span class=\"flow-count\">(×{count})</span>"
            segments.append(seg)
            i += count
        flow_html = "<span class=\"flow-arrow\">→</span>".join(segments)
        st.markdown(
            f"<div class=\"emotion-flow\">Emotion flow: {flow_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Audio is not long enough for meaningful sliding-window analysis. "
            "The single global prediction is shown above."
        )
    st.divider()

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
    f"Spectrogram ({len(audio)/sr:.1f}s @ {sr} Hz)",
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

# -- Similarity search — find nearest RAVDESS samples ------------------------

EMOTION_COLORS = {
    "neutral": "#A78BFA", "calm": "#60A5FA", "happy": "#34D399",
    "sad": "#FBBF24", "angry": "#F87171", "fearful": "#FB923C",
    "disgust": "#E879F9", "surprised": "#22D3EE",
}

st.markdown('<div class="section-title">Most Similar RAVDESS Samples</div>', unsafe_allow_html=True)

if embeddings_cached():
    try:
        ravdess_embeddings, ravdess_metadata = _load_ravdess_embeddings()
        similar = find_similar(embedding, ravdess_embeddings, ravdess_metadata, k=5)

        st.caption(
            "Closest matches in the RAVDESS emotion reference set "
            "(1,440 labelled recordings from 24 actors)."
        )

        for rank, entry in enumerate(similar, start=1):
            emo_lower = entry["emotion"]
            emo_color = EMOTION_COLORS.get(emo_lower, "#A78BFA")
            emo_title = emo_lower.title()
            actor_num = entry["actor"]
            sim_pct = f"{entry['similarity']:.1%}"
            extra = f"intensity {entry['intensity']}" if entry['intensity'] else ""

            card_html = f"""
            <div class="similar-card">
                <span class="dot" style="background:{emo_color};"></span>
                <div class="info">
                    <div class="emotion">#{rank}  {emo_title}</div>
                    <div class="meta">Actor {actor_num:02d}  ·  {extra}</div>
                </div>
                <div class="score">{sim_pct} match</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Audio player below each card
            try:
                with open(entry["path"], "rb") as fh:
                    st.audio(fh.read(), format="audio/wav")
            except Exception:
                st.caption("_(audio file not found on disk)_")

    except Exception as e:
        st.caption(f"Could not compute similarity search: {e}")
else:
    st.caption(
        "RAVDESS reference embeddings not pre-computed. Run "
        "`uv run python -c \"from audiovec.search import precompute_ravdess_embeddings; "
        "precompute_ravdess_embeddings()\"` to enable similarity search."
    )


# -- Footer -------------------------------------------------------------------

st.markdown(
    '<div class="footer">'
    "audiovec &middot; 256-dimensional audio sentiment embedding &middot; RAVDESS dataset"
    "</div>",
    unsafe_allow_html=True,
)
