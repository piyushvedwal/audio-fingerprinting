"""Mini Shazam: Streamlit UI."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.database import load_db
from src.fingerprint import SR, get_peaks, get_spectrogram
from src.match import MIN_RATIO, MIN_SCORE, identify
from src.plots import (plot_accuracy_lines, plot_heatmap, plot_offset_hist,
                       plot_peaks, plot_spectrogram)
from src.utils import AudioError, ensure_db, load_audio_bytes

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "fingerprints.npz"
SAMPLES = sorted((ROOT / "samples").glob("*.wav"))
RESULTS_CSV = ROOT / "assets" / "results.csv"

st.set_page_config(page_title="Mini Shazam", page_icon="🎵", layout="centered")


@st.cache_resource(show_spinner="Loading fingerprint database...")
def get_db():
    ensure_db(DB_PATH)               # downloads from FP_DB_URL if the file is missing
    return load_db(str(DB_PATH))


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)                   # free memory on the 1 GB server


st.title("🎵 Mini Shazam")
st.caption("Audio fingerprinting from scratch: spectrogram peaks → hashes → offset voting.")

if not DB_PATH.exists():
    st.error("Fingerprint database not found. Run `python scripts/build_db.py` first.")
    st.stop()
db = get_db()

tab_id, tab_how, tab_eval = st.tabs(["Identify", "How it works", "Evaluation results"])

# ------------------------------------------------------------------ Identify
with tab_id:
    st.write(f"Database: **{db.n_songs:,} tracks**, {db.n_hashes:,} fingerprints. "
             "Use a clip of 3-10 seconds.")

    uploaded = st.file_uploader("Upload a clip (wav, mp3, ogg, m4a, flac; max 10 MB)",
                                type=["wav", "mp3", "ogg", "m4a", "flac"])
    recorded = st.audio_input("...or record from your microphone") if hasattr(st, "audio_input") else None

    clicked = None
    if SAMPLES:
        st.write("...or try a sample clip:")
        cols = st.columns(min(len(SAMPLES), 4))
        for i, sample in enumerate(SAMPLES[:4]):
            if cols[i].button(f"Sample {i + 1}", use_container_width=True):
                clicked = sample
        if clicked:
            st.session_state["sample"] = str(clicked)

    # Decide which input to use: sample click > upload > recording > last sample
    data, fname = None, None
    if clicked:
        data, fname = clicked.read_bytes(), clicked.name
    elif uploaded:
        data, fname = uploaded.getvalue(), uploaded.name
    elif recorded:
        data, fname = recorded.getvalue(), "recording.wav"
    elif st.session_state.get("sample"):
        p = Path(st.session_state["sample"])
        data, fname = p.read_bytes(), p.name

    if data is not None:
        try:
            y = load_audio_bytes(data, fname)
            st.audio(data)
            with st.spinner("Listening..."):
                result = identify(y, db)
                S = get_spectrogram(y)
                p_times, p_freqs = get_peaks(S)
        except AudioError as exc:
            st.warning(str(exc))
        except Exception:
            st.error("Something went wrong while analysing this clip. Please try another one.")
        else:
            best = result.best
            if result.matched:
                st.success(f"### 🎶 {best['name']}\nGenre/folder: **{best['genre'] or 'n/a'}**")
            else:
                st.error(f"### No confident match\n{result.message}")
            if best:
                c1, c2, c3 = st.columns(3)
                c1.metric("Matching hashes", best["score"])
                c2.metric("Confidence (best ÷ 2nd)", f"{result.confidence:.1f}x")
                c3.metric("Clip starts at", f"{best['position_sec']:.1f}s")
                st.caption(f"A match needs ≥ {MIN_SCORE} aligned hashes and a ratio ≥ {MIN_RATIO}.")
                st.subheader("Top candidates")
                st.dataframe(pd.DataFrame(result.candidates)[["name", "genre", "score", "position_sec"]],
                             hide_index=True, use_container_width=True)
                st.subheader("Why this match?")
                show_fig(plot_spectrogram(S))
                show_fig(plot_peaks(S, p_times, p_freqs))
                show_fig(plot_offset_hist(result.hist_offsets_sec, result.hist_counts))

# --------------------------------------------------------------- How it works
with tab_how:
    st.markdown("""
**1. Spectrogram** – the clip is turned into a picture of *frequency vs time* (STFT).  
**2. Peaks** – only the loudest local points are kept (the *constellation map*). They survive noise.  
**3. Hashes** – each peak is paired with the next few peaks. A pair `(f1, f2, Δt)` is packed into one 30-bit integer.  
**4. Lookup** – all hashes live in one sorted numpy array; binary search finds every track containing a hash.  
**5. Offset voting** – for a true match, many hashes agree on *the same time offset* → a spike in the histogram. Wrong tracks give scattered offsets.
""")
    st.graphviz_chart("""
digraph {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3ff"];
  A [label="Audio clip"]; B [label="Spectrogram (STFT)"]; C [label="Peaks (constellation map)"];
  D [label="Hash pairs (f1, f2, Δt)"]; E [label="Sorted DB lookup (searchsorted)"];
  F [label="Offset voting (histogram)"]; G [label="Song + confidence"];
  A -> B -> C -> D -> E -> F -> G;
}""")

# ---------------------------------------------------------- Evaluation results
with tab_eval:
    if not RESULTS_CSV.exists():
        st.info("Run `python scripts/evaluate.py` to generate results.")
    else:
        df = pd.read_csv(RESULTS_CSV, dtype={"snr_db": str})
        st.write(f"Accuracy on **{int(df['n_queries'].iloc[0])} random queries per setting** "
                 "(clip length × added white noise).")
        show_fig(plot_heatmap(df))
        show_fig(plot_accuracy_lines(df))
        st.dataframe(df, hide_index=True, use_container_width=True)
