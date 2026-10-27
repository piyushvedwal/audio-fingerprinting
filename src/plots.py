"""Matplotlib figures used by the Streamlit app and the evaluation script."""
import matplotlib
matplotlib.use("Agg")                       # headless server: no display needed
import matplotlib.pyplot as plt
import numpy as np

from .fingerprint import HOP, N_FFT, SR

SNR_ORDER = ["clean", "20", "10", "0"]


def _extent(S):
    return [0, S.shape[1] * HOP / SR, 0, SR / 2]


def plot_spectrogram(S_db):
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(S_db, origin="lower", aspect="auto", extent=_extent(S_db),
                   cmap="magma", vmin=-80, vmax=0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.colorbar(im, ax=ax, label="dB"); fig.tight_layout()
    return fig


def plot_peaks(S_db, times, freqs):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(S_db, origin="lower", aspect="auto", extent=_extent(S_db),
              cmap="gray_r", vmin=-80, vmax=0, alpha=0.5)
    ax.scatter(times * HOP / SR, freqs * SR / N_FFT, s=6, c="red")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation map ({len(times)} peaks)")
    fig.tight_layout()
    return fig


def plot_offset_hist(offsets_sec, counts):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(offsets_sec, counts, width=0.02, color="tab:blue")
    ax.set_xlabel("Offset: position in song - position in clip (s)")
    ax.set_ylabel("Matching hashes")
    ax.set_title("Offset histogram (a sharp spike = real match)")
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    """df columns: clip_len_sec, snr_db (str), accuracy"""
    pivot = (df.pivot(index="clip_len_sec", columns="snr_db", values="accuracy")
               .reindex(columns=[c for c in SNR_ORDER if c in set(df["snr_db"])]))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels([f"{c}s" for c in pivot.index])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Clip length")
    ax.set_title("Identification accuracy")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}", ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    return fig


def plot_accuracy_lines(df):
    """Two charts: accuracy vs SNR (one line per clip length) and vs clip length."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 6))
    snrs = [s for s in SNR_ORDER if s in set(df["snr_db"])]
    for clip, g in df.groupby("clip_len_sec"):
        g = g.set_index("snr_db").reindex(snrs)
        a1.plot(snrs, g["accuracy"], marker="o", label=f"{clip}s clip")
    a1.set_xlabel("SNR (dB)"); a1.set_ylabel("Accuracy"); a1.set_ylim(0, 1.05)
    a1.set_title("Accuracy vs noise"); a1.legend(); a1.grid(alpha=0.3)
    for snr in snrs:
        g = df[df["snr_db"] == snr].sort_values("clip_len_sec")
        a2.plot(g["clip_len_sec"], g["accuracy"], marker="o", label=f"SNR {snr}")
    a2.set_xlabel("Clip length (s)"); a2.set_ylabel("Accuracy"); a2.set_ylim(0, 1.05)
    a2.set_title("Accuracy vs clip length"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig
