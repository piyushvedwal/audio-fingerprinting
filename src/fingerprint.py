"""
Core audio-fingerprinting DSP (Shazam-style).

Pipeline:  audio -> spectrogram -> peaks (constellation map) -> hashes

This module depends ONLY on numpy + scipy, so it is light, fast to import,
and easy to test. Decoding mp3/m4a/ogg lives in utils.py (uses librosa).
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft

# ---------------------------------------------------------------- parameters
SR = 22050            # sample rate everything is resampled to
N_FFT = 2048          # FFT window  -> 1025 frequency bins
HOP = 512             # hop length  -> ~23 ms per frame
NEIGHBORHOOD = 20     # a peak must be the maximum of a 20x20 (freq x time) area
AMP_MIN_DB = -40      # ignore points quieter than this (relative to loudest = 0 dB)
FAN_OUT = 10          # each peak is paired with the next FAN_OUT peaks
MIN_DT = 1            # min / max time gap (in frames) between two paired peaks
MAX_DT = 200          # (must fit in 8 bits, i.e. <= 255)

# Bit layout of a packed hash (30 bits total -> fits in uint32):
#   | f1 : 11 bits | f2 : 11 bits | dt : 8 bits |
F_BITS, DT_BITS = 11, 8


def get_spectrogram(y: np.ndarray) -> np.ndarray:
    """Return a magnitude spectrogram in dB, shape (n_freq_bins, n_frames).

    The signal is peak-normalised first, so a quiet phone recording and a
    loud studio track end up on the same dB scale (0 dB = loudest point).
    """
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    _, _, Z = stft(y, fs=SR, window="hann", nperseg=N_FFT,
                   noverlap=N_FFT - HOP, boundary=None, padded=False)
    mag = np.abs(Z)
    ref = mag.max() if mag.size and mag.max() > 0 else 1.0
    return 20.0 * np.log10(np.maximum(mag, 1e-10) / ref)


def get_peaks(S_db: np.ndarray):
    """Find the constellation map: local maxima that are loud enough.

    Returns two int arrays (times, freqs), sorted by time then frequency,
    so the output is fully deterministic.
    """
    is_local_max = maximum_filter(S_db, size=NEIGHBORHOOD) == S_db
    freq_idx, time_idx = np.nonzero(is_local_max & (S_db > AMP_MIN_DB))
    order = np.lexsort((freq_idx, time_idx))          # primary key: time
    return time_idx[order], freq_idx[order]


def pack_hash(f1, f2, dt):
    """Pack (f1, f2, dt) into ONE integer using bit-shifts."""
    return ((f1.astype(np.uint32) << (F_BITS + DT_BITS))
            | (f2.astype(np.uint32) << DT_BITS)
            | dt.astype(np.uint32))


def generate_hashes(times: np.ndarray, freqs: np.ndarray):
    """Pair every peak with the next FAN_OUT peaks.

    Returns (hashes uint32, anchor_times int32). The hash does not depend on
    *where* in the song the pair occurs, only on (f1, f2, dt), which is why
    a clip cut from anywhere still matches.
    """
    n = len(times)
    all_h, all_t = [], []
    for j in range(1, FAN_OUT + 1):
        if n <= j:
            break
        t1, t2 = times[:-j], times[j:]
        f1, f2 = freqs[:-j], freqs[j:]
        dt = t2 - t1
        ok = (dt >= MIN_DT) & (dt <= MAX_DT)
        all_h.append(pack_hash(f1[ok], f2[ok], dt[ok]))
        all_t.append(t1[ok].astype(np.int32))
    if not all_h:
        return np.empty(0, np.uint32), np.empty(0, np.int32)
    return np.concatenate(all_h), np.concatenate(all_t)


def fingerprint(y: np.ndarray):
    """Full pipeline: raw audio -> (hashes, times)."""
    t, f = get_peaks(get_spectrogram(y))
    return generate_hashes(t, f)
