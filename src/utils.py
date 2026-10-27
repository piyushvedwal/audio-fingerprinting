"""Audio loading, validation and small helpers shared by the app and scripts."""
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from .fingerprint import SR

MIN_SECONDS = 2.0
MAX_SECONDS = 30.0
MAX_UPLOAD_MB = 10
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".m4a", ".flac")


class AudioError(Exception):
    """A problem with the audio that we can explain to the user in plain words."""


def load_audio_file(path, duration=None, offset=0.0) -> np.ndarray:
    """Decode any supported file -> mono float32 at 22050 Hz.

    librosa is imported lazily so the core DSP/tests never need it.
    """
    import librosa  # noqa: WPS433
    y, _ = librosa.load(str(path), sr=SR, mono=True, offset=offset, duration=duration)
    return y


def load_audio_bytes(data: bytes, filename: str = "clip.wav") -> np.ndarray:
    """Validate and decode an uploaded / recorded clip. Raises AudioError."""
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioError(f"File is too large (max {MAX_UPLOAD_MB} MB).")
    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in AUDIO_EXTS:
        raise AudioError("Unsupported format. Please use wav, mp3, ogg, m4a or flac.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            y = load_audio_file(tmp_path)
        except Exception as exc:  # decoder errors vary a lot between formats
            raise AudioError("Could not read this audio file. "
                             "Try a different file or format (wav/mp3 work best).") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    seconds = len(y) / SR
    if seconds < MIN_SECONDS:
        raise AudioError(f"Clip is too short ({seconds:.1f}s). Please use at least {MIN_SECONDS:.0f} seconds.")
    if seconds > MAX_SECONDS:
        raise AudioError(f"Clip is too long ({seconds:.0f}s). Please use at most {MAX_SECONDS:.0f} seconds.")
    if np.max(np.abs(y)) < 1e-4:
        raise AudioError("This clip looks silent. Please try another one.")
    return y


def add_noise(y: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio (dB). None = clean."""
    if snr_db is None:
        return y
    noise_power = np.mean(y ** 2) / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(noise_power), len(y)).astype(np.float32)


def ensure_db(path) -> None:
    """If the DB file is missing and FP_DB_URL is set, download it once.

    Use this when the .npz is too big for GitHub (e.g. host it on Hugging Face
    Hub / a GitHub Release and set FP_DB_URL as an env var or Streamlit secret).
    """
    path = Path(path)
    if path.exists():
        return
    url = os.environ.get("FP_DB_URL")
    if not url:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
