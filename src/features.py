"""
features.py

Single source of truth for turning a raw EEG window into a feature
vector. Both the training pipeline (dataset.py -> train.py) and the
live Streamlit app import THIS SAME function, so train/live can never
drift apart.

Parameterized by `fs` (sampling rate) because the training dataset
(NeuroSky-style chip, 512Hz) and your live BioAmp+Arduino (~250Hz)
are not the same rate. Band definitions are in Hz, so as long as fs
comfortably covers each band (Nyquist = fs/2), the relative-power and
ratio features are comparable across sample rates.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FEATURE_NAMES = [f"rel_{b}" for b in BANDS] + [
    "alpha_beta_ratio",
    "theta_beta_ratio",
    "alpha_theta_ratio",
]

MIN_SAMPLES_FOR_PSD = 32  # below this, PSD estimate is too unreliable


def _bandpass(signal, fs, low=1.0, high=45.0, order=4):
    nyq = fs / 2.0
    high = min(high, nyq - 1.0)  # guard against invalid filter design at low fs
    if high <= low:
        return signal
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def _notch(signal, fs, freq=50.0, quality=30.0):
    nyq = fs / 2.0
    if freq >= nyq:
        return signal  # can't notch above Nyquist, skip safely
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, signal)


def _band_powers(signal, fs):
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        powers[name] = float(trapz_fn(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def extract_features(raw_window, fs):
    """
    raw_window: 1D array-like of raw EEG samples (any length >= MIN_SAMPLES_FOR_PSD)
    fs: sampling rate in Hz for THIS window

    Returns a dict of feature_name -> value, or None if the window is
    too short / degenerate to extract meaningful features from.
    """
    raw_window = np.asarray(raw_window, dtype=np.float64)
    if len(raw_window) < MIN_SAMPLES_FOR_PSD:
        return None
    if np.std(raw_window) < 1e-9:
        return None  # flat/dead signal, e.g. disconnected electrode

    filtered = _bandpass(raw_window, fs)
    filtered = _notch(filtered, fs)
    powers = _band_powers(filtered, fs)

    total = sum(powers.values()) + 1e-12
    feats = {f"rel_{k}": v / total for k, v in powers.items()}
    feats["alpha_beta_ratio"] = powers["alpha"] / (powers["beta"] + 1e-12)
    feats["theta_beta_ratio"] = powers["theta"] / (powers["beta"] + 1e-12)
    feats["alpha_theta_ratio"] = powers["alpha"] / (powers["theta"] + 1e-12)
    return feats
