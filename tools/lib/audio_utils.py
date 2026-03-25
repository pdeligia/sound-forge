"""Audio I/O and analysis utilities for SoundForge tools."""

import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    Samples are returned as a 1D float32 array (mono) or 2D (channels, samples).
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels) — transpose to (channels, samples)
    data = data.T
    if data.shape[0] == 1:
        data = data[0]
    return data, sr


def save_audio(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Save audio samples to a WAV file.

    Accepts 1D (mono) or 2D (channels, samples) float32 arrays.
    """
    if data.ndim == 2:
        data = data.T  # soundfile expects (samples, channels)
    sf.write(path, data, sample_rate)


def get_duration(data: np.ndarray, sample_rate: int) -> float:
    """Return duration in seconds."""
    n_samples = data.shape[-1] if data.ndim > 1 else data.shape[0]
    return n_samples / sample_rate


def compute_rms(data: np.ndarray) -> float:
    """Compute RMS (root mean square) level of audio."""
    flat = data.flatten() if data.ndim > 1 else data
    return float(np.sqrt(np.mean(flat**2)))


def compute_peak(data: np.ndarray) -> float:
    """Compute peak absolute amplitude."""
    flat = data.flatten() if data.ndim > 1 else data
    return float(np.max(np.abs(flat)))


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate using torchaudio."""
    if orig_sr == target_sr:
        return data

    import torch
    import torchaudio.transforms as T

    was_1d = data.ndim == 1
    if was_1d:
        data = data[np.newaxis, :]

    tensor = torch.from_numpy(data)
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(tensor).numpy()

    if was_1d:
        resampled = resampled[0]
    return resampled


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the mean from each channel."""
    if data.ndim > 1:
        return data - np.mean(data, axis=1, keepdims=True)
    return data - np.mean(data)


def highpass_filter(data: np.ndarray, sample_rate: int, cutoff_hz: float = 150.0) -> np.ndarray:
    """Apply a high-pass filter to remove low-frequency rumble/hum.

    Uses a second-order Butterworth biquad, applied forward+backward for zero phase shift.
    """
    w0 = 2 * np.pi * cutoff_hz / sample_rate
    alpha = np.sin(w0) / (2 * 0.707)  # Q = 0.707 (Butterworth)
    cos_w0 = np.cos(w0)

    a0 = 1 + alpha
    b = np.array([(1 + cos_w0) / 2 / a0, -(1 + cos_w0) / a0, (1 + cos_w0) / 2 / a0])
    a = np.array([1.0, -2 * cos_w0 / a0, (1 - alpha) / a0])

    def _apply(b, a, x):
        y = np.zeros_like(x)
        for n in range(2, len(x)):
            y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]
        return y

    # Forward-backward for zero phase shift
    forward = _apply(b, a, data)
    return _apply(b, a, forward[::-1])[::-1]
