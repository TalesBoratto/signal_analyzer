from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from common import Signal

FILTER_TYPES = ("None", "High-pass", "Low-pass", "Band-pass")


def _butter_filter(signal: Signal, btype: str, wn, order: int) -> Signal:
    sos = butter(N=int(order), Wn=wn, btype=btype, fs=signal.fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal.data)
    except ValueError as exc:
        raise ValueError("Signal may be too short for the selected filter.") from exc
    return Signal(filtered, signal.fs, signal.name, signal.unity)


def validate_filter(filter_type, low_hz, high_hz, fs) -> str | None:
    """Returns an error message, or None when the settings are valid."""
    nyquist = fs / 2
    if filter_type == "None":
        return None
    if filter_type == "High-pass":
        if not 0 < low_hz < nyquist:
            return "High-pass cutoff must satisfy 0 < cutoff < fs/2."
    elif filter_type == "Low-pass":
        if not 0 < high_hz < nyquist:
            return "Low-pass cutoff must satisfy 0 < cutoff < fs/2."
    elif filter_type == "Band-pass":
        if not 0 < low_hz < high_hz < nyquist:
            return "Band-pass must satisfy 0 < low < high < fs/2."
    return None


def apply_filter(signal: Signal, filter_type, low_hz, high_hz, order) -> Signal:
    error = validate_filter(filter_type, low_hz, high_hz, signal.fs)
    if error:
        raise ValueError(error)
    if filter_type == "None":
        return signal
    if filter_type == "High-pass":
        return _butter_filter(signal, "highpass", low_hz, order)
    if filter_type == "Low-pass":
        return _butter_filter(signal, "lowpass", high_hz, order)
    return _butter_filter(signal, "bandpass", [low_hz, high_hz], order)


def compute_envelope(
    signal: Signal,
    filter_type="None",
    low_hz=None,
    high_hz=None,
    order=4,
    smooth_hz=None,
    smooth_order=4,
):
    """
    Envelope = |analytic signal| (Hilbert transform) of the filtered signal.
    Optionally smoothed by a zero-phase Butterworth low-pass.
    The envelope spectrum is computed after removing the envelope mean,
    otherwise the DC term (the carrier amplitude) hides everything else.
    """
    filtered = apply_filter(signal, filter_type, low_hz, high_hz, order)
    magnitude = np.abs(hilbert(filtered.data))

    envelope = Signal(
        magnitude, signal.fs, f"{signal.name} | envelope", signal.unity
    )

    if smooth_hz:
        if not 0 < smooth_hz < signal.fs / 2:
            raise ValueError("Envelope smoothing cutoff must satisfy 0 < cutoff < fs/2.")
        envelope = _butter_filter(envelope, "lowpass", smooth_hz, smooth_order)

    envelope_ac = envelope.remove_dc()

    return {
        "filtered": filtered,
        "envelope": envelope,
        "envelope_spectrum": envelope_ac.to_spectrum(),
    }
