from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

TEMPORAL_FEATURES = (
    "Mean Absolute Value (MAV)", "Standard Deviation (STD)", "RMS",
    "Peak", "Peak-to-Peak", "Skewness", "Kurtosis", "Energy",
    "Crest Factor", "Impulse Factor", "Shape Factor", "Zero Crossing Rate",
)
SPECTRAL_FEATURES = (
    "Spectral Skewness", "Spectral Kurtosis", "Spectral Centroid",
)
FEATURE_NAMES = list(TEMPORAL_FEATURES + SPECTRAL_FEATURES)


def select_names(selected, available):
    names = list(dict.fromkeys(available if selected is None else selected))
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    return names


def extract_temporal(data, selected=None):
    names = select_names(selected, TEMPORAL_FEATURES)
    if not names:
        return {}
    x = np.asarray(data, dtype=float)
    values = {}
    need_mav = bool(set(names) & {
        "Mean Absolute Value (MAV)", "Impulse Factor", "Shape Factor"})
    need_rms = bool(set(names) & {
        "Energy", "RMS", "Crest Factor", "Shape Factor"})
    need_peak = bool(set(names) & {"Peak", "Crest Factor", "Impulse Factor"})
    if need_mav:
        mav = float(np.mean(np.abs(x)))
        values["Mean Absolute Value (MAV)"] = mav
    if need_rms:
        energy = float(np.dot(x, x))
        rms = float(np.sqrt(energy / x.size))
        values.update({"Energy": energy, "RMS": rms})
    if need_peak:
        peak = float(np.max(np.abs(x)))
        values["Peak"] = peak
    if set(names) & {"Standard Deviation (STD)", "Skewness", "Kurtosis"}:
        std = float(np.std(x, ddof=0))
        values["Standard Deviation (STD)"] = std
        if "Skewness" in names:
            values["Skewness"] = float(skew(x, bias=True)) if std > 0 else np.nan
        if "Kurtosis" in names:
            values["Kurtosis"] = float(kurtosis(x, fisher=False, bias=True)) if std > 0 else np.nan
    if "Peak-to-Peak" in names:
        values["Peak-to-Peak"] = float(np.ptp(x))
    if "Crest Factor" in names:
        values["Crest Factor"] = peak / rms if rms > 0 else np.nan
    if "Impulse Factor" in names:
        values["Impulse Factor"] = peak / mav if mav > 0 else np.nan
    if "Shape Factor" in names:
        values["Shape Factor"] = rms / mav if mav > 0 else np.nan
    if "Zero Crossing Rate" in names:
        values["Zero Crossing Rate"] = float(np.count_nonzero(
            np.signbit(x[1:]) != np.signbit(x[:-1])) / (x.size - 1))
    return {name: values[name] for name in names}


def reconstruction_feature_difference(table):
    reference = table["Acceleration after DC"]
    estimates = table[["Reconstructed — FFT", "Reconstructed — finite differences"]]
    denominator = reference.abs().replace(0.0, np.nan)
    return estimates.rsub(reference, axis=0).div(denominator, axis=0) * 100.0
