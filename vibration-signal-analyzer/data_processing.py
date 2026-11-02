from __future__ import annotations

import numpy as np
import pandas as pd

from common import Signal, Spectrum
from feature_extraction import (
    TEMPORAL_FEATURES, SPECTRAL_FEATURES, FEATURE_NAMES, select_names,
)


def error_metrics(reference, estimate):
    error = reference.data - estimate.data
    rmse = float(np.sqrt(np.mean(error**2)))
    ref_rms = features(reference)["RMS"]
    correlation = np.nan
    if np.std(reference.data) > 0 and np.std(estimate.data) > 0:
        correlation = float(np.corrcoef(reference.data, estimate.data)[0, 1])
    return {
        "Bias (Ref − Est)": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": rmse,
        "Max Abs Error": float(np.max(np.abs(error))),
        "NRMSE (%)": 100 * rmse / ref_rms if ref_rms > 0 else np.nan,
        "Correlation": correlation,
    }


def features(signal: Signal) -> dict:
    x = signal.data

    return {
        "Mean Value / DC": np.round(float(np.mean(x)),6),
        "RMS": np.round(float(np.sqrt(np.mean(x**2))),6),
        "Max Value": np.round(float(np.max(x)),6),
        "Min Value": np.round(float(np.min(x)),6),
        "Abs Peak": np.round(float(np.max(np.abs(x))),6),
        "Peak-to-Peak": np.round(float(np.ptp(x)),6),
    }


def amplitude_spectrum(value):
    spectrum = value if isinstance(value, Spectrum) else value.to_spectrum()
    return spectrum.frequencies, spectrum.amplitude


def process(data, fs, cutoff, order, name, unit=""):
    original = Signal(data, fs, name, unit)
    acc_dc = original.remove_dc()
    acc_filtered = acc_dc.highpass_filter(cutoff, order)

    input_spectrum = acc_dc.to_spectrum()
    filtered_spectrum = acc_filtered.to_spectrum()
    velocity_spectrum = filtered_spectrum.integrate(
        name=f"{name} | velocity", unity=f"({unit})·s")
    velocity = velocity_spectrum.to_signal()
    reconstructed_spectrum = velocity_spectrum.differentiate(
        name=f"{name} | reconstructed FFT", unity=unit)
    acc_fft = reconstructed_spectrum.to_signal()
    acc_time = velocity.derivate_time(
        name=f"{name} | reconstructed finite differences", unity=unit)

    signals = {
        "Original": original, "Após DC": acc_dc,
        "Após DC + HPF": acc_filtered,
        "Reconstruída FFT": acc_fft,
        "Reconstruída diferenças finitas": acc_time,
    }
    points = pd.DataFrame({
        "time_s": original.time, "acc_original": original.data,
        "acc_dc": acc_dc.data, "acc_filtered": acc_filtered.data,
        "velocity": velocity.data, "acc_fft": acc_fft.data,
        "acc_time": acc_time.data,
    })
    for method, estimate in (("fft", acc_fft), ("time", acc_time)):
        error = acc_dc.data - estimate.data
        relative = np.full(acc_dc.size, np.nan)
        np.divide(error, acc_dc.data, out=relative, where=acc_dc.data != 0)
        points[f"error_{method}"] = error
        points[f"abs_error_{method}"] = np.abs(error)
        points[f"relative_error_{method}"] = relative
        points[f"relative_error_percent_{method}"] = 100 * relative

    return {
        "original": original, "acc_dc": acc_dc,
        "acc_filtered": acc_filtered, "velocity": velocity,
        "acc_fft": acc_fft, "acc_time": acc_time,
        "input_spectrum": input_spectrum,
        "filtered_spectrum": filtered_spectrum,
        "velocity_spectrum": velocity_spectrum,
        "acc_fft_spectrum": reconstructed_spectrum,
        "points": points,
        "acceleration_metrics": pd.DataFrame({
            label: features(signal) for label, signal in signals.items()}).T,
        "velocity_metrics": pd.DataFrame([features(velocity)], index=["Velocidade estimada"]),
        "reconstruction_metrics": pd.DataFrame({
            "Derivação FFT": error_metrics(acc_dc, acc_fft),
            "Diferenças finitas": error_metrics(acc_dc, acc_time)}).T,
    }


def build_feature_table(points, fs, selected):
    """Compatibility with the previous app API."""
    columns = {
        "Original acceleration": "acc_original",
        "Acceleration after DC": "acc_dc",
        "Reconstructed — FFT": "acc_fft",
        "Reconstructed — finite differences": "acc_time",
        "Estimated velocity": "velocity",
    }
    return pd.DataFrame({
        label: Signal(points[column].to_numpy(), fs).extract_features(selected)
        for label, column in columns.items()
    }).reindex(selected)


def build_feature_table_from_result(result, selected):
    """Reuses existing spectra; computes new ones only when required."""
    names = select_names(selected, FEATURE_NAMES)
    temporal = [name for name in names if name in TEMPORAL_FEATURES]
    spectral = [name for name in names if name in SPECTRAL_FEATURES]
    objects = {
        "Original acceleration": ("original", None),
        "Acceleration after DC": ("acc_dc", "input_spectrum"),
        "Reconstructed — FFT": ("acc_fft", "acc_fft_spectrum"),
        "Reconstructed — finite differences": ("acc_time", None),
        "Estimated velocity": ("velocity", "velocity_spectrum"),
    }
    output = {}
    for label, (signal_key, spectrum_key) in objects.items():
        signal = result[signal_key]
        values = signal.extract_temporal_features(temporal)
        if spectral:
            if spectrum_key is None:
                spectrum_key = f"_feature_spectrum_{signal_key}"
                if spectrum_key not in result:
                    result[spectrum_key] = signal.to_spectrum()
            values.update(result[spectrum_key].extract_features(spectral, exclude_dc=True))
        output[label] = {name: values[name] for name in names}
    return pd.DataFrame(output).reindex(names)