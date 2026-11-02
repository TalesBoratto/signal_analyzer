from __future__ import annotations

import numpy as np
import pandas as pd

from common import Signal


def error_metrics(reference: Signal, estimate: Signal) -> dict:
    error = reference.data - estimate.data

    rmse = float(np.sqrt(np.mean(error**2)))
    ref_rms = features(reference)["RMS"]

    correlation = np.nan
    if np.std(reference.data) > 0 and np.std(estimate.data) > 0:
        correlation = float(
            np.corrcoef(reference.data, estimate.data)[0, 1]
        )

    return {
        "Bias (Ref − Est)": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": rmse,
        "Erro absoluto máximo": float(np.max(np.abs(error))),
        "NRMSE (%)": 100.0 * rmse / ref_rms if ref_rms > 0 else np.nan,
        "Correlação": correlation,
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


def amplitude_spectrum(signal: Signal):
    """FFT unilateral de amplitude, sem janela e sem HPF."""
    n = signal.size
    frequency = np.fft.rfftfreq(n, d=signal.dt)
    amplitude = np.abs(np.fft.rfft(signal.data)) / n

    if n % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0

    return frequency, amplitude


def process(data, fs, cutoff, order, name, unit:str=""):
    original = Signal(data, fs, name=name, unity=unit)
    acc_dc = original.remove_dc()
    acc_filtered = acc_dc.highpass_filter(cutoff, order)

    velocity = acc_filtered.integrate()
    velocity.unity = f"({unit})·s"

    acc_fft = velocity.derivate_freq()
    acc_fft.unity = unit

    acc_time = velocity.derivate_time()
    acc_time.unity = unit

    signals = {
        "Original": original,
        "After DC": acc_dc,
        "After DC + HPF": acc_filtered,
        "Rebuild via FFT": acc_fft,
        "Rebuild via time derivation": acc_time,
    }

    points = pd.DataFrame({
        "time_s": original.time,
        "acc_original": original.data,
        "acc_dc": acc_dc.data,
        "acc_filtered": acc_filtered.data,
        "velocity": velocity.data,
        "acc_fft": acc_fft.data,
        "acc_time": acc_time.data,
    })

    for method, estimate in (
        ("fft", acc_fft),
        ("time", acc_time),
    ):
        error = acc_dc.data - estimate.data
        relative = np.full(acc_dc.size, np.nan)

        np.divide(
            error,
            acc_dc.data,
            out=relative,
            where=acc_dc.data != 0.0,
        )

        points[f"error_{method}"] = error
        points[f"relative_error_{method}"] = relative
        points[f"relative_error_percent_{method}"] = 100.0 * relative

    acceleration_metrics = pd.DataFrame({
        label: features(signal)
        for label, signal in signals.items()
    }).T

    velocity_metrics = pd.DataFrame(
        [features(velocity)],
        index=["Estimated Velocity"],
    )

    reconstruction_metrics = pd.DataFrame({
        "FFT Derivation": error_metrics(acc_dc, acc_fft),
        "Time Derivation": error_metrics(acc_dc, acc_time),
    }).T

    return {
        "original": original,
        "acc_dc": acc_dc,
        "velocity": velocity,
        "points": points,
        "acceleration_metrics": acceleration_metrics,
        "velocity_metrics": velocity_metrics,
        "reconstruction_metrics": reconstruction_metrics,
    }
