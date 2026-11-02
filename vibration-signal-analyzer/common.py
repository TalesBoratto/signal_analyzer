from __future__ import annotations
from functools import cached_property

import numpy as np
from scipy.signal import butter, sosfiltfilt

from feature_extraction import (
    FEATURE_NAMES, TEMPORAL_FEATURES, SPECTRAL_FEATURES,
    select_names, extract_temporal,
)


class Signal:
    def __init__(self, data, fs: float, name: str = "", unity: str = ""):
        values = np.asarray(data)

        if np.iscomplexobj(values):
            raise ValueError("Signal must be real.")

        self.data = np.array(values, dtype=float, copy=True)
        self.fs = float(fs)
        self.name = str(name)
        self.unity = str(unity)

        if self.data.ndim != 1:
            raise ValueError("Only unidimensional signal is available.")

        if self.data.size < 3:
            raise ValueError("The signal must have at least 3 samples.")

        if not np.all(np.isfinite(self.data)):
            raise ValueError("Signal contains NaN ou infinity values.")

        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("Sampling rate must be postive.")

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.size) / self.fs

    @property
    def duration_in_sec(self):
        return self.size / self.fs

    def _new(self, data):
        return Signal(data, self.fs, self.name, self.unity)

    def remove_dc(self) -> Signal:
        return self._new(self.data - np.mean(self.data))

    def highpass_filter(
        self,
        cutoff_hz: float,
        order: int = 4,
    ) -> Signal:
        if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < self.fs / 2:
            raise ValueError("Cutoff must satisfy 0 < cutoff < fs/2.")

        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 1
        ):
            raise ValueError("Order must be a positive integer.")

        sos = butter(
            N=int(order),
            Wn=cutoff_hz,
            btype="highpass",
            fs=self.fs,
            output="sos",
        )

        try:
            filtered = sosfiltfilt(sos, self.data)
        except ValueError as exc:
            raise ValueError("Signal may be too short for the selected filter.") from exc

        return self._new(filtered)

    def to_spectrum(self):
        # No Signal-level cache: data and fs remain publicly mutable.
        return Spectrum.from_signal(self)

    def spectrum(self, scaling="amplitude"):
        """Compatibility helper. Reuse to_spectrum() explicitly for efficiency."""
        spectrum = self.to_spectrum()
        if scaling == "amplitude":
            return spectrum.frequencies, spectrum.amplitude
        if scaling == "power":
            return spectrum.frequencies, spectrum.power
        raise ValueError("scaling must be amplitude or power.")
    
    def integrate(self, *, name=None, unity=None):
        return self.to_spectrum().integrate(name=name, unity=unity).to_signal()

    def derivate_freq(self, *, name=None, unity=None):
        return self.to_spectrum().differentiate(name=name, unity=unity).to_signal()

    def derivate_time(self, *, name=None, unity=None):
        return Signal(np.gradient(self.data, self.dt, edge_order=2), self.fs,
                        self.name if name is None else name,
                        f"({self.unity})/s" if unity is None else unity)

    def extract_temporal_features(self, selected=None):
        return extract_temporal(self.data, selected)

    def extract_spectral_features(self, selected=None, *, exclude_dc=True):
        return self.to_spectrum().extract_features(selected, exclude_dc=exclude_dc)

    def extract_features(self, selected=None, *, spectral_exclude_dc=True):
        names = select_names(selected, FEATURE_NAMES)
        temporal = [name for name in names if name in TEMPORAL_FEATURES]
        spectral = [name for name in names if name in SPECTRAL_FEATURES]
        values = self.extract_temporal_features(temporal)
        if spectral:
            values.update(self.extract_spectral_features(
                spectral, exclude_dc=spectral_exclude_dc))
        return {name: values[name] for name in names}

    
class Spectrum:
    """One-sided complex DFT of an unwindowed real Signal.

    Stores a snapshot, not a live link to Signal.data.
    Power means mean-square contribution per bin, not PSD.
    """
    def __init__(self, coefficients, n_samples, fs, name="", unity=""):

        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise ValueError("n_samples must be an integer.")
        
        if n_samples < 3:
            raise ValueError("At least 3 samples are required.")
        
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be finite and positive.")
        
        coefficients = np.array(coefficients, dtype=complex, copy=True)
        if coefficients.ndim != 1 or coefficients.size != n_samples // 2 + 1:
            raise ValueError("Invalid number of rFFT coefficients.")
        
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Coefficients must be finite.")
        
        if coefficients[0].imag != 0:
            raise ValueError("DC must be real for a real signal.")
        
        if n_samples % 2 == 0 and coefficients[-1].imag != 0:
            raise ValueError("Nyquist must be real for an even-length real signal.")
        
        coefficients.setflags(write=False)
        self._coefficients = coefficients
        self._n_samples = int(n_samples)
        self._fs = float(fs)
        self.name = str(name)
        self.unity = str(unity)


    @classmethod
    def from_signal(cls, signal):
        return cls(np.fft.rfft(signal.data), signal.size, signal.fs,
                   signal.name, signal.unity)


    @property
    def coefficients(self):
        view = self._coefficients.view()
        view.setflags(write=False)
        return view


    @property
    def n_samples(self):
        return self._n_samples


    @property
    def fs(self):
        return self._fs


    @cached_property
    def frequencies(self):
        values = np.fft.rfftfreq(self.n_samples, d=1.0 / self.fs)
        values.setflags(write=False)
        return values


    def _one_sided(self, values):
        if self.n_samples % 2 == 0:
            values[1:-1] *= 2.0
        else:
            values[1:] *= 2.0
        values.setflags(write=False)
        return values


    @cached_property
    def amplitude(self):
        return self._one_sided(np.abs(self._coefficients) / self.n_samples)


    @cached_property
    def power(self):
        return self._one_sided(np.abs(self._coefficients)**2 / self.n_samples**2)


    @cached_property
    def phase(self):
        values = np.angle(self._coefficients)
        values.setflags(write=False)
        return values


    def to_signal(self):
        return Signal(np.fft.irfft(self._coefficients, n=self.n_samples),
                      self.fs, self.name, self.unity)


    def integrate(self, *, name=None, unity=None):
        omega = 2.0 * np.pi * self.frequencies
        values = np.zeros_like(self._coefficients)
        valid = omega > 0
        values[valid] = self._coefficients[valid] / (1j * omega[valid])
        if self.n_samples % 2 == 0:
            values[-1] = 0.0
        return Spectrum(values, self.n_samples, self.fs,
                        self.name if name is None else name,
                        f"({self.unity})·s" if unity is None else unity)


    def differentiate(self, *, name=None, unity=None):
        values = self._coefficients * (1j * 2.0 * np.pi * self.frequencies)
        values[0] = 0.0
        if self.n_samples % 2 == 0:
            values[-1] = 0.0
        return Spectrum(values, self.n_samples, self.fs,
                        self.name if name is None else name,
                        f"({self.unity})/s" if unity is None else unity)


    def extract_features(self, selected=None, *, exclude_dc=True):
        """Power-weighted frequency moments; Pearson kurtosis.

        No filtering or waveform mean removal is performed here.
        exclude_dc=True removes only the DC bin from the moments.
        This is not a segment-based spectral-kurtosis estimator.
        """
        names = select_names(selected, SPECTRAL_FEATURES)
        if not names:
            return {}
        
        f = self.frequencies
        p = self.power
        if exclude_dc:
            f, p = f[1:], p[1:]

        values = {name: np.nan for name in SPECTRAL_FEATURES}
        total = float(np.sum(p))
        if total <= 0 or not np.isfinite(total):
            return {name: values[name] for name in names}
        
        weights = p / total
        centroid = float(np.sum(weights * f))
        values["Spectral Centroid"] = centroid

        if set(names) & {"Spectral Skewness", "Spectral Kurtosis"}:
            centered = f - centroid
            variance = float(np.sum(weights * centered**2))
            tolerance = np.finfo(float).eps * max(self.fs, 1.0)**2

            if variance > tolerance:
                if "Spectral Skewness" in names:
                    values["Spectral Skewness"] = float(
                        np.sum(weights * centered**3) / variance**1.5)
                    
                if "Spectral Kurtosis" in names:
                    values["Spectral Kurtosis"] = float(
                        np.sum(weights * centered**4) / variance**2)
                    
        return {name: values[name] for name in names}
