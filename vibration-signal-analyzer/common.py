from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class Signal:
    def __init__(
        self,
        data,
        fs: float,
        name: str = "",
        unity: str = "",
    ):
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

    def _new(self, data) -> Signal:
        return Signal(
            data=data,
            fs=self.fs,
            name=self.name,
            unity=self.unity,
        )

    def remove_dc(self) -> Signal:
        return self._new(self.data - np.mean(self.data))

    def highpass_filter(
        self,
        cutoff_hz: float,
        order: int = 4,
    ) -> Signal:
        if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < self.fs / 2:
            raise ValueError("Cutoff should be: 0 < cutoff < fs/2.")

        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 1
        ):
            raise ValueError("Filter order must be a positive integer value.")

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
            raise ValueError(
                "It was not possible to filter the signal. "
                "It should be too short for the selected filter order."
            ) from exc

        return self._new(filtered)

    def integrate(self) -> Signal:
        frequencies = np.fft.rfftfreq(self.size, d=self.dt)
        spectrum = np.fft.rfft(self.data)

        integrated = np.zeros_like(spectrum)
        valid = frequencies > 0

        integrated[valid] = (
            spectrum[valid] / (1j * 2.0 * np.pi * frequencies[valid])
        )

        if self.size % 2 == 0:
            integrated[-1] = 0.0

        return self._new(np.fft.irfft(integrated, n=self.size))

    def derivate_freq(self) -> Signal:
        frequencies = np.fft.rfftfreq(self.size, d=self.dt)
        spectrum = np.fft.rfft(self.data)

        differentiated = (
            spectrum * (1j * 2.0 * np.pi * frequencies)
        )
        differentiated[0] = 0.0

        if self.size % 2 == 0:
            differentiated[-1] = 0.0

        return self._new(
            np.fft.irfft(differentiated, n=self.size)
        )

    def derivate_time(self) -> Signal:
        return self._new(
            np.gradient(self.data, self.dt, edge_order=2)
        )