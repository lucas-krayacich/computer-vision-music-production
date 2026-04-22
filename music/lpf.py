"""
Real-time low-pass filter using a 2nd-order Butterworth biquad.

scipy.signal.lfilter is implemented in C so it's safe to call from the
audio callback. Filter state (zi) is carried between callbacks so there
are no discontinuities at buffer boundaries.
"""
import threading
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from config import LPF_CUTOFFS_HZ, LPF_DEFAULT_IDX, LPF_ENABLED_DEFAULT, SAMPLE_RATE


class LowPassFilter:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr       = sample_rate
        self.enabled  = LPF_ENABLED_DEFAULT
        self._lock    = threading.Lock()
        self._cutoff_idx = LPF_DEFAULT_IDX
        self._b = self._a = self._zi = None
        self._build(LPF_CUTOFFS_HZ[LPF_DEFAULT_IDX])

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def cutoff_hz(self) -> int:
        return LPF_CUTOFFS_HZ[self._cutoff_idx]

    def step_cutoff(self, direction: int):
        """direction: +1 to raise cutoff, -1 to lower."""
        new_idx = self._cutoff_idx + direction
        new_idx = max(0, min(len(LPF_CUTOFFS_HZ) - 1, new_idx))
        if new_idx != self._cutoff_idx:
            self._cutoff_idx = new_idx
            self._build(LPF_CUTOFFS_HZ[new_idx])

    def process(self, signal: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return signal
        with self._lock:
            filtered, self._zi = lfilter(self._b, self._a, signal, zi=self._zi)
        return filtered.astype(np.float32)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build(self, cutoff_hz: float):
        b, a = butter(2, cutoff_hz / (self.sr / 2), btype="low")
        zi   = lfilter_zi(b, a) * 0.0   # zero initial state
        with self._lock:
            self._b  = b
            self._a  = a
            self._zi = zi
