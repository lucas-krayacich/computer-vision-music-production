"""
Continuous soft-pad synthesiser.

Each note is a Voice: fundamental sine + harmonics + two detuned copies
(chorus effect). Voices carry an ADSR envelope. When the chord changes,
old voices fade out and new ones fade in — no clicks or pops.
"""
import numpy as np
import threading
from config import SAMPLE_RATE, PAD_ATTACK, PAD_RELEASE, PAD_VOLUME

TWO_PI = 2.0 * np.pi


class Voice:
    # Harmonic partials: (frequency-multiplier, relative-amplitude)
    PARTIALS = [
        (1.0,   1.00),   # fundamental
        (2.0,   0.45),   # 2nd harmonic
        (3.0,   0.18),   # 3rd harmonic
        (1.006, 0.32),   # detuned +10 cents  (chorus)
        (0.997, 0.22),   # detuned –5 cents   (chorus)
    ]
    TOTAL_AMP = sum(a for _, a in PARTIALS)

    def __init__(self, freq: float, sample_rate: int, attack_samples: int):
        self.freq = freq
        self.sr   = sample_rate
        self.attack_samples  = attack_samples
        self.release_samples = int(PAD_RELEASE * sample_rate)

        self.phases = np.zeros(len(self.PARTIALS))   # maintain continuity
        self.state  = "attack"
        self.env_pos = 0
        self.done   = False

    # ── public ────────────────────────────────────────────────────────────────

    def start_release(self):
        if self.state != "release":
            self.state   = "release"
            self.env_pos = 0

    def generate(self, n_frames: int) -> np.ndarray:
        t = np.arange(n_frames, dtype=np.float32) / self.sr
        signal = np.zeros(n_frames, dtype=np.float32)

        for i, (mul, amp) in enumerate(self.PARTIALS):
            f = self.freq * mul
            wave = amp * np.sin(TWO_PI * f * t + self.phases[i])
            signal += wave
            self.phases[i] = (self.phases[i] + TWO_PI * f * n_frames / self.sr) % TWO_PI

        signal /= self.TOTAL_AMP
        signal *= self._envelope(n_frames)
        return signal

    # ── private ───────────────────────────────────────────────────────────────

    def _envelope(self, n_frames: int) -> np.ndarray:
        env = np.ones(n_frames, dtype=np.float32)

        if self.state == "attack":
            pos  = self.env_pos
            left = self.attack_samples - pos
            if left >= n_frames:
                env = np.linspace(pos / self.attack_samples,
                                  (pos + n_frames) / self.attack_samples,
                                  n_frames, dtype=np.float32)
                self.env_pos += n_frames
                if self.env_pos >= self.attack_samples:
                    self.state = "sustain"
            else:
                env[:left] = np.linspace(pos / self.attack_samples, 1.0,
                                         left, dtype=np.float32)
                env[left:]  = 1.0
                self.state   = "sustain"
                self.env_pos = 0

        elif self.state == "release":
            left = self.release_samples - self.env_pos
            if left <= 0:
                self.done = True
                return np.zeros(n_frames, dtype=np.float32)
            start_amp = 1.0 - self.env_pos / self.release_samples
            if left >= n_frames:
                end_amp = 1.0 - (self.env_pos + n_frames) / self.release_samples
                env = np.linspace(start_amp, max(end_amp, 0.0),
                                  n_frames, dtype=np.float32)
                self.env_pos += n_frames
            else:
                env[:left] = np.linspace(start_amp, 0.0, left, dtype=np.float32)
                env[left:]  = 0.0
                self.done   = True

        return env


class PadSynth:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr              = sample_rate
        self.attack_samples  = int(PAD_ATTACK * sample_rate)
        self.lock            = threading.Lock()
        self.voices: list[Voice] = []
        self.current_freqs: list[float] = []

    def set_chord(self, freqs: list[float]):
        with self.lock:
            if freqs == self.current_freqs:
                return
            for v in self.voices:
                v.start_release()
            for f in freqs:
                self.voices.append(Voice(f, self.sr, self.attack_samples))
            self.current_freqs = list(freqs)

    def silence(self):
        with self.lock:
            for v in self.voices:
                v.start_release()
            self.current_freqs = []

    def generate(self, n_frames: int) -> np.ndarray:
        out = np.zeros(n_frames, dtype=np.float32)
        with self.lock:
            for v in self.voices:
                out += v.generate(n_frames)
            self.voices = [v for v in self.voices if not v.done]
        return out * PAD_VOLUME
