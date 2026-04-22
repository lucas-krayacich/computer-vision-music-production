"""
16-step drum machine with 5 house patterns.

All sounds are synthesised with numpy/scipy at startup.
generate() is called from the audio callback and mixes active hits
sample-accurately against the sequencer clock.
"""
import numpy as np
import threading
from scipy.signal import butter, lfilter
from config import SAMPLE_RATE, BPM, DRUM_VOLUME, DRUM_PATTERN_NAMES

# ── Sound synthesis helpers ───────────────────────────────────────────────────

def _hp(signal: np.ndarray, cutoff: float, sr: int, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (sr / 2), btype="high")
    return lfilter(b, a, signal)

def _bp(signal: np.ndarray, lo: float, hi: float, sr: int, order: int = 2) -> np.ndarray:
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return lfilter(b, a, signal)

def _make_kick(sr: int) -> np.ndarray:
    dur = 0.55
    n = int(sr * dur)
    t = np.linspace(0, dur, n)
    # Frequency sweep 150 → 40 Hz
    freq  = 150.0 * np.exp(-18.0 * t) + 40.0
    phase = np.cumsum(2 * np.pi * freq / sr)
    body  = np.sin(phase) * np.exp(-7.0 * t)
    click = 0.45 * np.random.default_rng(0).standard_normal(n) * np.exp(-180.0 * t)
    return np.clip(body + click, -1.0, 1.0).astype(np.float32)

def _make_snare(sr: int) -> np.ndarray:
    dur = 0.28
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n)
    tone  = 0.5 * np.sin(2 * np.pi * 200.0 * t) * np.exp(-28.0 * t)
    rng   = np.random.default_rng(1)
    noise = rng.standard_normal(n).astype(np.float64)
    noise = _hp(noise, 1800.0, sr) * np.exp(-14.0 * t)
    return np.clip((tone + 0.85 * noise), -1.0, 1.0).astype(np.float32)

def _make_hihat_closed(sr: int) -> np.ndarray:
    dur = 0.07
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n)
    rng = np.random.default_rng(2)
    noise = _hp(rng.standard_normal(n).astype(np.float64), 7500.0, sr)
    return np.clip(noise * np.exp(-75.0 * t) * 0.8, -1.0, 1.0).astype(np.float32)

def _make_hihat_open(sr: int) -> np.ndarray:
    dur = 0.38
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n)
    rng = np.random.default_rng(3)
    noise = _hp(rng.standard_normal(n).astype(np.float64), 6000.0, sr)
    return np.clip(noise * np.exp(-7.0 * t) * 0.65, -1.0, 1.0).astype(np.float32)

def _make_clap(sr: int) -> np.ndarray:
    dur = 0.22
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n)
    rng = np.random.default_rng(4)
    burst = np.zeros(n)
    for delay_s in [0.0, 0.006, 0.012, 0.018]:
        d = int(delay_s * sr)
        seg_t = t[d:]
        seg   = rng.standard_normal(len(seg_t)) * np.exp(-55.0 * seg_t)
        burst[d:] += seg
    burst = _bp(burst, 900.0, 9000.0, sr)
    return np.clip(burst, -1.0, 1.0).astype(np.float32)


# ── Patterns ─────────────────────────────────────────────────────────────────
# Each pattern is a dict of instrument → 16-step boolean list.
# Instruments: kick  snare  hh_c (closed)  hh_o (open)  clap

PATTERNS = [
    # 0 — Basic four-on-the-floor
    {
        "kick":  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hh_c":  [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hh_o":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    # 1 — Driving
    {
        "kick":  [1,0,0,1, 1,0,0,0, 1,0,0,1, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hh_c":  [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
        "hh_o":  [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    # 2 — Minimal
    {
        "kick":  [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hh_c":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hh_o":  [0,0,0,1, 0,0,1,0, 0,0,0,1, 0,0,1,0],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    # 3 — Syncopated
    {
        "kick":  [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,1,0],
        "snare": [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
        "hh_c":  [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        "hh_o":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    # 4 — Breakdown (no kick, hi-hats + clap only)
    {
        "kick":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hh_c":  [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
        "hh_o":  [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
        "clap":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    },
    # 5 — Off (silence)
    {
        "kick":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hh_c":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "hh_o":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        "clap":  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
]


# ── Drum machine ─────────────────────────────────────────────────────────────

class DrumMachine:
    def __init__(self, sample_rate: int = SAMPLE_RATE, bpm: int = BPM):
        self.sr           = sample_rate
        self.bpm          = bpm
        # Samples per 16th note
        self.step_samples = int(round(sample_rate * 60.0 / (bpm * 4)))

        self.lock         = threading.Lock()
        self.pattern_idx  = 0
        self.current_step = 0
        self.samples_in_step = 0
        # Active hits: list of [buffer_array, read_position]
        self.active_hits: list[list] = []

        # Pre-synthesise all drum sounds at init time
        self.sounds = {
            "kick":  _make_kick(sample_rate),
            "snare": _make_snare(sample_rate),
            "hh_c":  _make_hihat_closed(sample_rate),
            "hh_o":  _make_hihat_open(sample_rate),
            "clap":  _make_clap(sample_rate),
        }

        # Trigger step 0 immediately
        self._trigger_step(0)

    def set_pattern(self, idx: int):
        with self.lock:
            idx = max(0, min(idx, len(PATTERNS) - 1))
            if idx != self.pattern_idx:
                self.pattern_idx = idx
                # Reset to start of pattern on next step boundary (feels musical)

    def generate(self, n_frames: int) -> np.ndarray:
        out = np.zeros(n_frames, dtype=np.float32)

        with self.lock:
            frame = 0
            while frame < n_frames:
                until_next = self.step_samples - self.samples_in_step
                chunk      = min(until_next, n_frames - frame)

                # Mix active hits into this chunk
                survivors = []
                for hit in self.active_hits:
                    buf, pos = hit
                    end      = min(pos + chunk, len(buf))
                    copy_len = end - pos
                    if copy_len > 0:
                        out[frame:frame + copy_len] += buf[pos:end]
                    new_pos = pos + chunk
                    if new_pos < len(buf):
                        hit[1] = new_pos
                        survivors.append(hit)
                self.active_hits = survivors

                self.samples_in_step += chunk
                frame += chunk

                if self.samples_in_step >= self.step_samples:
                    self.samples_in_step -= self.step_samples
                    self.current_step = (self.current_step + 1) % 16
                    self._trigger_step(self.current_step)

        return out * DRUM_VOLUME

    # ── private ───────────────────────────────────────────────────────────────

    def _trigger_step(self, step: int):
        pattern = PATTERNS[self.pattern_idx]
        for inst, steps in pattern.items():
            if steps[step]:
                # Copy so concurrent playback of same sound works
                buf = self.sounds[inst].copy()
                self.active_hits.append([buf, 0])
