"""
Gesture classifier.

Tracking points:
  - Chord wheels  → index fingertip (landmark 8)  — point at a segment to select
  - Drum strip    → index fingertip, dwell 450 ms  — hold to confirm pattern

Selection model:
  - Chord wheels update instantly every frame (no dwell) so the chord responds
    immediately as you point around the wheel.
  - A tiny 3-frame segment buffer smooths out single-frame jitter.
  - Drum pattern keeps a dwell so you don't accidentally change it.
"""
import time
from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import (
    LEFT_WHEEL_X,  LEFT_WHEEL_Y,
    RIGHT_WHEEL_X, RIGHT_WHEEL_Y,
    WHEEL_RADIUS_RATIO, DEAD_ZONE_RATIO,
    DRUM_STRIP_Y_RATIO, DRUM_DWELL_MS,
    QUALITY_NAMES,
)

# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    root_idx:     int  = 0
    quality_idx:  int  = 0
    chord_active: bool = False

    # pending drum (dwell-based)
    drum_pattern:     int            = 0
    drum_pending:     Optional[int]  = None
    drum_dwell_start: Optional[float]= None

    # rendering hints
    left_tip:        Optional[tuple[int, int]] = None   # index fingertip px
    right_tip:       Optional[tuple[int, int]] = None
    in_drum_strip:   bool  = False
    drum_strip_xfrac: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _index_tip(landmarks) -> tuple[float, float]:
    """Normalised (x, y) of index fingertip — landmark 8."""
    lm = landmarks[8]
    return float(lm[0]), float(lm[1])


def _angle_segment(dx: float, dy: float, n: int) -> int:
    """Angle from wheel centre → 0-based segment index (clockwise from top)."""
    angle = np.arctan2(dx, -dy)
    if angle < 0:
        angle += 2 * np.pi
    return int(angle / (2 * np.pi / n)) % n


def _mode(buf: deque) -> Optional[int]:
    """Most common value in buffer, or None if empty."""
    if not buf:
        return None
    return max(set(buf), key=buf.count)


# ── Classifier ────────────────────────────────────────────────────────────────

class GestureClassifier:
    N_QUALITY   = len(QUALITY_NAMES)
    SMOOTH_FRAMES = 3   # frames to smooth chord segment flicker

    def __init__(self):
        self._state       = AppState()
        self._root_buf    = deque(maxlen=self.SMOOTH_FRAMES)
        self._quality_buf = deque(maxlen=self.SMOOTH_FRAMES)

    def update(
        self,
        hands,
        frame_shape: tuple,
        n_scale_degrees: int = 7,
    ) -> AppState:

        h, w = frame_shape[:2]
        now  = time.monotonic() * 1000.0

        cx_l   = int(LEFT_WHEEL_X  * w);  cy_l = int(LEFT_WHEEL_Y  * h)
        cx_r   = int(RIGHT_WHEEL_X * w);  cy_r = int(RIGHT_WHEEL_Y * h)
        radius    = int(WHEEL_RADIUS_RATIO * min(w, h))
        dead_zone = int(DEAD_ZONE_RATIO    * radius)
        strip_y   = int(DRUM_STRIP_Y_RATIO * h)

        s = self._state
        s.left_tip      = None
        s.right_tip     = None
        s.in_drum_strip = False

        left_hand  = next((hd for hd in hands if hd.handedness == "Left"),  None)
        right_hand = next((hd for hd in hands if hd.handedness == "Right"), None)

        # ── Left hand → root wheel ────────────────────────────────────────────
        if left_hand:
            fx, fy   = _index_tip(left_hand.landmarks)
            fpx, fpy = int(fx * w), int(fy * h)
            s.left_tip = (fpx, fpy)

            if fpy < strip_y:
                # Drum strip — left hand can also select drums
                s.in_drum_strip    = True
                s.drum_strip_xfrac = float(np.clip(fx, 0.0, 1.0))
                seg = max(0, min(4, int(fx * 5)))
                self._drum_dwell(seg, s, now)

            else:
                dx   = fpx - cx_l
                dy   = fpy - cy_l
                dist = np.hypot(dx, dy)

                if dist <= dead_zone:
                    # Centre dead zone → silence
                    s.chord_active = False
                    self._root_buf.clear()

                elif dist <= radius:
                    # Inside wheel ring → instant selection
                    seg = _angle_segment(dx, dy, n_scale_degrees)
                    self._root_buf.append(seg)
                    smoothed = _mode(self._root_buf)
                    if smoothed is not None:
                        s.root_idx    = smoothed
                        s.chord_active = True
                # outside radius entirely → no change, keep last chord

        else:
            self._root_buf.clear()

        # ── Right hand → quality wheel ────────────────────────────────────────
        if right_hand:
            fx, fy   = _index_tip(right_hand.landmarks)
            fpx, fpy = int(fx * w), int(fy * h)
            s.right_tip = (fpx, fpy)

            if fpy < strip_y:
                s.in_drum_strip    = True
                s.drum_strip_xfrac = float(np.clip(fx, 0.0, 1.0))
                seg = max(0, min(4, int(fx * 5)))
                self._drum_dwell(seg, s, now)

            else:
                dx   = fpx - cx_r
                dy   = fpy - cy_r
                dist = np.hypot(dx, dy)

                if dead_zone < dist <= radius:
                    seg = _angle_segment(dx, dy, self.N_QUALITY)
                    self._quality_buf.append(seg)
                    smoothed = _mode(self._quality_buf)
                    if smoothed is not None:
                        s.quality_idx = smoothed
                else:
                    self._quality_buf.clear()

        else:
            self._quality_buf.clear()

        return s

    # ── Drum dwell ────────────────────────────────────────────────────────────

    @staticmethod
    def _drum_dwell(seg: int, s: AppState, now: float):
        if seg != s.drum_pending:
            s.drum_pending     = seg
            s.drum_dwell_start = now
        elif s.drum_dwell_start is not None and (now - s.drum_dwell_start) >= DRUM_DWELL_MS:
            s.drum_pattern     = seg
            s.drum_dwell_start = now
