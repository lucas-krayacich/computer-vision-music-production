"""
OpenCV overlay renderer.

Draws:
  • Drum pattern strip at the top
  • Two segmented wheels (root note / chord quality)
  • Hand skeleton on top
  • Status bar at the bottom
"""
import cv2
import numpy as np
from config import (
    LEFT_WHEEL_X,  LEFT_WHEEL_Y,
    RIGHT_WHEEL_X, RIGHT_WHEEL_Y,
    WHEEL_RADIUS_RATIO, DEAD_ZONE_RATIO,
    DRUM_STRIP_Y_RATIO,
    COLOR_ACTIVE, COLOR_INACTIVE, COLOR_TEXT,
    COLOR_HAND, COLOR_STRIP_BG, COLOR_DRUM_ACTIVE,
    ALPHA_WHEEL,
    QUALITY_NAMES, DRUM_PATTERN_NAMES,
)
from vision.gesture import AppState

# MediaPipe hand skeleton connections (landmark index pairs)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── Private drawing helpers ───────────────────────────────────────────────────

def _pie_pts(cx, cy, r_outer, r_inner, a_start_deg, a_end_deg, steps=24):
    """Return integer point array for an annular wedge (pie slice)."""
    angles = np.linspace(np.radians(a_start_deg), np.radians(a_end_deg), steps)
    outer  = [(cx + r_outer * np.cos(a), cy + r_outer * np.sin(a)) for a in angles]
    inner  = [(cx + r_inner * np.cos(a), cy + r_inner * np.sin(a)) for a in reversed(angles)]
    return np.array(outer + inner, dtype=np.int32)


def _draw_wheel(canvas, cx, cy, radius, dead_zone,
                n_seg, active_seg, pending_seg, labels,
                color_on, color_off, alpha):
    overlay = canvas.copy()
    for i in range(n_seg):
        a0 = i       * 360.0 / n_seg - 90.0
        a1 = (i + 1) * 360.0 / n_seg - 90.0
        if i == active_seg:
            col = color_on
        elif i == pending_seg:
            # Show pending as slightly dimmer version of active
            col = tuple(int(c * 0.6) for c in color_on)
        else:
            col = color_off
        pts = _pie_pts(cx, cy, radius, dead_zone, a0, a1)
        cv2.fillPoly(overlay, [pts], col)
        cv2.polylines(canvas, [pts], True, (90, 85, 110), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.circle(canvas, (cx, cy), radius,    (160, 150, 190), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), dead_zone, ( 80,  75, 100), 2, cv2.LINE_AA)

    # Segment labels
    for i, label in enumerate(labels):
        mid = np.radians((i + 0.5) * 360.0 / n_seg - 90.0)
        r   = (radius + dead_zone) / 2
        lx  = int(cx + r * np.cos(mid))
        ly  = int(cy + r * np.sin(mid))
        col = (255, 255, 255) if i == active_seg else (170, 160, 195)
        fs  = 0.46
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        cv2.putText(canvas, label, (lx - tw // 2, ly + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)


def _draw_drum_strip(canvas, state: AppState, w, strip_h):
    n      = len(DRUM_PATTERN_NAMES)
    zone_w = w // n
    overlay = canvas.copy()

    for i, name in enumerate(DRUM_PATTERN_NAMES):
        x0, x1 = i * zone_w, (i + 1) * zone_w
        if i == state.drum_pattern:
            col = COLOR_DRUM_ACTIVE
        elif state.in_drum_strip and int(state.drum_strip_xfrac * 5) == i:
            col = (70, 65, 100)
        else:
            col = COLOR_STRIP_BG
        cv2.rectangle(overlay, (x0, 0), (x1, strip_h), col, -1)

    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    # Labels + dividers
    for i, name in enumerate(DRUM_PATTERN_NAMES):
        cx = i * zone_w + zone_w // 2
        cy = strip_h // 2
        col = (255, 255, 255) if i == state.drum_pattern else (155, 150, 175)
        tw, th = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.putText(canvas, name, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        if i:
            cv2.line(canvas, (i * zone_w, 0), (i * zone_w, strip_h), (90, 85, 115), 1)

    cv2.rectangle(canvas, (0, 0), (w, strip_h), (90, 85, 115), 1)


# ── Public Overlay class ──────────────────────────────────────────────────────

class Overlay:
    def draw(
        self,
        frame:       np.ndarray,
        state:       AppState,
        hands:       list,             # list[HandData] for skeleton
        scale_labels: list[str],       # note names for left wheel segments
        chord_name:  str,
        key:         str,
        scale_name:  str,
        bpm:         int,
    ) -> np.ndarray:

        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        cx_l   = int(LEFT_WHEEL_X  * w);  cy_l = int(LEFT_WHEEL_Y  * h)
        cx_r   = int(RIGHT_WHEEL_X * w);  cy_r = int(RIGHT_WHEEL_Y * h)
        radius    = int(WHEEL_RADIUS_RATIO * min(w, h))
        dead_zone = int(DEAD_ZONE_RATIO    * radius)
        strip_h   = int(DRUM_STRIP_Y_RATIO * h)

        # 1 — Drum strip
        _draw_drum_strip(canvas, state, w, strip_h)

        # 2 — Root note wheel (left)
        _draw_wheel(
            canvas, cx_l, cy_l, radius, dead_zone,
            n_seg      = len(scale_labels),
            active_seg = state.root_idx,
            pending_seg= -1,
            labels     = scale_labels,
            color_on   = COLOR_ACTIVE,
            color_off  = COLOR_INACTIVE,
            alpha      = ALPHA_WHEEL,
        )
        _centre_text(canvas, "ROOT", cx_l, cy_l)

        # 3 — Chord quality wheel (right)
        _draw_wheel(
            canvas, cx_r, cy_r, radius, dead_zone,
            n_seg      = len(QUALITY_NAMES),
            active_seg = state.quality_idx,
            pending_seg= -1,
            labels     = QUALITY_NAMES,
            color_on   = COLOR_ACTIVE,
            color_off  = COLOR_INACTIVE,
            alpha      = ALPHA_WHEEL,
        )
        _centre_text(canvas, "QUALITY", cx_r, cy_r)

        # 4 — Hand skeletons (drawn last so they're on top)
        for hand in hands:
            _draw_skeleton(canvas, hand.landmarks, w, h)

        # 5 — Fingertip cursor dots (large, easy to see)
        for tip in (state.left_tip, state.right_tip):
            if tip is not None:
                cv2.circle(canvas, tip, 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, tip,  5, (220, 80, 255),  -1, cv2.LINE_AA)

        # 6 — Status bar
        _draw_statusbar(canvas, w, h, chord_name, key, scale_name, bpm)

        return canvas


# ── Small helpers ─────────────────────────────────────────────────────────────

def _centre_text(canvas, text, cx, cy):
    fs = 0.4
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
    cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 185, 220), 1, cv2.LINE_AA)


def _draw_skeleton(canvas, landmarks, w, h):
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], COLOR_HAND, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 3, (200, 255, 180), -1, cv2.LINE_AA)


def _draw_statusbar(canvas, w, h, chord_name, key, scale_name, bpm):
    bar_y = h - 42
    cv2.rectangle(canvas, (0, bar_y - 6), (w, h), (22, 20, 32), -1)
    line1 = f"Chord: {chord_name}    Key: {key}    Scale: {scale_name}    BPM: {bpm}"
    line2 = "S: cycle scale    K: cycle key    Q: quit"
    cv2.putText(canvas, line1, (10, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT,       1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, bar_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 125, 155),  1, cv2.LINE_AA)
