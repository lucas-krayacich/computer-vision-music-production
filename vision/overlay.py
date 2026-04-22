"""
Liquid-glass UI overlay.

Visual recipe per element
─────────────────────────
1. Frosted backdrop  — Gaussian-blurred video + 12 % lavender tint, masked to shape
2. Active fill       — semi-transparent violet (alpha blend, 38 %)
3. Active glow       — blurred violet layer, additive blend (brightens underneath)
4. Segment dividers  — thin white lines, 45 % opacity
5. Labels            — white (active) / soft lavender (inactive), 1-px shadow
6. Rim lights        — outer + inner circle, additive white, 60 %
7. Specular crescent — soft white ellipse near top, additive, 30 %

Performance: one global blur per frame; all per-element ops work on a tight
bounding-box ROI so cost scales with wheel size, not frame size.
"""
import cv2
import numpy as np
from config import (
    LEFT_WHEEL_X,  LEFT_WHEEL_Y,
    RIGHT_WHEEL_X, RIGHT_WHEEL_Y,
    WHEEL_RADIUS_RATIO, DEAD_ZONE_RATIO,
    DRUM_STRIP_Y_RATIO,
    QUALITY_NAMES, DRUM_PATTERN_NAMES,
)
from vision.gesture import AppState

# ── Palette ───────────────────────────────────────────────────────────────────
_TINT_RGB   = np.array([255, 215, 235], np.float32)   # lavender  (RGB for float ops)
_ACTIVE_RGB = np.array([230,  80, 255], np.float32)   # violet
_GLOW_RGB   = np.array([190,  40, 210], np.float32)   # deeper glow
_DRUM_OFF_RGB = np.array([60, 60, 80], np.float32)    # dim blue-grey for "Off"

# MediaPipe hand skeleton connections
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _pie_pts(cx, cy, r_outer, r_inner, a0_deg, a1_deg, steps=32):
    angles = np.linspace(np.radians(a0_deg), np.radians(a1_deg), steps)
    outer = [(cx + r_outer * np.cos(a), cy + r_outer * np.sin(a)) for a in angles]
    inner = [(cx + r_inner * np.cos(a), cy + r_inner * np.sin(a)) for a in reversed(angles)]
    return np.array(outer + inner, np.int32)


def _roi_bounds(canvas_shape, cx, cy, r):
    h, w = canvas_shape[:2]
    return (max(0, cy - r), min(h, cy + r + 1),
            max(0, cx - r), min(w, cx + r + 1))


def _additive(canvas, layer_f32, strength=1.0):
    """Additive blend: canvas += layer * strength  (clamped to uint8)."""
    np.clip(canvas.astype(np.float32) + layer_f32 * strength,
            0, 255, out=layer_f32)                   # reuse buffer
    canvas[:] = layer_f32.astype(np.uint8)


def _alpha_blend(canvas, color_rgb, mask_f32, alpha):
    """Blend a flat colour onto canvas inside mask at given alpha."""
    c = canvas.astype(np.float32)
    # colour in BGR
    col_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], np.float32)
    m = mask_f32[:, :, np.newaxis]
    canvas[:] = np.clip(c * (1 - m * alpha) + col_bgr * m * alpha, 0, 255).astype(np.uint8)


# ── Glass backdrop ────────────────────────────────────────────────────────────

def _glass_circle(canvas, blurred, cx, cy, radius, tint_str=0.13):
    """Composite frosted-glass circle onto canvas."""
    y0, y1, x0, x1 = _roi_bounds(canvas.shape, cx, cy, radius)
    roi   = canvas[y0:y1, x0:x1].astype(np.float32)
    blur  = blurred[y0:y1, x0:x1].astype(np.float32)

    # Build float mask for the circle, local coordinates
    rh, rw = roi.shape[:2]
    ys = np.arange(rh, dtype=np.float32) + (y0 - cy)
    xs = np.arange(rw, dtype=np.float32) + (x0 - cx)
    xx, yy = np.meshgrid(xs, ys)
    mask = np.clip(radius - np.sqrt(xx**2 + yy**2), 0, 1)[:, :, np.newaxis]

    tint_bgr = np.array([_TINT_RGB[2], _TINT_RGB[1], _TINT_RGB[0]], np.float32)
    glass = blur * (1 - tint_str) + tint_bgr * tint_str

    canvas[y0:y1, x0:x1] = np.clip(roi * (1 - mask) + glass * mask, 0, 255).astype(np.uint8)


def _glass_rect(canvas, blurred, x0, y0, x1, y1, tint_str=0.13):
    """Composite frosted-glass rectangle onto canvas."""
    roi  = canvas[y0:y1, x0:x1].astype(np.float32)
    blur = blurred[y0:y1, x0:x1].astype(np.float32)
    tint_bgr = np.array([_TINT_RGB[2], _TINT_RGB[1], _TINT_RGB[0]], np.float32)
    glass = blur * (1 - tint_str) + tint_bgr * tint_str
    canvas[y0:y1, x0:x1] = np.clip(roi * 0.0 + glass, 0, 255).astype(np.uint8)


# ── Wheel ─────────────────────────────────────────────────────────────────────

def _draw_glass_wheel(canvas, blurred, cx, cy, radius, dead_zone,
                       n_seg, active_seg, labels, centre_label):

    # 1. Frosted glass backdrop
    _glass_circle(canvas, blurred, cx, cy, radius,      tint_str=0.13)
    _glass_circle(canvas, blurred, cx, cy, dead_zone,   tint_str=0.22)

    # 2. Active segment — alpha fill
    a0 = active_seg * 360.0 / n_seg - 90
    a1 = (active_seg + 1) * 360.0 / n_seg - 90
    pts = _pie_pts(cx, cy, radius - 2, dead_zone + 2, a0, a1, steps=40)

    seg_mask = np.zeros(canvas.shape[:2], np.float32)
    cv2.fillPoly(seg_mask, [pts], 1.0)
    _alpha_blend(canvas, _ACTIVE_RGB, seg_mask, alpha=0.38)

    # 3. Active segment — additive glow (blurred halo)
    y0, y1, x0, x1 = _roi_bounds(canvas.shape, cx, cy, radius + 20)
    glow_roi = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    seg_mask_roi = seg_mask[y0:y1, x0:x1]
    col_bgr = np.array([_GLOW_RGB[2], _GLOW_RGB[1], _GLOW_RGB[0]], np.float32)
    glow_roi += col_bgr * seg_mask_roi[:, :, np.newaxis]
    glow_roi = cv2.GaussianBlur(glow_roi, (29, 29), 0)
    tmp = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(tmp + glow_roi * 0.55, 0, 255).astype(np.uint8)

    # 4. Segment dividers
    div = np.zeros_like(canvas, np.float32)
    for i in range(n_seg):
        angle = i * 2 * np.pi / n_seg - np.pi / 2
        px1 = int(cx + (dead_zone + 2) * np.cos(angle))
        py1 = int(cy + (dead_zone + 2) * np.sin(angle))
        px2 = int(cx + (radius - 2)    * np.cos(angle))
        py2 = int(cy + (radius - 2)    * np.sin(angle))
        cv2.line(div, (px1, py1), (px2, py2), (200, 185, 220), 1, cv2.LINE_AA)
    canvas[:] = np.clip(
        canvas.astype(np.float32) * 0.55 + div * 0.45 + canvas.astype(np.float32) * 0.45,
        0, 255
    ).astype(np.uint8)

    # 5. Labels
    for i, label in enumerate(labels):
        mid  = np.radians((i + 0.5) * 360.0 / n_seg - 90.0)
        lr   = (radius + dead_zone) / 2
        lx   = int(cx + lr * np.cos(mid))
        ly   = int(cy + lr * np.sin(mid))
        fs   = 0.46
        col  = (255, 255, 255) if i == active_seg else (195, 182, 215)
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        ox, oy = lx - tw // 2, ly + th // 2
        cv2.putText(canvas, label, (ox + 1, oy + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 10, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (ox, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

    # Centre label
    fs = 0.38
    tw, th = cv2.getTextSize(centre_label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
    cv2.putText(canvas, centre_label, (cx - tw // 2 + 1, cy + th // 2 + 1),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 10, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, centre_label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (210, 200, 230), 1, cv2.LINE_AA)

    # 6. Rim lights — additive white
    rim = np.zeros(canvas.shape[:2], np.float32)
    cv2.circle(rim, (cx, cy), radius,    1.0, 2, cv2.LINE_AA)
    cv2.circle(rim, (cx, cy), dead_zone, 1.0, 1, cv2.LINE_AA)
    rim3 = np.stack([rim, rim, rim], axis=2) * 255
    tmp  = canvas.astype(np.float32)
    canvas[:] = np.clip(tmp + rim3 * 0.60, 0, 255).astype(np.uint8)

    # 7. Specular crescent (top)
    spec = np.zeros(canvas.shape[:2], np.float32)
    sy   = cy - int(radius * 0.44)
    srx  = int(radius * 0.46)
    sry  = int(radius * 0.13)
    cv2.ellipse(spec, (cx, sy), (srx, sry), 0, 0, 360, 1.0, -1)
    spec = cv2.GaussianBlur(spec, (33, 33), 0)
    spec3 = np.stack([spec, spec, spec], axis=2) * 255
    canvas[:] = np.clip(canvas.astype(np.float32) + spec3 * 0.30, 0, 255).astype(np.uint8)


# ── Drum strip ────────────────────────────────────────────────────────────────

def _draw_glass_strip(canvas, blurred, state: AppState, w, strip_h):
    n      = len(DRUM_PATTERN_NAMES)
    zone_w = w // n
    off_i  = n - 1   # "Off" is always last

    # Frosted glass backdrop for entire strip
    _glass_rect(canvas, blurred, 0, 0, w, strip_h, tint_str=0.16)

    # Active zone fill + glow
    ai  = state.drum_pattern
    ax0 = ai * zone_w
    ax1 = (ai + 1) * zone_w

    col = _DRUM_OFF_RGB if ai == off_i else _ACTIVE_RGB

    fill_mask = np.zeros(canvas.shape[:2], np.float32)
    cv2.rectangle(fill_mask, (ax0, 0), (ax1, strip_h), 1.0, -1)
    _alpha_blend(canvas, col, fill_mask, alpha=0.42)

    # Glow
    glow_roi = np.zeros((strip_h, ax1 - ax0, 3), np.float32)
    col_bgr  = np.array([col[2], col[1], col[0]], np.float32)
    glow_roi += col_bgr
    glow_roi  = cv2.GaussianBlur(glow_roi, (31, 31), 0)
    tmp = canvas[0:strip_h, ax0:ax1].astype(np.float32)
    canvas[0:strip_h, ax0:ax1] = np.clip(tmp + glow_roi * 0.45, 0, 255).astype(np.uint8)

    # Hover highlight (hand in strip but not confirmed yet)
    if state.in_drum_strip:
        hi = max(0, min(n - 1, int(state.drum_strip_xfrac * n)))
        if hi != ai:
            hx0, hx1 = hi * zone_w, (hi + 1) * zone_w
            hover_mask = np.zeros(canvas.shape[:2], np.float32)
            cv2.rectangle(hover_mask, (hx0, 0), (hx1, strip_h), 1.0, -1)
            _alpha_blend(canvas, np.array([200, 185, 215], np.float32), hover_mask, 0.20)

    # Zone dividers
    for i in range(1, n):
        xd = i * zone_w
        cv2.line(canvas, (xd, 0), (xd, strip_h), (220, 210, 240), 1, cv2.LINE_AA)

    # Labels
    for i, name in enumerate(DRUM_PATTERN_NAMES):
        cx_t  = i * zone_w + zone_w // 2
        cy_t  = strip_h // 2
        col_t = (255, 255, 255) if i == state.drum_pattern else (165, 155, 185)
        fs    = 0.44
        tw, th = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        ox, oy = cx_t - tw // 2, cy_t + th // 2
        cv2.putText(canvas, name, (ox + 1, oy + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 10, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (ox, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col_t,         1, cv2.LINE_AA)

    # Top + bottom rim lights
    rim_top = np.zeros(canvas.shape[:2], np.float32)
    cv2.line(rim_top, (0, 0),        (w, 0),        1.0, 2)
    cv2.line(rim_top, (0, strip_h),  (w, strip_h),  1.0, 1)
    rim3 = np.stack([rim_top, rim_top, rim_top], axis=2) * 255
    canvas[:] = np.clip(canvas.astype(np.float32) + rim3 * 0.65, 0, 255).astype(np.uint8)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def _draw_skeleton(canvas, landmarks, w, h):
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
    # Glow lines — draw thick dim line first, then thin bright line
    for a, b in _CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], (160, 240, 200), 3, cv2.LINE_AA)
        cv2.line(canvas, pts[a], pts[b], (220, 255, 230), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 4, (180, 255, 210), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 4, (255, 255, 255),  1, cv2.LINE_AA)


# ── Status bar ────────────────────────────────────────────────────────────────

def _draw_statusbar(canvas, blurred, w, h, chord_name, key, scale_name, bpm,
                    lpf_enabled=False, lpf_hz=800):
    bar_h = 48
    y0    = h - bar_h
    _glass_rect(canvas, blurred, 0, y0, w, h, tint_str=0.18)

    # Top rim
    rim = np.zeros(canvas.shape[:2], np.float32)
    cv2.line(rim, (0, y0), (w, y0), 1.0, 1)
    rim3 = np.stack([rim, rim, rim], axis=2) * 255
    canvas[:] = np.clip(canvas.astype(np.float32) + rim3 * 0.55, 0, 255).astype(np.uint8)

    lpf_str = f"LPF {lpf_hz}Hz" if lpf_enabled else "LPF off"
    line1   = (f"Chord: {chord_name}    Key: {key}    "
               f"Scale: {scale_name}    BPM: {bpm}    {lpf_str}")
    line2   = "S: scale    K: key    F: LPF on/off    [ / ]: LPF cutoff    Q: quit"

    for txt, y, fs, col in [
        (line1, y0 + 16, 0.46, (245, 240, 255)),
        (line2, y0 + 34, 0.35, (160, 150, 185)),
    ]:
        cv2.putText(canvas, txt, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 10, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col,          1, cv2.LINE_AA)


# ── Public class ──────────────────────────────────────────────────────────────

class Overlay:
    def draw(
        self,
        frame:        np.ndarray,
        state:        AppState,
        hands:        list,
        scale_labels: list[str],
        chord_name:   str,
        key:          str,
        scale_name:   str,
        bpm:          int,
        lpf_enabled:  bool = False,
        lpf_hz:       int  = 800,
    ) -> np.ndarray:

        canvas  = frame.copy()
        h, w    = canvas.shape[:2]
        blurred = cv2.GaussianBlur(frame, (45, 45), 0)   # shared frosted-glass source

        cx_l   = int(LEFT_WHEEL_X  * w);  cy_l = int(LEFT_WHEEL_Y  * h)
        cx_r   = int(RIGHT_WHEEL_X * w);  cy_r = int(RIGHT_WHEEL_Y * h)
        radius    = int(WHEEL_RADIUS_RATIO * min(w, h))
        dead_zone = int(DEAD_ZONE_RATIO    * radius)
        strip_h   = int(DRUM_STRIP_Y_RATIO * h)

        # Drum strip
        _draw_glass_strip(canvas, blurred, state, w, strip_h)

        # Root note wheel (left)
        _draw_glass_wheel(
            canvas, blurred, cx_l, cy_l, radius, dead_zone,
            n_seg      = len(scale_labels),
            active_seg = state.root_idx,
            labels     = scale_labels,
            centre_label = "ROOT",
        )

        # Chord quality wheel (right)
        _draw_glass_wheel(
            canvas, blurred, cx_r, cy_r, radius, dead_zone,
            n_seg      = len(QUALITY_NAMES),
            active_seg = state.quality_idx,
            labels     = QUALITY_NAMES,
            centre_label = "QUALITY",
        )

        # Hand skeletons
        for hand in hands:
            _draw_skeleton(canvas, hand.landmarks, w, h)

        # Fingertip cursors
        for tip in (state.left_tip, state.right_tip):
            if tip is not None:
                glow = np.zeros_like(canvas, np.float32)
                cv2.circle(glow, tip, 18, (200, 100, 255), -1)
                glow = cv2.GaussianBlur(glow.astype(np.uint8), (25, 25), 0).astype(np.float32)
                canvas[:] = np.clip(canvas.astype(np.float32) + glow * 0.5, 0, 255).astype(np.uint8)
                cv2.circle(canvas, tip, 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, tip,  4, (230,  90, 255), -1, cv2.LINE_AA)

        # Status bar
        _draw_statusbar(canvas, blurred, w, h, chord_name, key, scale_name, bpm,
                        lpf_enabled, lpf_hz)

        return canvas
