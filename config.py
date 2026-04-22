# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 44100
BUFFER_SIZE  = 512          # frames per callback (lower = less latency)
BPM          = 128

# ── Pad synth ─────────────────────────────────────────────────────────────────
PAD_ATTACK   = 0.8          # seconds
PAD_RELEASE  = 0.6          # seconds
PAD_VOLUME   = 0.40         # 0-1

# ── Drums ─────────────────────────────────────────────────────────────────────
DRUM_VOLUME  = 0.55         # 0-1

# ── Music theory ──────────────────────────────────────────────────────────────
DEFAULT_KEY   = "C"
DEFAULT_SCALE = "major"

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":  [0, 2, 4, 7, 9],
}
SCALE_NAMES = list(SCALES.keys())

# Chord quality name → intervals from root (semitones)
CHORD_QUALITIES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "7":    [0, 4, 7, 10],
    "m7":   [0, 3, 7, 10],
    "sus4": [0, 5, 7],
}
QUALITY_NAMES = list(CHORD_QUALITIES.keys())

# ── Wheel geometry (0.0–1.0 relative to frame) ────────────────────────────────
LEFT_WHEEL_X      = 0.27     # root note wheel horizontal centre
LEFT_WHEEL_Y      = 0.58
RIGHT_WHEEL_X     = 0.73     # chord quality wheel horizontal centre
RIGHT_WHEEL_Y     = 0.58
WHEEL_RADIUS_RATIO = 0.185   # fraction of min(frame_w, frame_h)
DEAD_ZONE_RATIO    = 0.32    # fraction of wheel radius → centre dead zone

# ── Drum strip ────────────────────────────────────────────────────────────────
DRUM_STRIP_Y_RATIO  = 0.17   # hand y < this → drum selection zone
DRUM_PATTERN_NAMES  = ["Basic", "Driving", "Minimal", "Syncopated", "Breakdown"]
DRUM_DWELL_MS       = 450    # ms hand must hold zone before confirming
CHORD_DWELL_MS      = 280    # ms hand must hold wheel segment before confirming

# ── UI colours (BGR) ──────────────────────────────────────────────────────────
COLOR_ACTIVE      = (210,  90, 255)
COLOR_INACTIVE    = ( 55,  50,  75)
COLOR_TEXT        = (240, 240, 240)
COLOR_HAND        = ( 90, 220, 140)
COLOR_STRIP_BG    = ( 35,  32,  52)
COLOR_DRUM_ACTIVE = ( 90, 200, 255)
ALPHA_WHEEL       = 0.60
