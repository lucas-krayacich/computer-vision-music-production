"""
CV Music Production Tool
─────────────────────────
Left hand  → root note wheel (scale-snapped)
Right hand → chord quality wheel
Either hand raised above wheels → drum pattern strip

Keyboard:
  S  — cycle through scales
  K  — cycle through keys
  Q  — quit
"""
# Must be set before sounddevice / PortAudio loads to avoid a macOS C++ mutex
# conflict between PortAudio threads and MediaPipe's gRPC threads.
import os
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import sys
import numpy as np
import cv2

from config import (
    SAMPLE_RATE, BUFFER_SIZE, BPM,
    KEYS, SCALE_NAMES, SCALES, QUALITY_NAMES,
    DEFAULT_KEY, DEFAULT_SCALE,
)
from vision.hand_tracker import HandTracker
from vision.gesture      import GestureClassifier
from vision.overlay      import Overlay
from music.theory        import note_name, get_chord_freqs
from music.pad_synth     import PadSynth
from music.drum_machine  import DrumMachine
import sounddevice as sd


def main():
    # ── State ─────────────────────────────────────────────────────────────────
    key_idx   = KEYS.index(DEFAULT_KEY)
    scale_idx = SCALE_NAMES.index(DEFAULT_SCALE)

    def cur_key():   return KEYS[key_idx]
    def cur_scale(): return SCALE_NAMES[scale_idx]

    # ── Vision first — MediaPipe must init before PortAudio on macOS ──────────
    tracker = HandTracker()
    gesture = GestureClassifier()
    overlay = Overlay()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ── Audio (after MediaPipe is fully initialised) ───────────────────────────
    pad   = PadSynth(SAMPLE_RATE)
    drums = DrumMachine(SAMPLE_RATE, BPM)

    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mix = pad.generate(frames) + drums.generate(frames)
        mix = np.clip(mix * 0.85, -1.0, 1.0)
        outdata[:, 0] = mix
        outdata[:, 1] = mix

    prev_chord_sig = None

    # ── Main loop ──────────────────────────────────────────────────────────────
    with sd.OutputStream(
        samplerate = SAMPLE_RATE,
        channels   = 2,
        dtype      = "float32",
        blocksize  = BUFFER_SIZE,
        callback   = audio_callback,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera not available.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            n_deg = len(SCALES[cur_scale()])
            state = gesture.update(hands, frame.shape, n_scale_degrees=n_deg)

            # ── Music logic ───────────────────────────────────────────────────
            key   = cur_key()
            scale = cur_scale()

            root_idx    = state.root_idx    % n_deg
            quality_idx = state.quality_idx % len(QUALITY_NAMES)
            quality     = QUALITY_NAMES[quality_idx]
            root_name   = note_name(root_idx, key, scale)
            chord_name  = f"{root_name}{quality}" if state.chord_active else "---"

            chord_sig = (root_idx, quality, key, scale, state.chord_active)
            if chord_sig != prev_chord_sig:
                if state.chord_active:
                    pad.set_chord(get_chord_freqs(root_idx, quality, key, scale))
                else:
                    pad.silence()
                prev_chord_sig = chord_sig

            drums.set_pattern(state.drum_pattern)

            # ── Render ────────────────────────────────────────────────────────
            scale_labels = [note_name(i, key, scale) for i in range(n_deg)]
            display = overlay.draw(
                frame, state, hands,
                scale_labels, chord_name, key, scale, BPM,
            )
            cv2.imshow("CV Music", display)

            # ── Keyboard ──────────────────────────────────────────────────────
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                scale_idx = (scale_idx + 1) % len(SCALE_NAMES)
                prev_chord_sig = None
            elif k == ord("k"):
                key_idx = (key_idx + 1) % len(KEYS)
                prev_chord_sig = None

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
