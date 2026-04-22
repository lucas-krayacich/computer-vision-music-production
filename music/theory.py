"""
Music theory helpers: scales, chord voicing, frequency conversion.
"""
import numpy as np
from config import KEYS, SCALES, CHORD_QUALITIES

BASE_OCTAVE = 4   # root notes start around octave 4 (MIDI 60 = C4)


def key_to_semitone(key: str) -> int:
    return KEYS.index(key)


def scale_degree_midis(key: str, scale_name: str) -> list[int]:
    """Return MIDI numbers for each scale degree, starting at BASE_OCTAVE."""
    root = key_to_semitone(key) + 12 * BASE_OCTAVE + 12   # C4 = 60
    return [root + i for i in SCALES[scale_name]]


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def note_name(degree_idx: int, key: str, scale_name: str) -> str:
    """Human-readable note name for a scale degree (e.g. 'F#')."""
    midis = scale_degree_midis(key, scale_name)
    n = len(midis)
    midi = midis[degree_idx % n]
    return KEYS[midi % 12]


def get_chord_freqs(degree_idx: int, quality: str, key: str, scale_name: str) -> list[float]:
    """
    Return a list of Hz values for a spread pad voicing.
    Layout: bass (root-12), root, 3rd, 5th, [7th if applicable]
    """
    midis = scale_degree_midis(key, scale_name)
    n = len(midis)
    root_midi = midis[degree_idx % n]

    intervals = CHORD_QUALITIES[quality]
    chord_notes = [root_midi + i for i in intervals]

    # Bass note one octave below root
    bass = root_midi - 12

    all_notes = [bass] + chord_notes
    return [midi_to_hz(m) for m in all_notes]
