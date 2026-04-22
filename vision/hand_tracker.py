"""
Thin MediaPipe wrapper.
Returns HandData objects with normalised landmark coordinates (x, y, z ∈ 0–1).
Handedness is corrected for the mirrored (selfie) view.
"""
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

_mp_hands   = mp.solutions.hands


@dataclass
class HandData:
    landmarks:  list[tuple[float, float, float]]   # 21 points, normalised
    handedness: str                                 # "Left" or "Right" (mirror-corrected)


class HandTracker:
    def __init__(self, max_hands: int = 2):
        self._hands = _mp_hands.Hands(
            model_complexity=0,
            max_num_hands=max_hands,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )

    def process(self, frame_bgr: np.ndarray) -> list[HandData]:
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hand_list: list[HandData] = []
        if not results.multi_hand_landmarks:
            return hand_list

        for lm_set, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            mirror_label = handedness.classification[0].label   # use as-is; frame is flipped

            landmarks = [
                (lm_set.landmark[i].x, lm_set.landmark[i].y, lm_set.landmark[i].z)
                for i in range(21)
            ]
            hand_list.append(HandData(landmarks, mirror_label))

        return hand_list

    def close(self):
        self._hands.close()
