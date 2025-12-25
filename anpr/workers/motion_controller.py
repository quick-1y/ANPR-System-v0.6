from __future__ import annotations

from typing import Optional, Tuple

import cv2

from anpr.detection.motion_detector import MotionDetector, MotionDetectorConfig


class MotionController:
    """Управляет детекцией движения и состоянием ожидания."""

    def __init__(self, detection_mode: str, config: MotionDetectorConfig) -> None:
        self._detection_mode = detection_mode
        self._detector = MotionDetector(config)
        self._waiting_for_motion = False

    def update(self, roi_frame: cv2.Mat) -> Tuple[bool, Optional[str]]:
        """Обновляет состояние движения и возвращает флаг и статус для UI."""
        if self._detection_mode != "motion":
            return True, None

        motion_detected = self._detector.update(roi_frame)
        status_message: Optional[str] = None

        if not motion_detected:
            if not self._waiting_for_motion:
                status_message = "Ожидание движения"
            self._waiting_for_motion = True
        else:
            if self._waiting_for_motion:
                status_message = "Движение обнаружено"
            self._waiting_for_motion = False

        return motion_detected, status_message
