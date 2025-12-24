#!/usr/bin/env python3
# /anpr/detection/motion_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class MotionDetectorConfig:
    threshold: float = 0.01
    frame_stride: int = 1
    activation_frames: int = 3
    release_frames: int = 6


class MotionDetector:
    """Простой детектор движения с учётом частоты обработки и устойчивостью к шуму."""

    def __init__(self, config: MotionDetectorConfig) -> None:
        self.config = config
        self._previous_frame: Optional[cv2.Mat] = None
        self._frame_index: int = 0
        self._motion_frames: int = 0
        self._static_frames: int = 0
        self._motion_active: bool = False

    def _should_analyze(self) -> bool:
        self._frame_index += 1
        stride = max(1, int(self.config.frame_stride))
        return self._frame_index % stride == 0

    def update(self, frame: cv2.Mat) -> bool:
        """Обновляет состояние детектора и возвращает, активно ли движение."""

        if frame.size == 0:
            return False

        if not self._should_analyze():
            return self._motion_active

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._previous_frame is None:
            self._previous_frame = gray
            return False

        frame_delta = cv2.absdiff(self._previous_frame, gray)
        self._previous_frame = gray

        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = cv2.countNonZero(thresh) / float(gray.size)

        if motion_ratio > self.config.threshold:
            self._motion_frames += 1
            self._static_frames = 0
        else:
            self._static_frames += 1
            self._motion_frames = 0

        if not self._motion_active and self._motion_frames >= self.config.activation_frames:
            self._motion_active = True

        if self._motion_active and self._static_frames >= self.config.release_frames:
            self._motion_active = False

        return self._motion_active
