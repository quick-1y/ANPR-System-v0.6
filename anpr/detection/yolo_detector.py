# /anpr/detection/yolo_detector.py
"""Обертка для детектора номерных знаков YOLO."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from anpr.config import Config
from anpr.infrastructure.logging_manager import get_logger

logger = get_logger(__name__)


class YOLODetector:
    """Детектор с безопасным откатом к обычной детекции при ошибках трекера."""

    def __init__(
        self,
        model_path: str,
        device,
        min_plate_size: Optional[Dict[str, int]] = None,
        max_plate_size: Optional[Dict[str, int]] = None,
        size_filter_enabled: bool = True,
    ) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self._min_plate_size = min_plate_size or {}
        self._max_plate_size = max_plate_size or {}
        self._size_filter_enabled = bool(size_filter_enabled)
        self._tracking_supported = True
        self._last_frame_shape: Optional[tuple[int, ...]] = None
        logger.info("Детектор YOLO успешно загружен (model=%s, device=%s)", model_path, device)

    def _reset_tracker_state(self) -> None:
        """Сбрасывает состояние трекера YOLO при смене входного разрешения."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if not trackers:
            return

        for tracker in trackers:
            try:
                if hasattr(tracker, "reset"):
                    tracker.reset()
            except Exception:
                logger.debug("Не удалось сбросить состояние трекера YOLO", exc_info=True)

        if predictor and hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(trackers)

    def _maybe_reset_tracker(self, frame_shape: tuple[int, ...]) -> None:
        if self._last_frame_shape and self._last_frame_shape != frame_shape:
            logger.debug(
                "Сбрасываем состояние YOLO-трекера из-за смены размера кадра: %s -> %s",
                self._last_frame_shape,
                frame_shape,
            )
            self._reset_tracker_state()
        self._last_frame_shape = frame_shape

    def _filter_by_size(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        if not self._size_filter_enabled:
            return detections

        min_width = int(self._min_plate_size.get("width", 0) or 0)
        min_height = int(self._min_plate_size.get("height", 0) or 0)
        max_width = int(self._max_plate_size.get("width", 0) or 0)
        max_height = int(self._max_plate_size.get("height", 0) or 0)

        filtered: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            width = max(0, int(bbox[2]) - int(bbox[0]))
            height = max(0, int(bbox[3]) - int(bbox[1]))

            if min_width and width < min_width:
                continue
            if min_height and height < min_height:
                continue
            if max_width and width > max_width:
                continue
            if max_height and height > max_height:
                continue

            filtered.append(det)

        return filtered

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return []

        self._maybe_reset_tracker(frame.shape)
        detections = self.model.predict(frame, verbose=False, device=self.device)
        results: List[Dict[str, Any]] = []
        for det in detections[0].boxes.data:
            x1, y1, x2, y2, conf, _ = det.cpu().numpy()
            if conf >= Config().detection_confidence_threshold:
                results.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(conf)})
        return self._filter_by_size(results)

    def _track_internal(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.track(frame, persist=True, verbose=False, device=self.device)
        results: List[Dict[str, Any]] = []
        if detections[0].boxes.id is None:
            return results

        track_ids = detections[0].boxes.id.int().cpu().tolist()
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confs = detections[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            if conf >= Config().detection_confidence_threshold:
                results.append(
                    {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "confidence": float(conf),
                        "track_id": track_id,
                    }
                )
        return self._filter_by_size(results)

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return []

        self._maybe_reset_tracker(frame.shape)
        if not self._tracking_supported:
            return self.detect(frame)

        try:
            return self._track_internal(frame)
        except ModuleNotFoundError:
            self._tracking_supported = False
            logger.warning("Отключаем трекинг YOLO: отсутствуют зависимости")
            return self.detect(frame)
        except Exception:
            self._tracking_supported = False
            self._reset_tracker_state()
            logger.exception("Отключаем трекинг YOLO из-за ошибки, переключаемся на detect")
            return self.detect(frame)
