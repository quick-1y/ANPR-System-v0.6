#anpr/inftastructure/event_writer.py
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
from PyQt5 import QtGui

from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.storage import AsyncEventDatabase

logger = get_logger(__name__)


class EventWriter:
    """Сервис записи событий (скриншоты + база данных)."""

    def __init__(self, storage: AsyncEventDatabase, screenshot_dir: str) -> None:
        self._storage = storage
        self._screenshot_dir = screenshot_dir
        os.makedirs(self._screenshot_dir, exist_ok=True)

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        """Очищает строку для использования в имени файла."""
        normalized = value.replace(os.sep, "_")
        safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in normalized]
        return "".join(safe_chars) or "event"

    def _build_screenshot_paths(self, channel_name: str, plate: str) -> tuple[str, str]:
        """Генерирует пути для сохранения скриншотов."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        channel_safe = self._sanitize_for_filename(channel_name)
        plate_safe = self._sanitize_for_filename(plate or "plate")
        uid = uuid.uuid4().hex[:8]
        base = f"{timestamp}_{channel_safe}_{plate_safe}_{uid}"

        return (
            os.path.join(self._screenshot_dir, f"{base}_frame.jpg"),
            os.path.join(self._screenshot_dir, f"{base}_plate.jpg"),
        )

    @staticmethod
    def _save_bgr_image(path: str, image: Optional[cv2.Mat]) -> Optional[str]:
        """Сохраняет BGR изображение на диск."""
        if image is None or image.size == 0:
            return None

        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if cv2.imwrite(path, image):
                return path
        except Exception as exc:
            logger.exception("Не удалось сохранить скриншот по пути %s: %s", path, exc)

        return None

    @staticmethod
    def _to_qimage(frame: Optional[cv2.Mat], *, is_rgb: bool = False) -> Optional[QtGui.QImage]:
        """Конвертирует OpenCV Mat в QImage."""
        if frame is None or frame.size == 0:
            return None

        rgb_frame = frame if is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        return QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()

    async def write_events(
        self,
        source: str,
        results: list[dict],
        channel_name: str,
        frame: cv2.Mat,
        rgb_frame: Optional[cv2.Mat] = None,
    ) -> list[dict]:
        """Обрабатывает результаты распознавания и сохраняет события."""
        events: list[dict] = []
        for res in results:
            if res.get("unreadable"):
                logger.debug(
                    "Канал %s: номер помечен как нечитаемый (confidence=%.2f)",
                    channel_name,
                    res.get("confidence", 0.0),
                )
                continue

            if not res.get("text"):
                continue

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "channel": channel_name,
                "plate": res.get("text", ""),
                "country": res.get("country"),
                "confidence": res.get("confidence", 0.0),
                "source": source,
                "direction": res.get("direction"),
            }

            x1, y1, x2, y2 = res.get("bbox", (0, 0, 0, 0))
            plate_crop = frame[y1:y2, x1:x2] if frame is not None else None

            frame_path, plate_path = self._build_screenshot_paths(channel_name, event["plate"])
            event["frame_path"] = self._save_bgr_image(frame_path, frame)
            event["plate_path"] = self._save_bgr_image(plate_path, plate_crop)

            event["frame_image"] = self._to_qimage(rgb_frame, is_rgb=True)
            event["plate_image"] = self._to_qimage(plate_crop) if plate_crop is not None else None

            event["id"] = await self._storage.insert_event_async(
                channel=event["channel"],
                plate=event["plate"],
                country=event.get("country"),
                confidence=event["confidence"],
                source=event["source"],
                timestamp=event["timestamp"],
                frame_path=event.get("frame_path"),
                plate_path=event.get("plate_path"),
                direction=event.get("direction"),
            )

            events.append(event)
            logger.info(
                "Канал %s: номер %s (conf=%.2f, dir=%s, track=%s)",
                event["channel"],
                event["plate"],
                event["confidence"],
                event.get("direction") or "-",
                res.get("track_id", "-"),
            )

        return events
