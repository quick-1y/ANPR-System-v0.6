#!/usr/bin/env python3
# /anpr/workers/channel_worker.py
import asyncio
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import threading
import atexit
from multiprocessing import shared_memory
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui

from anpr.detection.motion_detector import MotionDetectorConfig
from anpr.infrastructure.event_writer import EventWriter
from anpr.pipeline.anpr_pipeline import TrackDirectionEstimator
from anpr.pipeline.factory import build_components
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.settings_manager import SettingsManager
from anpr.infrastructure.storage import AsyncEventDatabase
from anpr.workers.motion_controller import MotionController

logger = get_logger(__name__)


@dataclass
class Region:
    """Произвольная область кадра, заданная точками."""

    points: List[Tuple[float, float]]
    unit: str = "px"

    @classmethod
    def from_dict(cls, region_conf: Optional[Dict[str, Any]]) -> "Region":
        if not region_conf:
            return cls(points=[], unit="px")

        unit = str(region_conf.get("unit", "px")).lower()
        raw_points = region_conf.get("points")
        if raw_points:
            points = [
                (float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
            return cls(points=points, unit=unit if unit in ("px", "percent") else "px")

        x = float(region_conf.get("x", 0))
        y = float(region_conf.get("y", 0))
        width = float(region_conf.get("width", 100))
        height = float(region_conf.get("height", 100))
        rect_points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        return cls(points=rect_points, unit="percent")

    def _clamp_points(self, points: List[Tuple[int, int]], width: int, height: int) -> List[Tuple[int, int]]:
        if not points:
            return []
        clamped: List[Tuple[int, int]] = []
        for x, y in points:
            clamped.append(
                (
                    max(0, min(width - 1, int(round(x)))),
                    max(0, min(height - 1, int(round(y)))),
                )
            )
        return clamped

    def polygon_points(self, frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        height, width, _ = frame_shape
        if not self.points:
            return [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

        if self.unit == "percent":
            scaled = [
                (width * x / 100.0, height * y / 100.0)
                for (x, y) in self.points
            ]
        else:
            scaled = self.points

        return self._clamp_points([(int(x), int(y)) for x, y in scaled], width, height)

    def bounding_rect(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        polygon = self.polygon_points(frame_shape)
        if not polygon:
            height, width, _ = frame_shape
            return 0, 0, width, height

        xs, ys = zip(*polygon)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return x1, y1, x2 + 1, y2 + 1

    def is_full_frame(self) -> bool:
        return not self.points


@dataclass
class DebugOptions:
    show_detection_boxes: bool = False
    show_ocr_text: bool = False
    show_direction_tracks: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DebugOptions":
        debug_conf = data or {}
        return cls(
            show_detection_boxes=bool(debug_conf.get("show_detection_boxes", False)),
            show_ocr_text=bool(debug_conf.get("show_ocr_text", False)),
            show_direction_tracks=bool(debug_conf.get("show_direction_tracks", False)),
        )


@dataclass
class PlateSize:
    """Размер рамки номерного знака в пикселях."""

    width: int = 0
    height: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None) -> "PlateSize":
        defaults = defaults or {}
        width = int((data or {}).get("width", defaults.get("width", 0)) or 0)
        height = int((data or {}).get("height", defaults.get("height", 0)) or 0)
        return cls(width=max(0, width), height=max(0, height))

    def to_dict(self) -> Dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}


@dataclass
class DirectionSettings:
    """Параметры оценки направления движения."""

    history_size: int = 12
    min_track_length: int = 3
    smoothing_window: int = 5
    confidence_threshold: float = 0.55
    jitter_pixels: float = 1.0
    min_area_change_ratio: float = 0.02

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "DirectionSettings":
        defaults = defaults or {}
        data = data or {}
        return cls(
            history_size=int(data.get("history_size", defaults.get("history_size", cls.history_size))),
            min_track_length=int(data.get("min_track_length", defaults.get("min_track_length", cls.min_track_length))),
            smoothing_window=int(data.get("smoothing_window", defaults.get("smoothing_window", cls.smoothing_window))),
            confidence_threshold=float(
                data.get("confidence_threshold", defaults.get("confidence_threshold", cls.confidence_threshold))
            ),
            jitter_pixels=float(data.get("jitter_pixels", defaults.get("jitter_pixels", cls.jitter_pixels))),
            min_area_change_ratio=float(
                data.get("min_area_change_ratio", defaults.get("min_area_change_ratio", cls.min_area_change_ratio))
            ),
        )

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "history_size": int(self.history_size),
            "min_track_length": int(self.min_track_length),
            "smoothing_window": int(self.smoothing_window),
            "confidence_threshold": float(self.confidence_threshold),
            "jitter_pixels": float(self.jitter_pixels),
            "min_area_change_ratio": float(self.min_area_change_ratio),
        }


@dataclass
class ReconnectPolicy:
    """Политика переподключения канала."""

    enabled: bool
    frame_timeout_seconds: float
    retry_interval_seconds: float
    periodic_enabled: bool
    periodic_reconnect_seconds: float

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "ReconnectPolicy":
        reconnect_conf = config or {}
        signal_loss_conf = reconnect_conf.get("signal_loss", {})
        periodic_conf = reconnect_conf.get("periodic", {})
        return cls(
            enabled=bool(signal_loss_conf.get("enabled", False)),
            frame_timeout_seconds=float(signal_loss_conf.get("frame_timeout_seconds", 5)),
            retry_interval_seconds=float(signal_loss_conf.get("retry_interval_seconds", 5)),
            periodic_enabled=bool(periodic_conf.get("enabled", False)),
            periodic_reconnect_seconds=float(periodic_conf.get("interval_minutes", 0)) * 60,
        )


@dataclass
class ChannelRuntimeConfig:
    """Нормализованная конфигурация канала."""

    name: str
    source: str
    best_shots: int
    cooldown_seconds: int
    min_confidence: float
    detector_frame_stride: int
    detection_mode: str
    motion_threshold: float
    motion_frame_stride: int
    motion_activation_frames: int
    motion_release_frames: int
    roi_enabled: bool
    region: Region
    debug: DebugOptions
    size_filter_enabled: bool
    min_plate_size: PlateSize
    max_plate_size: PlateSize
    direction: DirectionSettings

    @classmethod
    def from_dict(cls, channel_conf: Dict[str, Any]) -> "ChannelRuntimeConfig":
        settings = SettingsManager()
        size_defaults = settings.get_plate_size_defaults()
        direction_defaults = settings.get_direction_defaults()
        return cls(
            name=channel_conf.get("name", "Канал"),
            source=str(channel_conf.get("source", "0")),
            best_shots=int(channel_conf.get("best_shots", 3)),
            cooldown_seconds=int(channel_conf.get("cooldown_seconds", 5)),
            min_confidence=float(channel_conf.get("ocr_min_confidence", 0.6)),
            detector_frame_stride=max(1, int(channel_conf.get("detector_frame_stride", 2))),
            detection_mode=channel_conf.get("detection_mode", "continuous"),
            motion_threshold=float(channel_conf.get("motion_threshold", 0.01)),
            motion_frame_stride=int(channel_conf.get("motion_frame_stride", 1)),
            motion_activation_frames=int(channel_conf.get("motion_activation_frames", 3)),
            motion_release_frames=int(channel_conf.get("motion_release_frames", 6)),
            roi_enabled=bool(channel_conf.get("roi_enabled", True)),
            region=Region.from_dict(channel_conf.get("region")),
            debug=DebugOptions.from_dict(channel_conf.get("debug")),
            size_filter_enabled=bool(channel_conf.get("size_filter_enabled", True)),
            min_plate_size=PlateSize.from_dict(channel_conf.get("min_plate_size"), size_defaults.get("min_plate_size")),
            max_plate_size=PlateSize.from_dict(channel_conf.get("max_plate_size"), size_defaults.get("max_plate_size")),
            direction=DirectionSettings.from_dict(channel_conf.get("direction"), direction_defaults),
        )


# Общий ProcessPoolExecutor (master-worker) для всех каналов
_INFERENCE_EXECUTOR: ProcessPoolExecutor | None = None
_INFERENCE_EXECUTOR_LOCK = threading.Lock()


def _config_fingerprint(config: dict) -> str:
    """Детерминированный отпечаток для разделения экзекьюторов."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _get_inference_executor() -> ProcessPoolExecutor:
    """Возвращает общий ProcessPoolExecutor для инференса."""
    global _INFERENCE_EXECUTOR
    with _INFERENCE_EXECUTOR_LOCK:
        if _INFERENCE_EXECUTOR is None:
            settings = SettingsManager()
            inference_conf = settings.get_inference_settings()
            max_workers = int(inference_conf.get("workers", max(1, os.cpu_count() or 1)))
            _INFERENCE_EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
    return _INFERENCE_EXECUTOR


def _shutdown_executors() -> None:
    """Завершает все экзекьюторы при выходе из программы."""
    global _INFERENCE_EXECUTOR
    with _INFERENCE_EXECUTOR_LOCK:
        if _INFERENCE_EXECUTOR is not None:
            _INFERENCE_EXECUTOR.shutdown(cancel_futures=True)
            _INFERENCE_EXECUTOR = None


atexit.register(_shutdown_executors)


def _offset_detections_process(
    detections: list[dict], roi_rect: Tuple[int, int, int, int]
) -> list[dict]:
    """Смещает координаты детекций относительно ROI."""
    x1, y1, _, _ = roi_rect
    adjusted: list[dict] = []
    for det in detections:
        box = det.get("bbox")
        if not box:
            continue
        det_copy = det.copy()
        det_copy["bbox"] = [
            int(box[0] + x1),
            int(box[1] + y1),
            int(box[2] + x1),
            int(box[3] + y1),
        ]
        adjusted.append(det_copy)
    return adjusted


def _run_inference_task(
    frame: cv2.Mat,
    roi_frame: cv2.Mat,
    roi_rect: Tuple[int, int, int, int],
    config: dict,
) -> Tuple[list[dict], list[dict]]:
    """
    Выполняет инференс в отдельном процессе.
    Модели кэшируются локально в каждом процессе.
    """
    
    # Локальный кэш моделей для этого процесса
    if not hasattr(_run_inference_task, "_local_cache"):
        _run_inference_task._local_cache = {}
    
    # Создаем ключ для кэширования
    key = _config_fingerprint(config)
    
    # Получаем или создаем модели в этом процессе
    if key not in _run_inference_task._local_cache:
        logger.debug(f"Создание моделей inference для конфига {key[:32]}...")
        pipeline, detector = build_components(
            config["best_shots"],
            config["cooldown_seconds"],
            config["min_confidence"],
            config.get("plate_config", {}),
            config.get("direction", {}),
            config.get("min_plate_size"),
            config.get("max_plate_size"),
            config.get("size_filter_enabled", True),
        )
        _run_inference_task._local_cache[key] = (pipeline, detector)

    pipeline, detector = _run_inference_task._local_cache[key]

    # Выполняем детекцию и распознавание
    detections = detector.track(roi_frame)
    detections = _offset_detections_process(detections, roi_rect)
    results = pipeline.process_frame(frame, detections)
    
    return detections, results


@dataclass(frozen=True)
class SharedFrameInfo:
    name: str
    shape: Tuple[int, int, int]
    dtype: str


def _create_shared_frame(frame: cv2.Mat) -> tuple[SharedFrameInfo, shared_memory.SharedMemory]:
    """Копирует кадр в shared memory и возвращает метаданные."""
    contiguous = np.ascontiguousarray(frame)
    shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
    shm_array = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    shm_array[:] = contiguous
    info = SharedFrameInfo(name=shm.name, shape=contiguous.shape, dtype=str(contiguous.dtype))
    return info, shm


def _load_shared_frame(info: SharedFrameInfo) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """Открывает shared memory и возвращает ndarray."""
    shm = shared_memory.SharedMemory(name=info.name)
    array = np.ndarray(info.shape, dtype=np.dtype(info.dtype), buffer=shm.buf)
    return array, shm


def _run_inference_task_shared(
    frame_info: SharedFrameInfo,
    roi_info: SharedFrameInfo,
    roi_rect: Tuple[int, int, int, int],
    config: dict,
) -> Tuple[list[dict], list[dict]]:
    """Выполняет инференс, используя shared memory для кадров."""
    frame, frame_shm = _load_shared_frame(frame_info)
    roi_frame, roi_shm = _load_shared_frame(roi_info)
    try:
        return _run_inference_task(frame, roi_frame, roi_rect, config)
    finally:
        frame_shm.close()
        roi_shm.close()


class InferenceLimiter:
    """Пропускает лишние кадры для инференса детектора."""

    def __init__(self, stride: int) -> None:
        self.stride = max(1, stride)
        self._counter = 0

    def allow(self) -> bool:
        should_run = self._counter == 0
        self._counter = (self._counter + 1) % self.stride
        return should_run


class ChannelWorker(QtCore.QThread):
    """Background worker that captures frames, runs ANPR pipeline and emits UI events."""

    frame_ready = QtCore.pyqtSignal(str, QtGui.QImage)
    event_ready = QtCore.pyqtSignal(dict)
    status_ready = QtCore.pyqtSignal(str, str)
    metrics_ready = QtCore.pyqtSignal(str, dict)

    def __init__(
        self,
        channel_conf: Dict,
        db_path: str,
        screenshot_dir: str,
        reconnect_conf: Optional[Dict[str, Any]] = None,
        plate_config: Optional[Dict[str, Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.channel_id = int(channel_conf.get("id", 0))
        self.config = ChannelRuntimeConfig.from_dict(channel_conf)
        self.reconnect_policy = ReconnectPolicy.from_dict(reconnect_conf)
        self.db_path = db_path
        self.screenshot_dir = screenshot_dir
        self._running = True
        self.plate_config = plate_config or {}
        self._config_lock = threading.Lock()

        motion_config = MotionDetectorConfig(
            threshold=self.config.motion_threshold,
            frame_stride=self.config.motion_frame_stride,
            activation_frames=self.config.motion_activation_frames,
            release_frames=self.config.motion_release_frames,
        )
        self.motion_controller = MotionController(self.config.detection_mode, motion_config)
        self._inference_limiter = InferenceLimiter(self.config.detector_frame_stride)
        self._inference_task: Optional[asyncio.Task] = None
        self._last_debug: Dict[str, list] = {"detections": [], "results": []}
        self._track_history: dict[int, deque[tuple[int, int]]] = {}
        self._track_last_seen: dict[int, float] = {}
        self._track_directions: dict[int, str] = {}
        self._track_history_size = 32
        self._track_stale_seconds = 3.0
        self._frame_times: deque[float] = deque(maxlen=120)
        self._latency_ms: deque[float] = deque(maxlen=40)
        self._confidence_scores: deque[float] = deque(maxlen=60)
        self._last_metrics_emit = 0.0
        self._metrics_interval = 1.0
        inference_conf = SettingsManager().get_inference_settings()
        self._use_shared_memory = bool(inference_conf.get("shared_memory", True))
        self._roi_mask_cache: Optional[np.ndarray] = None
        self._roi_mask_key: Optional[tuple] = None

    def update_runtime_config(
        self,
        channel_conf: Dict[str, Any],
        reconnect_conf: Optional[Dict[str, Any]] = None,
        plate_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Обновляет параметры канала без перезапуска потока."""
        new_config = ChannelRuntimeConfig.from_dict(channel_conf)
        new_reconnect = ReconnectPolicy.from_dict(reconnect_conf)
        motion_config = MotionDetectorConfig(
            threshold=new_config.motion_threshold,
            frame_stride=new_config.motion_frame_stride,
            activation_frames=new_config.motion_activation_frames,
            release_frames=new_config.motion_release_frames,
        )
        with self._config_lock:
            self.config = new_config
            self.reconnect_policy = new_reconnect
            self.motion_controller = MotionController(new_config.detection_mode, motion_config)
            self._inference_limiter = InferenceLimiter(new_config.detector_frame_stride)
            self.plate_config = plate_config or {}
            self._roi_mask_cache = None
            self._roi_mask_key = None

    def _should_continue(self) -> bool:
        return self._running and not self.isInterruptionRequested()

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        """Открывает видеопоток с учетом типа источника."""
        try:
            capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
            if not capture.isOpened():
                logger.warning(f"Не удалось открыть источник: {source}")
                return None
            return capture
        except Exception as e:
            logger.error(f"Ошибка открытия источника {source}: {e}")
            return None

    async def _open_with_retries(self, source: str, channel_name: str) -> Optional[cv2.VideoCapture]:
        """Подключает источник с учетом настроек переподключения."""

        while self._should_continue():
            capture = await asyncio.to_thread(self._open_capture, source)
            if capture is not None:
                self.status_ready.emit(channel_name, "")
                return capture

            if not self.reconnect_policy.enabled:
                self.status_ready.emit(channel_name, "Нет сигнала")
                return None

            self.status_ready.emit(
                channel_name,
                f"Нет сигнала, повтор через {int(self.reconnect_policy.retry_interval_seconds)}с",
            )
            await asyncio.sleep(max(0.1, self.reconnect_policy.retry_interval_seconds))

        return None

    def _inference_config(self) -> dict:
        """Возвращает конфигурацию для inference."""
        with self._config_lock:
            plate_config = dict(self.plate_config)
            config = self.config
        if plate_config.get("config_dir"):
            plate_config["config_dir"] = os.path.abspath(str(plate_config.get("config_dir")))

        return {
            "channel_id": self.channel_id,
            "best_shots": config.best_shots,
            "cooldown_seconds": config.cooldown_seconds,
            "min_confidence": config.min_confidence,
            "plate_config": plate_config,
            "min_plate_size": config.min_plate_size.to_dict(),
            "max_plate_size": config.max_plate_size.to_dict(),
            "direction": config.direction.to_dict(),
            "size_filter_enabled": config.size_filter_enabled,
        }

    def _extract_region(self, frame: cv2.Mat) -> Tuple[cv2.Mat, Tuple[int, int, int, int]]:
        """Извлекает ROI из кадра с учетом произвольной формы."""
        with self._config_lock:
            region = self.config.region
            roi_enabled = self.config.roi_enabled
        if not roi_enabled:
            height, width, _ = frame.shape
            return frame, (0, 0, width, height)

        polygon = region.polygon_points(frame.shape)
        x1, y1, x2, y2 = region.bounding_rect(frame.shape)
        roi_frame = frame[y1:y2, x1:x2]

        if not region.is_full_frame() and roi_frame.size:
            local_polygon = np.array([[(x - x1), (y - y1)] for x, y in polygon], dtype=np.int32)
            cache_key = (frame.shape, tuple(polygon))
            if self._roi_mask_key != cache_key or self._roi_mask_cache is None:
                mask = np.zeros((roi_frame.shape[0], roi_frame.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [local_polygon], 255)
                self._roi_mask_cache = mask
                self._roi_mask_key = cache_key
            else:
                mask = self._roi_mask_cache
            roi_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)

        return roi_frame, (x1, y1, x2, y2)

    async def _run_inference(
        self, frame: cv2.Mat, roi_frame: cv2.Mat, roi_rect: Tuple[int, int, int, int]
    ) -> Tuple[list[dict], list[dict]]:
        """Запускает инференс в отдельном процессе."""
        loop = asyncio.get_running_loop()
        config = self._inference_config()
        if not self._use_shared_memory:
            return await loop.run_in_executor(
                _get_inference_executor(),
                _run_inference_task,
                frame,
                roi_frame,
                roi_rect,
                config,
            )

        frame_info = roi_info = None
        frame_shm = roi_shm = None
        try:
            frame_info, frame_shm = _create_shared_frame(frame)
            roi_info, roi_shm = _create_shared_frame(roi_frame)
            return await loop.run_in_executor(
                _get_inference_executor(),
                _run_inference_task_shared,
                frame_info,
                roi_info,
                roi_rect,
                config,
            )
        except Exception as exc:
            logger.warning("Shared memory отключена для канала %s: %s", self.config.name, exc)
            return await loop.run_in_executor(
                _get_inference_executor(),
                _run_inference_task,
                frame,
                roi_frame,
                roi_rect,
                config,
            )
        finally:
            for shm in (frame_shm, roi_shm):
                if shm is None:
                    continue
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    logger.debug("Ошибка освобождения shared memory: %s", exc)

    @staticmethod
    def _draw_label(frame: cv2.Mat, text: str, origin: Tuple[int, int]) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        x = max(0, x)
        y = max(text_size[1] + 4, y)
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 6),
            (x + text_size[0] + 8, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x + 4, y - 2), font, scale, (0, 255, 0), thickness)

    def _update_track_history(self, detections: list[dict]) -> None:
        with self._config_lock:
            debug_config = self.config.debug
        if not debug_config.show_direction_tracks:
            return

        now = time.monotonic()
        for det in detections:
            track_id = det.get("track_id")
            bbox = det.get("bbox")

            if track_id is None or not bbox or len(bbox) != 4:
                continue

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            history = self._track_history.setdefault(
                int(track_id), deque(maxlen=self._track_history_size)
            )
            history.append((cx, cy))
            self._track_last_seen[int(track_id)] = now

            direction = det.get("direction")
            if direction:
                self._track_directions[int(track_id)] = str(direction)

    def _draw_debug_info(self, frame: cv2.Mat) -> None:
        with self._config_lock:
            debug_config = self.config.debug
        if not (
            debug_config.show_detection_boxes
            or debug_config.show_ocr_text
            or debug_config.show_direction_tracks
        ):
            return

        detections = self._last_debug.get("detections", [])
        results = self._last_debug.get("results", [])

        if debug_config.show_detection_boxes:
            for det in detections:
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 200, 0), 2)
                if debug_config.show_ocr_text:
                    label = det.get("text") or ""
                    self._draw_label(frame, label, (bbox[0], bbox[1] - 6))

        if debug_config.show_ocr_text:
            for res in results:
                text = res.get("text")
                bbox = res.get("bbox") or res.get("plate_bbox")
                if not text or not bbox or len(bbox) != 4:
                    continue
                self._draw_label(frame, str(text), (bbox[0], bbox[1] - 6))

        if debug_config.show_direction_tracks:
            self._draw_direction_tracks(frame)

    def _draw_direction_tracks(self, frame: cv2.Mat) -> None:
        now = time.monotonic()
        stale_tracks = [
            track_id
            for track_id, last_seen in self._track_last_seen.items()
            if now - last_seen > self._track_stale_seconds
        ]

        for track_id in stale_tracks:
            self._track_history.pop(track_id, None)
            self._track_last_seen.pop(track_id, None)
            self._track_directions.pop(track_id, None)

        direction_colors = {
            "APPROACHING": (0, 200, 0),
            "RECEDING": (220, 0, 0),
            "UNKNOWN": (200, 200, 0),
        }

        for track_id, history in self._track_history.items():
            if not history:
                continue

            points = np.array(history, dtype=np.int32)
            direction = self._track_directions.get(track_id, TrackDirectionEstimator.UNKNOWN)
            color = direction_colors.get(direction, (180, 180, 180))

            if len(points) > 1:
                cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, color, 2)

            tail_x, tail_y = points[-1]
            cv2.circle(frame, (int(tail_x), int(tail_y)), 4, color, -1)
            label = f"#{track_id} {direction}"
            self._draw_label(frame, label, (int(tail_x) + 6, int(tail_y) - 6))

    def _update_metrics(self, now: float) -> None:
        self._frame_times.append(now)
        window_seconds = 2.0
        while self._frame_times and now - self._frame_times[0] > window_seconds:
            self._frame_times.popleft()

        if now - self._last_metrics_emit < self._metrics_interval:
            return

        fps = 0.0
        if len(self._frame_times) >= 2:
            duration = self._frame_times[-1] - self._frame_times[0]
            if duration > 0:
                fps = (len(self._frame_times) - 1) / duration

        latency_ms = float(np.mean(self._latency_ms)) if self._latency_ms else None
        accuracy = (
            float(np.mean(self._confidence_scores)) * 100.0
            if self._confidence_scores
            else None
        )

        self.metrics_ready.emit(
            self.config.name,
            {
                "fps": fps,
                "latency_ms": latency_ms,
                "accuracy": accuracy,
            },
        )
        self._last_metrics_emit = now

    async def _inference_and_process(
        self,
        event_writer: EventWriter,
        source: str,
        channel_name: str,
        frame: cv2.Mat,
        roi_frame: cv2.Mat,
        roi_rect: Tuple[int, int, int, int],
        rgb_frame: cv2.Mat,
    ) -> None:
        """Выполняет инференс и обработку результатов."""
        try:
            start_ts = time.monotonic()
            detections, results = await self._run_inference(frame, roi_frame, roi_rect)
            latency_ms = (time.monotonic() - start_ts) * 1000.0
            self._latency_ms.append(latency_ms)
            self._last_debug = {"detections": detections, "results": results}
            self._update_track_history(detections)
            confidences = [
                float(res.get("confidence", 0.0))
                for res in results
                if res.get("text") and not res.get("unreadable")
            ]
            if confidences:
                self._confidence_scores.extend(confidences)
            events = await event_writer.write_events(
                source,
                results,
                channel_name,
                frame,
                rgb_frame,
            )
            for event in events:
                self.event_ready.emit(event)
        except Exception as e:
            logger.exception("Ошибка инференса для канала %s: %s", channel_name, e)

    async def _loop(self) -> None:
        """Основной цикл обработки канала."""
        storage = AsyncEventDatabase(self.db_path)
        event_writer = EventWriter(storage, self.screenshot_dir)

        with self._config_lock:
            source = self.config.source
            channel_name = self.config.name
        
        # Подключаемся к источнику
        capture = await self._open_with_retries(source, channel_name)
        if capture is None:
            logger.warning("Не удалось открыть источник %s для канала %s", source, channel_name)
            return
        
        logger.info("Канал %s запущен (источник=%s)", channel_name, source)
        
        last_frame_ts = time.monotonic()
        last_reconnect_ts = last_frame_ts
        
        while self._should_continue():
            now = time.monotonic()
            with self._config_lock:
                channel_name = self.config.name
                source = self.config.source
                reconnect_policy = self.reconnect_policy
                motion_controller = self.motion_controller
                inference_limiter = self._inference_limiter
            
            # Плановое переподключение
            if (
                reconnect_policy.periodic_enabled
                and reconnect_policy.periodic_reconnect_seconds > 0
                and now - last_reconnect_ts >= reconnect_policy.periodic_reconnect_seconds
            ):
                self.status_ready.emit(channel_name, "Плановое переподключение...")
                capture.release()
                capture = await self._open_with_retries(source, channel_name)
                if capture is None:
                    logger.warning("Переподключение не удалось для канала %s", channel_name)
                    break
                last_reconnect_ts = time.monotonic()
                last_frame_ts = last_reconnect_ts
                continue

            # Чтение кадра
            ret, frame = await asyncio.to_thread(capture.read)
            
            # Проверка потери сигнала
            if not ret or frame is None:
                if not reconnect_policy.enabled:
                    self.status_ready.emit(channel_name, "Поток остановлен")
                    logger.warning("Поток остановлен для канала %s", channel_name)
                    break

                if time.monotonic() - last_frame_ts < reconnect_policy.frame_timeout_seconds:
                    await asyncio.sleep(0.05)
                    continue

                self.status_ready.emit(channel_name, "Потеря сигнала, переподключение...")
                logger.warning("Потеря сигнала на канале %s, выполняем переподключение", channel_name)
                capture.release()
                capture = await self._open_with_retries(source, channel_name)
                if capture is None:
                    logger.warning("Переподключение не удалось для канала %s", channel_name)
                    break
                
                last_reconnect_ts = time.monotonic()
                last_frame_ts = last_reconnect_ts
                continue

            last_frame_ts = time.monotonic()
            self._update_metrics(last_frame_ts)

            # Обработка кадра
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_frame, roi_rect = self._extract_region(frame)
            motion_detected, motion_status = motion_controller.update(roi_frame)
            if motion_status:
                self.status_ready.emit(channel_name, motion_status)

            if motion_detected:
                # Запуск инференса с учетом stride
                if inference_limiter.allow():
                    if self._inference_task is None or self._inference_task.done():
                        self._inference_task = asyncio.create_task(
                            self._inference_and_process(
                                event_writer,
                                source,
                                channel_name,
                                frame.copy(),
                                roi_frame.copy(),
                                roi_rect,
                                rgb_frame.copy(),
                            )
                        )
                    else:
                        logger.debug(
                            "Канал %s: пропуск инференса, предыдущая задача еще выполняется",
                            channel_name,
                        )

            # Отправка кадра в UI
            display_frame = rgb_frame
            with self._config_lock:
                debug_config = self.config.debug
            if debug_config.show_detection_boxes or debug_config.show_ocr_text:
                display_frame = rgb_frame.copy()
                self._draw_debug_info(display_frame)

            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(
                display_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit(channel_name, q_image)

        # Завершение работы
        capture.release()

        if self._inference_task is not None:
            try:
                await asyncio.wait_for(self._inference_task, timeout=1)
            except Exception as e:
                logger.warning("Задача инференса для канала %s не завершена корректно: %s", 
                             channel_name, e)

    def run(self) -> None:
        """Запуск потока."""
        if not self._should_continue():
            return
        try:
            asyncio.run(self._loop())
        except Exception as exc:
            with self._config_lock:
                channel_name = self.config.name
            self.status_ready.emit(channel_name, f"Ошибка: {exc}")
            logger.exception("Канал %s аварийно остановлен", channel_name)

    def stop(self) -> None:
        """Остановка потока."""
        self._running = False
        self.requestInterruption()
        if self._inference_task is not None:
            self._inference_task.cancel()
            self._inference_task = None

    @classmethod
    def shutdown_executor(cls) -> None:
        """Принудительно останавливает общий пул процессов инференса."""
        _shutdown_executors()
