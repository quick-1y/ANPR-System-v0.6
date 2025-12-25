#!/usr/bin/env python3
# /anpr/pipeline/factory.py
from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

from anpr.config import Config
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline
from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.postprocessing.validator import PlatePostProcessor
from anpr.recognition.crnn_recognizer import CRNNRecognizer


_RECOGNIZER_LOCK = threading.RLock()
_RECOGNIZER_INITIALIZING = False
_RECOGNIZER_READY = threading.Event()
_RECOGNIZER_SINGLETON: CRNNRecognizer | None = None


class _FallbackRecognizer:
    """Неблокирующий заглушка, пока OCR ещё не инициализирован."""

    def recognize(self, _plate_image):
        return "", 0.0

    def recognize_batch(self, _plate_images):
        return []


_NOOP_RECOGNIZER = _FallbackRecognizer()


def _initialize_recognizer_threadsafe() -> CRNNRecognizer:
    config = Config()
    return CRNNRecognizer(config.ocr_model_path, config.device)


def _get_fallback_recognizer() -> CRNNRecognizer:
    return _RECOGNIZER_SINGLETON or _NOOP_RECOGNIZER


def _get_shared_recognizer() -> CRNNRecognizer:
    """Lazily initializes a single OCR recognizer instance for all pipelines.

    CRNN quantization with ``prepare_fx`` is not thread-safe, so creating the
    recognizer concurrently for multiple channels can crash. By guarding
    initialization with a lock and reusing the instance across pipelines, we
    avoid the race while keeping inference stateless and reusable.
    """

    global _RECOGNIZER_INITIALIZING, _RECOGNIZER_SINGLETON

    if _RECOGNIZER_SINGLETON is None and not _RECOGNIZER_INITIALIZING:
        with _RECOGNIZER_LOCK:
            if _RECOGNIZER_SINGLETON is None and not _RECOGNIZER_INITIALIZING:
                _RECOGNIZER_INITIALIZING = True
                _RECOGNIZER_READY.clear()

                def _init() -> None:
                    global _RECOGNIZER_INITIALIZING, _RECOGNIZER_SINGLETON

                    try:
                        _RECOGNIZER_SINGLETON = _initialize_recognizer_threadsafe()
                    finally:
                        _RECOGNIZER_INITIALIZING = False
                        _RECOGNIZER_READY.set()

                threading.Thread(target=_init, daemon=True).start()

    if not _RECOGNIZER_READY.wait(timeout=0.1):
        _RECOGNIZER_READY.wait()

    return _RECOGNIZER_SINGLETON or _get_fallback_recognizer()


def _build_postprocessor(config: Dict[str, object]) -> PlatePostProcessor:
    config_dir = str(config.get("config_dir") or "config/countries")
    enabled_countries = config.get("enabled_countries")
    loader = CountryConfigLoader(os.path.abspath(config_dir))
    loader.ensure_dir()
    return PlatePostProcessor(loader, enabled_countries)


def build_components(
    best_shots: int,
    cooldown_seconds: int,
    min_confidence: float,
    plate_config: Dict[str, object] | None = None,
    direction_config: Dict[str, object] | None = None,
    min_plate_size: Dict[str, int] | None = None,
    max_plate_size: Dict[str, int] | None = None,
    size_filter_enabled: bool = True,
    segmentation_config: Dict[str, object] | None = None,
) -> Tuple[ANPRPipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    config = Config()
    detector = YOLODetector(
        config.yolo_model_path,
        config.device,
        min_plate_size=min_plate_size,
        max_plate_size=max_plate_size,
        size_filter_enabled=size_filter_enabled,
        detection_confidence_threshold=config.detection_confidence_threshold,
    )
    recognizer = _get_shared_recognizer()
    postprocessor = _build_postprocessor(plate_config or {})
    pipeline = ANPRPipeline(
        recognizer,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
        postprocessor=postprocessor,
        direction_config=direction_config,
        segmentation_config=segmentation_config,
    )
    return pipeline, detector
