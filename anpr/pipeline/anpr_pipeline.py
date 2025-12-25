# /anpr/pipeline/anpr_pipeline.py
"""Пайплайн объединяющий детекцию и OCR."""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np

from anpr.postprocessing.validator import PlatePostProcessor
from anpr.preprocessing.plate_preprocessor import PlatePreprocessor
from anpr.recognition.crnn_recognizer import CRNNRecognizer


class TrackAggregator:
    """Агрегирует результаты распознавания в рамках одного трека."""

    def __init__(self, best_shots: int):
        self.best_shots = max(1, best_shots)
        self.track_texts: Dict[int, List[tuple[str, float]]] = {}
        self.last_emitted: Dict[int, str] = {}

    def add_result(self, track_id: int, text: str, confidence: float) -> str:
        if not text:
            return ""

        bucket = self.track_texts.setdefault(track_id, [])
        bucket.append((text, max(0.0, float(confidence))))
        if len(bucket) > self.best_shots:
            bucket.pop(0)

        weights: Dict[str, float] = {}
        counts: Counter[str] = Counter()
        total_weight = 0.0
        for entry_text, entry_confidence in bucket:
            weights[entry_text] = weights.get(entry_text, 0.0) + entry_confidence
            counts[entry_text] += 1
            total_weight += entry_confidence

        if not weights or total_weight <= 0:
            return ""

        consensus = max(weights, key=lambda value: (weights[value], counts[value]))
        consensus_weight = weights[consensus]
        quorum = max(1, (self.best_shots + 1) // 2)
        has_quorum = len(bucket) >= self.best_shots and counts[consensus] >= quorum
        has_weighted_majority = consensus_weight >= total_weight * 0.5
        if has_quorum and self.last_emitted.get(track_id) != consensus:
            if not has_weighted_majority:
                return ""
            self.last_emitted[track_id] = consensus
            return consensus
        return ""

    def clear_last(self, track_id: int) -> None:
        self.last_emitted.pop(track_id, None)

    def reset(self, track_id: int) -> None:
        """Полностью сбрасывает историю и последний результат трека."""
        self.track_texts.pop(track_id, None)
        self.last_emitted.pop(track_id, None)


class TrackDirectionEstimator:
    """Оценивает направление движения по истории рамок номера."""

    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        history_size: int = 12,
        min_track_length: int = 3,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.55,
        jitter_pixels: float = 1.0,
        min_area_change_ratio: float = 0.02,
    ) -> None:
        self.history_size = max(1, history_size)
        self.min_track_length = max(1, min_track_length)
        self.smoothing_window = max(1, smoothing_window)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.jitter_pixels = max(0.0, jitter_pixels)
        self.min_area_change_ratio = max(0.0, min_area_change_ratio)
        self._history: Dict[int, deque[tuple[float, float]]] = {}

    @classmethod
    def from_config(cls, config: Dict[str, float | int]) -> "TrackDirectionEstimator":
        return cls(
            history_size=int(config.get("history_size", 12)),
            min_track_length=int(config.get("min_track_length", 3)),
            smoothing_window=int(config.get("smoothing_window", 5)),
            confidence_threshold=float(config.get("confidence_threshold", 0.55)),
            jitter_pixels=float(config.get("jitter_pixels", 1.0)),
            min_area_change_ratio=float(config.get("min_area_change_ratio", 0.02)),
        )

    def _filtered(self, deltas: np.ndarray, threshold: float) -> np.ndarray:
        if deltas.size == 0:
            return deltas
        mask = np.abs(deltas) >= threshold
        return deltas[mask]

    def _recent_trend(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        window = values[-self.smoothing_window :]
        return float(window.mean())

    def _votes(self, vertical_deltas: np.ndarray, area_deltas: np.ndarray, current_area: float) -> list[int]:
        votes: list[int] = []
        filtered_vertical = self._filtered(vertical_deltas, self.jitter_pixels)
        area_threshold = max(self.min_area_change_ratio * max(current_area, 1.0), 1.0)
        filtered_area = self._filtered(area_deltas, area_threshold)

        for delta in filtered_vertical:
            votes.append(1 if delta > 0 else -1)
        for delta in filtered_area:
            votes.append(1 if delta > 0 else -1)
        return votes

    def _confidence(self, score: float, vote_count: int) -> float:
        if vote_count == 0:
            return 0.0
        normalized = np.tanh(abs(score))
        density = min(1.0, vote_count / max(1, self.min_track_length))
        return float(normalized * density)

    def update(self, track_id: int, bbox: list[int]) -> Dict[str, str]:
        if not bbox or len(bbox) != 4:
            return {"direction": self.UNKNOWN}

        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        area = width * height

        history = self._history.setdefault(track_id, deque(maxlen=self.history_size))
        history.append((center_y, area))

        if len(history) < self.min_track_length:
            return {"direction": self.UNKNOWN}

        centers = np.array([item[0] for item in history], dtype=float)
        areas = np.array([item[1] for item in history], dtype=float)
        vertical_deltas = np.diff(centers)
        area_deltas = np.diff(areas)

        trend_vertical = self._recent_trend(vertical_deltas)
        trend_area = self._recent_trend(area_deltas)

        votes = self._votes(vertical_deltas, area_deltas, areas[-1])
        if not votes:
            return {"direction": self.UNKNOWN}

        score = float(np.mean(votes)) * 0.7 + (np.sign(trend_area) if trend_area != 0 else 0.0) * 0.3
        confidence = self._confidence(score, len(votes))

        if confidence < self.confidence_threshold:
            return {"direction": self.UNKNOWN}

        direction = self.APPROACHING if score >= 0 else self.RECEDING
        return {"direction": direction}


class ANPRPipeline:
    """Основной класс распознавания."""

    def __init__(
        self,
        recognizer: CRNNRecognizer,
        best_shots: int,
        cooldown_seconds: int = 0,
        min_confidence: float = 0.6,
        postprocessor: Optional[PlatePostProcessor] = None,
        direction_config: Optional[Dict[str, float | int]] = None,
        segmentation_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.recognizer = recognizer
        self.aggregator = TrackAggregator(best_shots)
        self.cooldown_seconds = max(0, cooldown_seconds)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self._last_seen: Dict[str, float] = {}
        self.postprocessor = postprocessor
        self.preprocessor = PlatePreprocessor()
        self.direction_estimator = TrackDirectionEstimator.from_config(direction_config or {})
        self.segmentation_config = segmentation_config or {}
        self._segmentation_enabled = bool(self.segmentation_config.get("enabled"))

    def _on_cooldown(self, plate: str) -> bool:
        last_seen = self._last_seen.get(plate)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < self.cooldown_seconds

    def _touch_plate(self, plate: str) -> None:
        self._last_seen[plate] = time.monotonic()

    def _merge_segments(self, segment_results: List[tuple[str, float]]) -> tuple[str, float]:
        if not segment_results:
            return "", 0.0
        texts = [text for text, _ in segment_results if text]
        combined_text = "".join(texts)
        confidences = [max(0.0, float(conf)) for _, conf in segment_results]
        avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return combined_text, avg_confidence

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plate_inputs: List[np.ndarray] = []
        recognition_plan: List[Dict[str, int | bool]] = []

        for idx, detection in enumerate(detections):
            if self.direction_estimator and detection.get("track_id") is not None:
                direction_info = self.direction_estimator.update(int(detection["track_id"]), detection.get("bbox", []))
                detection.update(direction_info)
            else:
                detection.setdefault("direction", TrackDirectionEstimator.UNKNOWN)

            x1, y1, x2, y2 = detection["bbox"]
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                processed_plate = None
                segments: list[np.ndarray] = []
                if self._segmentation_enabled:
                    processed_plate, segments = self.preprocessor.extract_components(roi, self.segmentation_config)
                else:
                    processed_plate = self.preprocessor.preprocess(roi)

                if processed_plate is not None and processed_plate.size > 0:
                    use_segments = self._segmentation_enabled and bool(segments)
                    start_index = len(plate_inputs)
                    if use_segments:
                        plate_inputs.extend(segments)
                        recognition_plan.append(
                            {"detection_idx": idx, "start": start_index, "count": len(segments), "segmented": True}
                        )
                    else:
                        plate_inputs.append(processed_plate)
                        recognition_plan.append(
                            {"detection_idx": idx, "start": start_index, "count": 1, "segmented": False}
                        )

        batch_results = self.recognizer.recognize_batch(plate_inputs)

        for plan in recognition_plan:
            detection_idx = int(plan["detection_idx"])
            detection = detections[detection_idx]
            start = int(plan["start"])
            count = int(plan["count"])
            is_segmented = bool(plan.get("segmented"))
            segment_results = batch_results[start : start + count] if start < len(batch_results) else []

            if is_segmented and segment_results:
                current_text, confidence = self._merge_segments(segment_results)
            elif segment_results:
                current_text, confidence = segment_results[0]
            else:
                current_text, confidence = "", 0.0

            if confidence < self.min_confidence:
                detection["text"] = "Нечитаемо"
                detection["unreadable"] = True
                detection["confidence"] = confidence
                continue

            if "track_id" in detection:
                detection["text"] = self.aggregator.add_result(detection["track_id"], current_text, confidence)
            else:
                detection["text"] = current_text

            detection["confidence"] = confidence

            if self.postprocessor and detection.get("text"):
                processed = self.postprocessor.process(detection["text"])
                detection["original_text"] = detection.get("text")
                detection["country"] = processed.country
                detection["format"] = processed.format_name
                detection["validated"] = processed.is_valid

                if processed.is_valid:
                    detection["text"] = processed.plate
                else:
                    detection["text"] = ""
                    if "track_id" in detection:
                        self.aggregator.reset(int(detection["track_id"]))

            if self.cooldown_seconds > 0 and detection.get("text"):
                if self._on_cooldown(detection["text"]):
                    detection["text"] = ""
                else:
                    self._touch_plate(detection["text"])
        return detections
