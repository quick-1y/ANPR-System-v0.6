#!/usr/bin/env python3
#/anpr/infrastructure/settings_manager.py
import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from anpr.infrastructure.logging_manager import get_logger

DEFAULT_ROI_POINTS = [
    {"x": 500, "y": 300},
    {"x": 1200, "y": 300},
    {"x": 1200, "y": 900},
    {"x": 500, "y": 900},
]


logger = get_logger(__name__)


def normalize_region_config(region: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Приводит ROI к единому формату с точками.

    Поддерживает старые варианты (x/y/width/height) и гарантирует наличие
    валидных точек и единицы измерения.
    """

    if not region:
        return {"unit": "px", "points": [point.copy() for point in DEFAULT_ROI_POINTS]}

    normalized_unit = str(region.get("unit", "px")).lower()
    if normalized_unit not in ("px", "percent"):
        normalized_unit = "px"

    raw_points = region.get("points") or []
    points: list[dict[str, float]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        points.append({"x": float(point.get("x", 0)), "y": float(point.get("y", 0))})

    if points:
        return {"unit": normalized_unit, "points": points}

    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    width = float(region.get("width", 100))
    height = float(region.get("height", 100))
    rect_points = [
        {"x": x, "y": y},
        {"x": x + width, "y": y},
        {"x": x + width, "y": y + height},
        {"x": x, "y": y + height},
    ]
    return {"unit": "percent", "points": rect_points}



class SettingsManager:
    """Управляет конфигурацией приложения и каналами."""

    _file_lock = threading.RLock()

    def __init__(self, path: str = "settings.json") -> None:
        self.path = path
        self._settings_lock = threading.RLock()
        self.settings = self._load()

    def _default(self) -> Dict[str, Any]:
        return {
            "models": self._model_defaults(),
            "ocr": self._ocr_defaults(),
            "detector": self._detector_defaults(),
            "inference": self._inference_defaults(),
            "debug": self._debug_defaults(),
            "grid": "2x2",
            "theme": "dark",
            "channels": [],
            "reconnect": {
                "signal_loss": {
                    "enabled": True,
                    "frame_timeout_seconds": 5,
                    "retry_interval_seconds": 5,
                },
                "periodic": {"enabled": False, "interval_minutes": 60},
            },
            "storage": {
                "db_dir": "data/db",
                "database_file": "anpr.db",
                "screenshots_dir": "data/screenshots",
            },
            "tracking": {
                "best_shots": 3,
                "cooldown_seconds": 5,
                "ocr_min_confidence": 0.6,
                "direction": self._direction_defaults(),
            },
            "plates": {
                "config_dir": "config/countries",
                "enabled_countries": ["RU", "UA", "BY", "KZ"],
            },
            "logging": {
                "level": "INFO",
                "file": "data/app.log",
                "max_bytes": 1048576,
                "backup_count": 5,
            },
            "time": self._time_defaults(),
        }

    def _load(self) -> Dict[str, Any]:
        with self._file_lock:
            if not os.path.exists(self.path):
                defaults = self._default()
                self._write_to_disk(defaults)
                return defaults
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return self._upgrade(data)

    def _upgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обновляет существующие настройки, добавляя недостающие поля."""

        changed = False
        tracking_defaults = data.get("tracking", {})
        reconnect_defaults = self._reconnect_defaults()
        storage_defaults = self._storage_defaults()
        plate_defaults = self._plate_defaults()
        time_defaults = self._time_defaults()
        model_defaults = self._model_defaults()
        ocr_defaults = self._ocr_defaults()
        detector_defaults = self._detector_defaults()
        inference_defaults = self._inference_defaults()
        debug_defaults = self._debug_defaults()

        if not data.get("theme"):
            data["theme"] = "dark"
            changed = True

        direction_defaults = self._direction_defaults()
        direction_settings = tracking_defaults.get("direction")
        if direction_settings is None:
            tracking_defaults["direction"] = direction_defaults
            data["tracking"] = tracking_defaults
            changed = True
        else:
            for key, value in direction_defaults.items():
                if key not in direction_settings:
                    direction_settings[key] = value
                    changed = True

        for channel in data.get("channels", []):
            if self._fill_channel_defaults(channel, tracking_defaults):
                changed = True

        if self._fill_reconnect_defaults(data, reconnect_defaults):
            changed = True

        if self._fill_model_defaults(data, model_defaults):
            changed = True

        if self._fill_ocr_defaults(data, ocr_defaults):
            changed = True

        if self._fill_detector_defaults(data, detector_defaults):
            changed = True

        if self._fill_inference_defaults(data, inference_defaults):
            changed = True

        if self._fill_storage_defaults(data, storage_defaults):
            changed = True

        if self._fill_plate_defaults(data, plate_defaults):
            changed = True

        if self._fill_time_defaults(data, time_defaults):
            changed = True

        if self._fill_debug_defaults(data, debug_defaults):
            changed = True

        if changed:
            self._save(data)
        return data

    @staticmethod
    def _channel_defaults(tracking_defaults: Dict[str, Any]) -> Dict[str, Any]:
        size_defaults = plate_size_defaults()
        return {
            "best_shots": int(tracking_defaults.get("best_shots", 3)),
            "cooldown_seconds": int(tracking_defaults.get("cooldown_seconds", 5)),
            "ocr_min_confidence": float(tracking_defaults.get("ocr_min_confidence", 0.6)),
            "direction": dict(tracking_defaults.get("direction", direction_defaults())),
            "roi_enabled": True,
            "region": {"unit": "px", "points": [point.copy() for point in DEFAULT_ROI_POINTS]},
            "detection_mode": "motion",
            "detector_frame_stride": 2,
            "motion_threshold": 0.01,
            "motion_frame_stride": 1,
            "motion_activation_frames": 3,
            "motion_release_frames": 6,
            "size_filter_enabled": True,
            "min_plate_size": size_defaults["min_plate_size"].copy(),
            "max_plate_size": size_defaults["max_plate_size"].copy(),
        }

    @staticmethod
    def _debug_defaults() -> Dict[str, Any]:
        return {
            "show_detection_boxes": False,
            "show_ocr_text": False,
            "show_direction_tracks": False,
            "log_panel_enabled": False,
        }

    @staticmethod
    def _upgrade_region(region: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return normalize_region_config(region)

    @staticmethod
    def _reconnect_defaults() -> Dict[str, Any]:
        return {
            "signal_loss": {
                "enabled": True,
                "frame_timeout_seconds": 5,
                "retry_interval_seconds": 5,
            },
            "periodic": {"enabled": False, "interval_minutes": 60},
        }

    @staticmethod
    def _storage_defaults() -> Dict[str, Any]:
        return {
            "db_dir": "data/db",
            "database_file": "anpr.db",
            "screenshots_dir": "data/screenshots",
        }

    @staticmethod
    def _plate_defaults() -> Dict[str, Any]:
        return {
            "config_dir": "config/countries",
            "enabled_countries": ["RU", "UA", "BY", "KZ"],
        }

    @staticmethod
    def _model_defaults() -> Dict[str, Any]:
        return {
            "yolo_model_path": "models/yolo/best.pt",
            "ocr_model_path": "models/ocr_crnn/crnn_ocr_model_int8_fx.pth",
            "device": "cpu",
        }

    @staticmethod
    def _inference_defaults() -> Dict[str, Any]:
        cpu_count = os.cpu_count() or 1
        return {
            "workers": max(1, cpu_count - 1),
            "shared_memory": True,
        }

    @staticmethod
    def _plate_size_defaults() -> Dict[str, Dict[str, int]]:
        return {
            "min_plate_size": {"width": 80, "height": 20},
            "max_plate_size": {"width": 600, "height": 240},
        }

    @staticmethod
    def _direction_defaults() -> Dict[str, float | int]:
        return {
            "history_size": 12,
            "min_track_length": 3,
            "smoothing_window": 5,
            "confidence_threshold": 0.55,
            "jitter_pixels": 1.0,
            "min_area_change_ratio": 0.02,
        }

    @staticmethod
    def _ocr_defaults() -> Dict[str, Any]:
        return {
            "img_height": 32,
            "img_width": 128,
            "alphabet": "0123456789ABCEHKMOPTXY",
            "confidence_threshold": 0.6,
        }

    @staticmethod
    def _detector_defaults() -> Dict[str, Any]:
        return {
            "confidence_threshold": 0.5,
            "bbox_padding_ratio": 0.08,
            "min_padding_pixels": 2,
        }

    @staticmethod
    def _time_defaults() -> Dict[str, Any]:
        now = datetime.now().astimezone()
        offset = now.utcoffset() or timedelta()
        minutes = int(offset.total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        total = abs(minutes)
        hours = total // 60
        mins = total % 60
        default_zone = f"UTC{sign}{hours:02d}:{mins:02d}"
        return {"timezone": default_zone, "offset_minutes": 0}

    def _fill_channel_defaults(self, channel: Dict[str, Any], tracking_defaults: Dict[str, Any]) -> bool:
        defaults = self._channel_defaults(tracking_defaults)
        changed = False
        for key, value in defaults.items():
            if key not in channel:
                # Сохраняем только отсутствующие ключи, не перезаписывая пользовательские значения.
                channel[key] = value
                changed = True
        if "debug" in channel:
            channel.pop("debug", None)
            changed = True

        direction_defaults = defaults.get("direction", self._direction_defaults())
        channel_direction = channel.get("direction")
        if channel_direction is None:
            channel["direction"] = dict(direction_defaults)
            changed = True
        else:
            for key, value in direction_defaults.items():
                if key not in channel_direction:
                    channel_direction[key] = value
                    changed = True

        upgraded_region = self._upgrade_region(channel.get("region"))
        if channel.get("region") != upgraded_region:
            channel["region"] = upgraded_region
            changed = True
        return changed

    def _fill_reconnect_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "reconnect" not in data:
            data["reconnect"] = defaults
            return True

        changed = False
        reconnect_section = data.get("reconnect", {})
        for key, default_value in defaults.items():
            if key not in reconnect_section:
                reconnect_section[key] = default_value
                changed = True
            elif isinstance(default_value, dict):
                for sub_key, sub_val in default_value.items():
                    if sub_key not in reconnect_section[key]:
                        reconnect_section[key][sub_key] = sub_val
                        changed = True
        data["reconnect"] = reconnect_section
        return changed

    def _fill_debug_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "debug" not in data:
            data["debug"] = defaults
            return True

        changed = False
        debug_section = data.get("debug", {})
        for key, value in defaults.items():
            if key not in debug_section:
                debug_section[key] = value
                changed = True
        data["debug"] = debug_section
        return changed

    def _fill_storage_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "storage" not in data:
            data["storage"] = defaults
            return True

        changed = False
        storage = data.get("storage", {})

        # Миграция со старого ключа events_db -> db_dir + database_file
        if "events_db" in storage:
            legacy_path = storage.get("events_db", "data/events.db")
            storage.setdefault("db_dir", os.path.dirname(legacy_path) or ".")
            storage.setdefault("database_file", os.path.basename(legacy_path) or "anpr.db")
            changed = True

        for key, val in defaults.items():
            if key not in storage:
                storage[key] = val
                changed = True
        data["storage"] = storage
        return changed

    def _fill_plate_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "plates" not in data:
            data["plates"] = defaults
            return True

        changed = False
        plates = data.get("plates", {})
        for key, val in defaults.items():
            if key not in plates:
                plates[key] = val
                changed = True
        data["plates"] = plates
        return changed

    def _fill_model_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "models" not in data:
            data["models"] = defaults
            return True

        changed = False
        models = data.get("models", {})
        for key, val in defaults.items():
            if key not in models:
                models[key] = val
                changed = True
        data["models"] = models
        return changed

    def _fill_ocr_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "ocr" not in data:
            data["ocr"] = defaults
            return True

        changed = False
        ocr = data.get("ocr", {})
        for key, val in defaults.items():
            if key not in ocr:
                ocr[key] = val
                changed = True
        data["ocr"] = ocr
        return changed

    def _fill_detector_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "detector" not in data:
            data["detector"] = defaults
            return True

        changed = False
        detector = data.get("detector", {})
        for key, val in defaults.items():
            if key not in detector:
                detector[key] = val
                changed = True
        data["detector"] = detector
        return changed

    def _fill_inference_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "inference" not in data:
            data["inference"] = defaults
            return True

        changed = False
        inference = data.get("inference", {})
        for key, val in defaults.items():
            if key not in inference:
                inference[key] = val
                changed = True
        data["inference"] = inference
        return changed

    def _fill_time_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "time" not in data:
            data["time"] = defaults
            return True

        changed = False
        time_section = data.get("time", {})
        for key, val in defaults.items():
            if key not in time_section:
                time_section[key] = val
                changed = True
        data["time"] = time_section
        return changed

    def _save(self, data: Dict[str, Any]) -> None:
        logger.debug(f"Сохранение настроек из потока: {threading.current_thread().name}")
        with self._file_lock:
            snapshot = copy.deepcopy(data)
        self._write_to_disk(snapshot)

    def _write_to_disk(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._file_lock:
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.path) or ".", prefix=".settings_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def get_channels(self) -> List[Dict[str, Any]]:
        with self._file_lock:
            channels = self.settings.get("channels", [])
            tracking_defaults = self.settings.get("tracking", {})
        changed = False
        max_id = 0
        for channel in channels:
            try:
                channel_id = int(channel.get("id", 0))
            except (TypeError, ValueError):
                channel_id = 0
            max_id = max(max_id, channel_id)

        for channel in channels:
            try:
                channel_id = int(channel.get("id", 0))
            except (TypeError, ValueError):
                channel_id = 0
            if channel_id <= 0:
                max_id += 1
                channel["id"] = max_id
                changed = True
            if self._fill_channel_defaults(channel, tracking_defaults):
                changed = True

        if changed:
            self.save_channels(channels)
        return copy.deepcopy(channels)

    def save_channels(self, channels: List[Dict[str, Any]]) -> None:
        with self._file_lock:
            self.settings["channels"] = copy.deepcopy(channels)
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_grid(self) -> str:
        with self._file_lock:
            return self.settings.get("grid", "2x2")

    def save_grid(self, grid: str) -> None:
        with self._file_lock:
            self.settings["grid"] = grid
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_theme(self) -> str:
        with self._file_lock:
            return self.settings.get("theme", "dark")

    def save_theme(self, theme: str) -> None:
        with self._file_lock:
            self.settings["theme"] = theme
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_reconnect(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_reconnect_defaults(self.settings, self._reconnect_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("reconnect", {}))

    def save_reconnect(self, reconnect_conf: Dict[str, Any]) -> None:
        with self._file_lock:
            self.settings["reconnect"] = reconnect_conf
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_db_dir(self) -> str:
        with self._file_lock:
            storage = self.settings.get("storage", {})
            return storage.get("db_dir", "data/db")

    def get_database_file(self) -> str:
        with self._file_lock:
            storage = self.settings.get("storage", {})
            return storage.get("database_file", "anpr.db")

    def get_db_path(self) -> str:
        directory = self.get_db_dir()
        filename = self.get_database_file()
        return os.path.join(directory, filename)

    def save_db_dir(self, path: str) -> None:
        with self._file_lock:
            storage = self.settings.get("storage", {})
            storage["db_dir"] = path
            self.settings["storage"] = storage
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def save_screenshot_dir(self, path: str) -> None:
        with self._file_lock:
            storage = self.settings.get("storage", {})
            storage["screenshots_dir"] = path
            self.settings["storage"] = storage
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_screenshot_dir(self) -> str:
        with self._file_lock:
            storage = self.settings.get("storage", {})
            return storage.get("screenshots_dir", "data/screenshots")

    def get_time_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_time_defaults(self.settings, self._time_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("time", {}))

    def save_time_settings(self, time_settings: Dict[str, Any]) -> None:
        with self._file_lock:
            self.settings["time"] = time_settings
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_timezone(self) -> str:
        time_settings = self.get_time_settings()
        return str(time_settings.get("timezone") or "UTC")

    def get_time_offset_minutes(self) -> int:
        time_settings = self.get_time_settings()
        try:
            return int(time_settings.get("offset_minutes", 0))
        except (TypeError, ValueError):
            return 0

    def get_best_shots(self) -> int:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            return int(tracking.get("best_shots", 3))

    def save_best_shots(self, best_shots: int) -> None:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            tracking["best_shots"] = int(best_shots)
            self.settings["tracking"] = tracking
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_cooldown_seconds(self) -> int:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            return int(tracking.get("cooldown_seconds", 5))

    def save_cooldown_seconds(self, cooldown: int) -> None:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            tracking["cooldown_seconds"] = int(cooldown)
            self.settings["tracking"] = tracking
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_min_confidence(self) -> float:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            return float(tracking.get("ocr_min_confidence", 0.6))

    def save_min_confidence(self, min_conf: float) -> None:
        with self._file_lock:
            tracking = self.settings.get("tracking", {})
            tracking["ocr_min_confidence"] = float(min_conf)
            self.settings["tracking"] = tracking
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_plate_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_plate_defaults(self.settings, self._plate_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("plates", {}))

    def save_plate_settings(self, plate_settings: Dict[str, Any]) -> None:
        with self._file_lock:
            self.settings["plates"] = plate_settings
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_logging_config(self) -> Dict[str, Any]:
        with self._file_lock:
            return copy.deepcopy(self.settings.get("logging", {}))

    def get_debug_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_debug_defaults(self.settings, self._debug_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("debug", {}))

    def save_debug_settings(self, debug_settings: Dict[str, Any]) -> None:
        with self._file_lock:
            self.settings["debug"] = debug_settings
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_model_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_model_defaults(self.settings, self._model_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("models", {}))

    def save_model_device(self, device: str) -> None:
        with self._file_lock:
            models = self.settings.get("models", {})
            models["device"] = device
            self.settings["models"] = models
            settings_snapshot = copy.deepcopy(self.settings)
        self._save(settings_snapshot)

    def get_ocr_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_ocr_defaults(self.settings, self._ocr_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("ocr", {}))

    def get_detector_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_detector_defaults(self.settings, self._detector_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("detector", {}))

    def get_inference_settings(self) -> Dict[str, Any]:
        with self._file_lock:
            if self._fill_inference_defaults(self.settings, self._inference_defaults()):
                settings_snapshot = copy.deepcopy(self.settings)
                self._save(settings_snapshot)
            return copy.deepcopy(self.settings.get("inference", {}))

    def get_plate_size_defaults(self) -> Dict[str, Dict[str, int]]:
        return plate_size_defaults()

    def get_direction_defaults(self) -> Dict[str, float | int]:
        return direction_defaults()

    def refresh(self) -> None:
        with self._file_lock:
            self.settings = self._load()

    def update_channel(self, channel_id: int, data: Dict[str, Any]) -> None:
        channels = self.get_channels()
        for idx, channel in enumerate(channels):
            if channel.get("id") == channel_id:
                channels[idx].update(data)
                break
        else:
            channels.append(data)
        self.save_channels(channels)


def plate_size_defaults() -> Dict[str, Dict[str, int]]:
    """Единый источник дефолтов размеров рамки номера."""
    defaults = SettingsManager._plate_size_defaults()
    return {key: value.copy() for key, value in defaults.items()}


def direction_defaults() -> Dict[str, float | int]:
    """Единый источник дефолтов определения направления движения."""
    defaults = SettingsManager._direction_defaults()
    return dict(defaults)
