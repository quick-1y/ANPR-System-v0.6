from anpr.infrastructure.settings_manager import SettingsManager, direction_defaults, plate_size_defaults
from anpr.workers.channel_worker import DirectionSettings, PlateSize


def test_direction_settings_follow_exported_defaults(monkeypatch):
    custom_defaults = {
        "history_size": 20,
        "min_track_length": 5,
        "smoothing_window": 7,
        "confidence_threshold": 0.75,
        "jitter_pixels": 2.0,
        "min_area_change_ratio": 0.05,
    }
    monkeypatch.setattr(SettingsManager, "_direction_defaults", staticmethod(lambda: custom_defaults))

    direction = DirectionSettings.from_dict(None)

    assert direction.to_dict() == custom_defaults
    assert direction_defaults() == custom_defaults


def test_plate_size_defaults_forwarded(monkeypatch):
    custom_defaults = {
        "min_plate_size": {"width": 10, "height": 5},
        "max_plate_size": {"width": 20, "height": 15},
    }
    monkeypatch.setattr(SettingsManager, "_plate_size_defaults", staticmethod(lambda: custom_defaults))

    min_size = PlateSize.from_dict(None, default_label="min_plate_size")
    max_size = PlateSize.from_dict(None, default_label="max_plate_size")

    assert min_size.to_dict() == custom_defaults["min_plate_size"]
    assert max_size.to_dict() == custom_defaults["max_plate_size"]
    assert plate_size_defaults() == custom_defaults
