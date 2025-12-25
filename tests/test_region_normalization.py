import copy

from anpr.infrastructure.settings_manager import DEFAULT_ROI_POINTS, normalize_region_config
from anpr.workers.channel_worker import Region


def test_normalize_region_round_trip_preserves_points():
    raw_region = {
        "unit": "percent",
        "points": [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 50, "y": 50}, {"x": 0, "y": 50}],
    }
    normalized = normalize_region_config(copy.deepcopy(raw_region))
    region = Region.from_dict(raw_region)

    assert region.unit == normalized["unit"]
    assert region.to_dict()["points"] == normalized["points"]


def test_normalize_legacy_rect_round_trip():
    raw_region = {"x": 10, "y": 20, "width": 30, "height": 40}
    normalized = normalize_region_config(raw_region)
    region = Region.from_dict(raw_region)

    assert normalized["unit"] == "percent"
    assert region.unit == normalized["unit"]
    assert region.to_dict()["points"] == normalized["points"]


def test_default_region_is_consistent():
    normalized_default = normalize_region_config(None)
    region = Region.from_dict(None)

    assert normalized_default["points"] == DEFAULT_ROI_POINTS
    assert region.to_dict()["points"] == normalized_default["points"]
    assert region.unit == normalized_default["unit"]
