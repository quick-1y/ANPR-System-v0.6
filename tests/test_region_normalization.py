import copy
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2_stub = types.SimpleNamespace(
    Mat=object,
    FONT_HERSHEY_SIMPLEX=0,
    setNumThreads=lambda *args, **kwargs: None,
    rectangle=lambda *args, **kwargs: None,
    putText=lambda *args, **kwargs: None,
    getTextSize=lambda text, font, scale, thickness: ((0, 0), None),
    VideoCapture=lambda *args, **kwargs: types.SimpleNamespace(isOpened=lambda: False),
)
sys.modules.setdefault("cv2", cv2_stub)

class _DummySignal:
    def __init__(self, *args, **kwargs) -> None:
        self._last = None

    def emit(self, *args, **kwargs) -> None:
        self._last = (args, kwargs)


class _DummyQThread:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def isInterruptionRequested(self) -> bool:
        return False

    def requestInterruption(self) -> None:
        pass


qtcore_stub = types.SimpleNamespace(QThread=_DummyQThread, pyqtSignal=lambda *args, **kwargs: _DummySignal())
qtgui_stub = types.SimpleNamespace(QImage=object)
pyqt5_stub = types.ModuleType("PyQt5")
pyqt5_stub.QtCore = qtcore_stub
pyqt5_stub.QtGui = qtgui_stub

sys.modules.setdefault("PyQt5", pyqt5_stub)
sys.modules.setdefault("PyQt5.QtCore", qtcore_stub)
sys.modules.setdefault("PyQt5.QtGui", qtgui_stub)


class _DummyYOLO:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def to(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        boxes = types.SimpleNamespace(data=[])
        return [types.SimpleNamespace(boxes=boxes)]

    def track(self, *args, **kwargs):
        boxes = types.SimpleNamespace(
            data=[], id=None, xyxy=types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: [])),
            conf=types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: [])),
        )
        return [types.SimpleNamespace(boxes=boxes)]


ultralytics_stub = types.SimpleNamespace(YOLO=_DummyYOLO)
sys.modules.setdefault("ultralytics", ultralytics_stub)

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
