#!/usr/bin/env python3
# /anpr/ui/main_window.py
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import psutil
import torch
from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
from zoneinfo import ZoneInfo

from anpr.config import Config
from anpr.infrastructure.settings_manager import DEFAULT_ROI_POINTS
from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.workers.channel_worker import ChannelWorker
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.storage import EventDatabase

logger = get_logger(__name__)


class LogSignalEmitter(QtCore.QObject):
    """Безопасная доставка строк логов в UI-поток."""

    message = QtCore.pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Qt-хендлер для проброса логов в окно наблюдения."""

    def __init__(self, emitter: LogSignalEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            return
        self._emitter.message.emit(msg)


class PixmapPool:
    """Простой пул QPixmap для повторного использования буферов по размеру."""

    def __init__(self, max_per_size: int = 5) -> None:
        self._pool: Dict[Tuple[int, int], List[QtGui.QPixmap]] = {}
        self._max_per_size = max_per_size

    def acquire(self, size: QtCore.QSize) -> QtGui.QPixmap:
        key = (size.width(), size.height())
        pixmaps = self._pool.get(key)
        if pixmaps:
            pixmap = pixmaps.pop()
        else:
            pixmap = QtGui.QPixmap(size)
        if pixmap.size() != size:
            pixmap = QtGui.QPixmap(size)
        return pixmap

    def release(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap.isNull():
            return

        key = (pixmap.width(), pixmap.height())
        pixmaps = self._pool.setdefault(key, [])

        if len(pixmaps) >= self._max_per_size:
            old_pixmap = pixmaps.pop(0)
            old_pixmap.detach()

        pixmaps.append(pixmap)


class ChannelView(QtWidgets.QWidget):
    """Отображает поток канала с подсказками и индикатором движения."""

    channelDropped = QtCore.pyqtSignal(int, int)
    channelActivated = QtCore.pyqtSignal(str)
    dragStarted = QtCore.pyqtSignal()
    dragFinished = QtCore.pyqtSignal()

    def __init__(self, name: str, pixmap_pool: Optional[PixmapPool], colors: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.name = name
        self._pixmap_pool = pixmap_pool
        self._current_pixmap: Optional[QtGui.QPixmap] = None
        self._channel_name: Optional[str] = None
        self._grid_position: int = -1
        self._drag_start_pos: Optional[QtCore.QPoint] = None
        self.setAcceptDrops(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._colors: Dict[str, str] = colors or {}
        self.video_label = QtWidgets.QLabel("Нет сигнала")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(220, 170)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.motion_indicator = QtWidgets.QLabel("Движение")
        self.motion_indicator.setParent(self.video_label)
        self.motion_indicator.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.motion_indicator.hide()

        self.last_plate = QtWidgets.QLabel("—")
        self.last_plate.setParent(self.video_label)
        self.last_plate.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.last_plate.hide()

        self.status_hint = QtWidgets.QLabel("")
        self.status_hint.setParent(self.video_label)
        self.status_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.status_hint.hide()

        self.metrics_hint = QtWidgets.QLabel("")
        self.metrics_hint.setParent(self.video_label)
        self.metrics_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.metrics_hint.hide()
        self.set_theme(self._colors or MainWindow.THEME_PALETTES["dark"])

    def set_channel_name(self, channel_name: Optional[str]) -> None:
        self._channel_name = channel_name
        if channel_name is None:
            self.clear()

    def channel_name(self) -> Optional[str]:
        return self._channel_name

    def set_grid_position(self, position: int) -> None:
        self._grid_position = position

    def set_theme(self, colors: Dict[str, str]) -> None:
        self._colors = colors
        video_bg = colors.get("background", "#000000")
        text = colors.get("text_secondary", "#cccccc")
        overlay_bg = colors.get("overlay_bg", "rgba(0,0,0,0.55)")
        accent = colors.get("accent", "#22d3ee")
        self.video_label.setStyleSheet(
            f"background-color: {video_bg}; color: {text}; border: 1px solid {colors.get('border', '#2e2e2e')}; padding: 4px;"
        )
        self.motion_indicator.setStyleSheet(
            "background-color: rgba(220, 53, 69, 0.85); color: white;"
            "padding: 3px 6px; border-radius: 6px; font-weight: bold;"
        )
        self.last_plate.setStyleSheet(
            f"background-color: {overlay_bg}; color: {colors.get('text_primary', '#ffffff')};"
            "padding: 2px 6px; border-radius: 4px; font-weight: bold;"
        )
        self.status_hint.setStyleSheet(
            f"background-color: {overlay_bg}; color: {text}; padding: 2px 4px;"
        )
        self.metrics_hint.setStyleSheet(
            f"background-color: {overlay_bg}; color: {accent}; padding: 2px 4px;"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_overlay_positions()

    def _update_overlay_positions(self) -> None:
        rect = self.video_label.contentsRect()
        margin = 8
        indicator_size = self.motion_indicator.sizeHint()
        self.motion_indicator.move(
            rect.right() - indicator_size.width() - margin, rect.top() + margin
        )
        self.last_plate.move(rect.left() + margin, rect.top() + margin)
        status_size = self.status_hint.sizeHint()
        self.status_hint.move(rect.left() + margin, rect.bottom() - status_size.height() - margin)
        metrics_size = self.metrics_hint.sizeHint()
        self.metrics_hint.move(
            rect.right() - metrics_size.width() - margin,
            rect.bottom() - metrics_size.height() - margin,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if (
            self._channel_name
            and event.buttons() & QtCore.Qt.LeftButton
            and self._drag_start_pos
            and (event.pos() - self._drag_start_pos).manhattanLength()
            >= QtWidgets.QApplication.startDragDistance()
        ):
            self.dragStarted.emit()
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setText(self._channel_name)
            mime_data.setData("application/x-channel-index", str(self._grid_position).encode())
            drag.setMimeData(mime_data)
            drag.exec_(QtCore.Qt.MoveAction)
            self.dragFinished.emit()
            return
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat("application/x-channel-index"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat("application/x-channel-index"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        if not event.mimeData().hasFormat("application/x-channel-index"):
            event.ignore()
            return
        try:
            source_index = int(bytes(event.mimeData().data("application/x-channel-index")).decode())
        except (ValueError, TypeError):
            event.ignore()
            return
        self.channelDropped.emit(source_index, self._grid_position)
        event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self._channel_name:
            self.channelActivated.emit(self._channel_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if self._pixmap_pool and self._current_pixmap is not None:
            self._pixmap_pool.release(self._current_pixmap)
        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)

    def clear(self) -> None:
        if self._pixmap_pool and self._current_pixmap is not None:
            self._pixmap_pool.release(self._current_pixmap)
        self._current_pixmap = None
        self.video_label.clear()
        self.video_label.setText("Нет сигнала")
        self.motion_indicator.hide()
        self.last_plate.hide()
        self.status_hint.hide()
        self.metrics_hint.hide()

    def set_motion_active(self, active: bool) -> None:
        self.motion_indicator.setVisible(active)

    def set_last_plate(self, plate: str) -> None:
        self.last_plate.setVisible(bool(plate))
        self.last_plate.setText(plate or "—")
        self.last_plate.adjustSize()
        self._update_overlay_positions()

    def set_status(self, text: str) -> None:
        self.status_hint.setVisible(bool(text))
        self.status_hint.setText(text)
        if text:
            self.status_hint.adjustSize()
        self._update_overlay_positions()

    def set_metrics(self, text: str) -> None:
        self.metrics_hint.setVisible(bool(text))
        self.metrics_hint.setText(text)
        if text:
            self.metrics_hint.adjustSize()
        self._update_overlay_positions()


class ROIEditor(QtWidgets.QLabel):
    """Виджет предпросмотра канала с настраиваемой областью распознавания."""

    roi_changed = QtCore.pyqtSignal(dict)
    plate_size_selected = QtCore.pyqtSignal(str, int, int)

    def __init__(self) -> None:
        super().__init__("Нет кадра")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 260)
        self.set_theme(MainWindow.THEME_PALETTES["dark"])
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._roi_data: Dict[str, Any] = {"unit": "px", "points": []}
        self._points: List[QtCore.QPointF] = []
        self._drag_index: Optional[int] = None
        self._size_rects: Dict[str, Optional[QtCore.QRectF]] = {"min": None, "max": None}
        self._active_size_target: Optional[str] = None
        self._active_size_handle: Optional[str] = None
        self._active_size_origin: Optional[QtCore.QPointF] = None
        self._active_size_rect: Optional[QtCore.QRectF] = None
        self._size_capture_target: Optional[str] = None
        self._size_capture_start: Optional[QtCore.QPointF] = None
        self._size_capture_end: Optional[QtCore.QPointF] = None
        self._roi_usage_enabled = True
        self._size_overlay_enabled = True

    def set_theme(self, colors: Dict[str, str]) -> None:
        self.setStyleSheet(
            f"background-color: {colors['field_bg']}; color: {colors['text_muted']}; border: 1px solid {colors['border']}; padding: 6px;"
        )

    def image_size(self) -> Optional[QtCore.QSize]:
        return self._pixmap.size() if self._pixmap else None

    def current_pixmap(self) -> Optional[QtGui.QPixmap]:
        return self._pixmap

    def _clamp_points(self) -> None:
        if not self._pixmap:
            return
        width = self._pixmap.width()
        height = self._pixmap.height()
        clamped: List[QtCore.QPointF] = []
        for p in self._points:
            clamped.append(
                QtCore.QPointF(
                    max(0.0, min(float(width - 1), p.x())),
                    max(0.0, min(float(height - 1), p.y())),
                )
            )
        self._points = clamped

    def _recalculate_points(self) -> None:
        roi = self._roi_data or {}
        unit = str(roi.get("unit", "px")).lower()
        raw_points = roi.get("points") or []
        if not raw_points:
            self._points = []
            return

        if self._pixmap is None:
            self._points = [
                QtCore.QPointF(float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
            return

        width = self._pixmap.width()
        height = self._pixmap.height()
        if unit == "percent":
            self._points = [
                QtCore.QPointF(width * float(p.get("x", 0)) / 100.0, height * float(p.get("y", 0)) / 100.0)
                for p in raw_points
                if isinstance(p, dict)
            ]
        else:
            self._points = [
                QtCore.QPointF(float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
        self._clamp_points()

    def _clamp_rect(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        if self._pixmap is None:
            return QtCore.QRectF(rect)
        width = max(1.0, float(self._pixmap.width()))
        height = max(1.0, float(self._pixmap.height()))
        left = max(0.0, min(rect.left(), width))
        top = max(0.0, min(rect.top(), height))
        right = max(left + 1.0, min(rect.right(), width))
        bottom = max(top + 1.0, min(rect.bottom(), height))
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))

    def set_roi(self, roi: Dict[str, Any]) -> None:
        self._roi_data = roi or {"unit": "px", "points": []}
        self._recalculate_points()
        self.update()

    def set_plate_sizes(
        self,
        min_width: int,
        min_height: int,
        max_width: int,
        max_height: int,
    ) -> None:
        self._update_size_rect("min", float(min_width), float(min_height))
        self._update_size_rect("max", float(max_width), float(max_height))
        self.update()

    def set_roi_usage_enabled(self, enabled: bool) -> None:
        self._roi_usage_enabled = bool(enabled)
        self.update()

    def set_size_overlay_enabled(self, enabled: bool) -> None:
        self._size_overlay_enabled = bool(enabled)
        if not enabled:
            self._active_size_target = None
            self._active_size_handle = None
            self._active_size_origin = None
            self._active_size_rect = None
        self.update()

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:  # noqa: N802
        self._pixmap = pixmap
        if pixmap is None:
            super().setPixmap(QtGui.QPixmap())
            self.setText("Нет кадра")
            self._size_capture_target = None
            self._size_capture_start = None
            self._size_capture_end = None
            return
        self._recalculate_points()
        self._clamp_size_rects()
        scaled = self._scaled_pixmap(self.size())
        super().setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap(event.size()))

    def _update_size_rect(self, target: str, width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            self._size_rects[target] = None
            return

        if self._pixmap is None:
            left = 0.0
            top = 0.0
        else:
            existing = self._size_rects.get(target)
            anchor = existing.center() if existing else QtCore.QPointF(
                float(self._pixmap.width()) / 2.0,
                float(self._pixmap.height()) / 2.0,
            )
            left = anchor.x() - width / 2.0
            top = anchor.y() - height / 2.0
        rect = QtCore.QRectF(left, top, width, height)
        self._size_rects[target] = self._clamp_rect(rect)

    def _clamp_size_rects(self) -> None:
        for key, rect in self._size_rects.items():
            if rect is not None:
                if (
                    self._pixmap is not None
                    and rect.topLeft() == QtCore.QPointF(0.0, 0.0)
                    and rect.width() > 0
                    and rect.height() > 0
                ):
                    centered = QtCore.QPointF(
                        float(self._pixmap.width()) / 2.0,
                        float(self._pixmap.height()) / 2.0,
                    )
                    rect = QtCore.QRectF(
                        centered.x() - rect.width() / 2.0,
                        centered.y() - rect.height() / 2.0,
                        rect.width(),
                        rect.height(),
                    )
                self._size_rects[key] = self._clamp_rect(rect)

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self._pixmap is not None
        return self._pixmap.scaled(
            size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def _image_geometry(self) -> Optional[Tuple[QtCore.QPoint, QtCore.QSize]]:
        if self._pixmap is None:
            return None
        pixmap = self._scaled_pixmap(self.size())
        area = self.contentsRect()
        x = area.x() + (area.width() - pixmap.width()) // 2
        y = area.y() + (area.height() - pixmap.height()) // 2
        return QtCore.QPoint(x, y), pixmap.size()

    def _image_to_widget(self, point: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        scale_x = scaled_size.width() / max(1, self._pixmap.width())
        scale_y = scaled_size.height() / max(1, self._pixmap.height())
        return QtCore.QPointF(
            offset.x() + point.x() * scale_x,
            offset.y() + point.y() * scale_y,
        )

    def _widget_to_image(self, point: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        rect = QtCore.QRect(offset, scaled_size)
        if not rect.contains(point):
            return None
        scale_x = max(1, self._pixmap.width()) / max(1, scaled_size.width())
        scale_y = max(1, self._pixmap.height()) / max(1, scaled_size.height())
        return QtCore.QPointF(
            (point.x() - offset.x()) * scale_x,
            (point.y() - offset.y()) * scale_y,
        )

    def _widget_to_image_clamped(self, point: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        img_pos = self._widget_to_image(point)
        if img_pos is not None:
            return img_pos
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        rect = QtCore.QRect(offset, scaled_size)
        clamped = QtCore.QPoint(
            max(rect.left(), min(point.x(), rect.right())),
            max(rect.top(), min(point.y(), rect.bottom())),
        )
        scale_x = max(1, self._pixmap.width()) / max(1, scaled_size.width())
        scale_y = max(1, self._pixmap.height()) / max(1, scaled_size.height())
        return QtCore.QPointF(
            (clamped.x() - offset.x()) * scale_x,
            (clamped.y() - offset.y()) * scale_y,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        geom = self._image_geometry()
        if geom is None:
            return
        _, scaled_size = geom
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # ROI отрисовываем поверх, чтобы рамка выбора не закрывала границы полигона

        polygon_points = self._points
        if not polygon_points and self._pixmap:
            polygon_points = [
                QtCore.QPointF(0, 0),
                QtCore.QPointF(self._pixmap.width() - 1, 0),
                QtCore.QPointF(self._pixmap.width() - 1, self._pixmap.height() - 1),
                QtCore.QPointF(0, self._pixmap.height() - 1),
            ]

        if not polygon_points:
            return

        widget_points = [self._image_to_widget(p) for p in polygon_points]
        widget_points = [p for p in widget_points if p is not None]
        if len(widget_points) < 3:
            return

        pen_color = QtGui.QColor(0, 200, 0) if self._roi_usage_enabled else QtGui.QColor(170, 170, 170)
        fill_color = QtGui.QColor(0, 200, 0, 40) if self._roi_usage_enabled else QtGui.QColor(170, 170, 170, 30)
        pen = QtGui.QPen(pen_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(fill_color)
        painter.drawPolygon(QtGui.QPolygonF(widget_points))

        handle_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        painter.setBrush(handle_brush)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        for p in widget_points:
            painter.drawEllipse(QtCore.QPointF(p), 5, 5)

        if not self._roi_usage_enabled:
            painter.setPen(QtGui.QPen(QtGui.QColor(210, 210, 210)))
            painter.drawText(
                self.rect().adjusted(10, 10, -10, -10),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                "ROI отключена — поиск по всему кадру",
            )

        if self._size_overlay_enabled:
            for target, color in ("min", QtGui.QColor(34, 211, 238)), ("max", QtGui.QColor(249, 115, 22)):
                rect = self._size_rects.get(target)
                if rect is None:
                    continue
                top_left = self._image_to_widget(rect.topLeft())
                bottom_right = self._image_to_widget(rect.bottomRight())
                if top_left is None or bottom_right is None:
                    continue
                widget_rect = QtCore.QRectF(top_left, bottom_right).normalized()
                pen = QtGui.QPen(color)
                pen.setWidth(2)
                pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 40))
                painter.drawRect(widget_rect)

                painter.setPen(QtGui.QPen(color))
                painter.setBrush(QtGui.QBrush(color))
                for corner in (
                    widget_rect.topLeft(),
                    widget_rect.topRight(),
                    widget_rect.bottomLeft(),
                    widget_rect.bottomRight(),
                ):
                    painter.drawEllipse(corner, 5, 5)

                label = "минимальный" if target == "min" else "максимальный"
                metrics = painter.fontMetrics()
                text_rect = metrics.boundingRect(label)
                label_rect = QtCore.QRectF(
                    widget_rect.left(),
                    max(0.0, widget_rect.top() - text_rect.height() - 4),
                    text_rect.width(),
                    text_rect.height(),
                )
                painter.setPen(QtGui.QPen(color))
                painter.drawText(label_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)

    def _emit_roi(self) -> None:
        roi = {
            "unit": "px",
            "points": [
                {"x": int(p.x()), "y": int(p.y())}
                for p in self._points
            ],
        }
        self.roi_changed.emit(roi)

    @staticmethod
    def _point_to_segment_distance(point: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF) -> float:
        ab_x = b.x() - a.x()
        ab_y = b.y() - a.y()
        ab_len_sq = ab_x * ab_x + ab_y * ab_y
        if ab_len_sq == 0:
            return math.hypot(point.x() - a.x(), point.y() - a.y())
        ap_x = point.x() - a.x()
        ap_y = point.y() - a.y()
        t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
        proj_x = a.x() + t * ab_x
        proj_y = a.y() + t * ab_y
        return math.hypot(point.x() - proj_x, point.y() - proj_y)

    def _find_insertion_index(self, img_pos: QtCore.QPointF) -> int:
        if len(self._points) < 2:
            return len(self._points)

        best_index = len(self._points)
        best_distance = float("inf")
        for i, start in enumerate(self._points):
            end = self._points[(i + 1) % len(self._points)]
            dist = self._point_to_segment_distance(img_pos, start, end)
            if dist < best_distance:
                best_distance = dist
                best_index = i + 1
        return best_index

    def _size_handle_at(self, pos: QtCore.QPoint) -> Optional[Tuple[str, str]]:
        if not self._size_overlay_enabled:
            return None
        if self._pixmap is None:
            return None
        handle_radius = 10
        for target in ("min", "max"):
            rect = self._size_rects.get(target)
            if rect is None:
                continue
            top_left = self._image_to_widget(rect.topLeft())
            bottom_right = self._image_to_widget(rect.bottomRight())
            if top_left is None or bottom_right is None:
                continue
            widget_rect = QtCore.QRectF(top_left, bottom_right).normalized()
            corners = {
                "tl": widget_rect.topLeft(),
                "tr": widget_rect.topRight(),
                "bl": widget_rect.bottomLeft(),
                "br": widget_rect.bottomRight(),
            }
            for name, corner in corners.items():
                if (corner - QtCore.QPointF(pos)).manhattanLength() <= handle_radius:
                    return target, name
            if widget_rect.contains(pos):
                return target, "move"
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return

        size_handle = self._size_handle_at(event.pos()) if self._size_overlay_enabled else None
        if size_handle:
            self._active_size_target, self._active_size_handle = size_handle
            self._active_size_origin = self._widget_to_image_clamped(event.pos())
            rect = self._size_rects.get(self._active_size_target)
            self._active_size_rect = QtCore.QRectF(rect) if rect else None
            return

        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return

        if self._size_capture_target:
            self._size_capture_start = self._size_capture_end = img_pos
            self.update()
            return

        if not self._roi_usage_enabled:
            return

        handle_radius = 8
        closest_idx = None
        closest_dist = handle_radius + 1
        for idx, point in enumerate(self._points):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            dist = (widget_point - QtCore.QPointF(event.pos())).manhattanLength()
            if dist < closest_dist:
                closest_idx = idx
                closest_dist = dist

        if closest_idx is not None:
            self._drag_index = closest_idx
        else:
            self._drag_index = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._active_size_target and self._active_size_rect:
            img_pos = self._widget_to_image_clamped(event.pos())
            if img_pos is None:
                return
            rect = QtCore.QRectF(
                self._size_rects.get(self._active_size_target) or self._active_size_rect
            )
            handle = self._active_size_handle
            if handle == "move" and self._active_size_origin is not None:
                delta = img_pos - self._active_size_origin
                rect.translate(delta)
                self._active_size_origin = img_pos
            elif handle:
                if "l" in handle:
                    rect.setLeft(img_pos.x())
                if "r" in handle:
                    rect.setRight(img_pos.x())
                if "t" in handle:
                    rect.setTop(img_pos.y())
                if "b" in handle:
                    rect.setBottom(img_pos.y())
            rect = rect.normalized()
            rect = self._clamp_rect(rect)
            self._size_rects[self._active_size_target] = rect
            self.plate_size_selected.emit(
                self._active_size_target, int(rect.width()), int(rect.height())
            )
            self.update()
            return

        if self._drag_index is None:
            return
        if not self._roi_usage_enabled:
            return
        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return
        self._points[self._drag_index] = img_pos
        self._clamp_points()
        self._emit_roi()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._active_size_target:
            self._active_size_target = None
            self._active_size_handle = None
            self._active_size_origin = None
            self._active_size_rect = None
            return

        self._drag_index = None

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return

        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return
        if not self._roi_usage_enabled:
            return

        handle_radius = 8
        closest_idx = None
        closest_dist = handle_radius + 1
        for idx, point in enumerate(self._points):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            dist = (widget_point - QtCore.QPointF(event.pos())).manhattanLength()
            if dist < closest_dist:
                closest_idx = idx
                closest_dist = dist

        if closest_idx is not None:
            self._points.pop(closest_idx)
        else:
            insert_at = self._find_insertion_index(img_pos)
            self._points.insert(insert_at, img_pos)

        self._drag_index = None
        self._clamp_points()
        self._emit_roi()
        self.update()

class EventDetailView(QtWidgets.QWidget):
    """Отображение выбранного события: метаданные, кадр и область номера."""

    def __init__(self) -> None:
        super().__init__()
        self._theme_colors: Dict[str, str] = MainWindow.THEME_PALETTES["dark"]
        layout = QtWidgets.QVBoxLayout(self)

        self._frame_image: Optional[QtGui.QImage] = None
        self._plate_image: Optional[QtGui.QImage] = None

        self.frame_preview = self._build_preview("Кадр распознавания", min_height=320, keep_aspect=True)
        layout.addWidget(self.frame_preview, stretch=3)

        bottom_row = QtWidgets.QHBoxLayout()
        self.plate_preview = self._build_preview("Кадр номера", min_size=QtCore.QSize(200, 140), keep_aspect=True)
        self.frame_preview.display_label.installEventFilter(self)  # type: ignore[attr-defined]
        self.plate_preview.display_label.installEventFilter(self)  # type: ignore[attr-defined]
        bottom_row.addWidget(self.plate_preview, 1)

        meta_group = QtWidgets.QGroupBox("Данные распознавания")
        meta_group.setMinimumWidth(220)
        meta_layout = QtWidgets.QFormLayout(meta_group)
        self.time_label = QtWidgets.QLabel("—")
        self.channel_label = QtWidgets.QLabel("—")
        self.country_label = QtWidgets.QLabel("—")
        self.plate_label = QtWidgets.QLabel("—")
        self.conf_label = QtWidgets.QLabel("—")
        self.direction_label = QtWidgets.QLabel("—")
        meta_layout.addRow("Дата/Время:", self.time_label)
        meta_layout.addRow("Канал:", self.channel_label)
        meta_layout.addRow("Страна:", self.country_label)
        meta_layout.addRow("Гос. номер:", self.plate_label)
        meta_layout.addRow("Уверенность:", self.conf_label)
        meta_layout.addRow("Направление:", self.direction_label)
        bottom_row.addWidget(meta_group, 1)

        layout.addLayout(bottom_row, stretch=1)
        self.set_theme(self._theme_colors)

    def set_theme(self, colors: Dict[str, str]) -> None:
        self._theme_colors = colors
        self.setStyleSheet(
            f"QWidget {{ background-color: {colors['surface']}; }}"
            f"QGroupBox {{ background-color: {colors['panel']}; color: {colors['text_primary']}; border: 1px solid {colors['border']}; border-radius: 12px; padding: 10px; margin-top: 8px; }}"
            f"QLabel {{ color: {colors['text_primary']}; }}"
        )
        for group in (self.frame_preview, self.plate_preview):
            group.display_label.setStyleSheet(
                f"background-color: {colors['field_bg']}; color: {colors['text_muted']}; border: 1px solid {colors['border']}; border-radius: 10px;"
            )  # type: ignore[attr-defined]

    def _build_preview(
        self,
        title: str,
        min_height: int = 180,
        min_size: Optional[QtCore.QSize] = None,
        keep_aspect: bool = False,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        wrapper = QtWidgets.QVBoxLayout(group)
        label = QtWidgets.QLabel("Нет изображения")
        label.setAlignment(QtCore.Qt.AlignCenter)
        if min_size:
            label.setMinimumSize(min_size)
        else:
            label.setMinimumHeight(min_height)
        label.setStyleSheet(
            "background-color: #0b0c10; color: #9ca3af; border: 1px solid #1f2937; border-radius: 10px;"
        )
        label.setScaledContents(not keep_aspect)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        wrapper.addWidget(label)
        group.display_label = label  # type: ignore[attr-defined]
        return group

    def clear(self) -> None:
        self.time_label.setText("—")
        self.channel_label.setText("—")
        self.country_label.setText("—")
        self.plate_label.setText("—")
        self.conf_label.setText("—")
        self.direction_label.setText("—")
        self._frame_image = None
        self._plate_image = None
        for group in (self.frame_preview, self.plate_preview):
            group.display_label.setPixmap(QtGui.QPixmap())  # type: ignore[attr-defined]
            group.display_label.setText("Нет изображения")  # type: ignore[attr-defined]

    def set_event(
        self,
        event: Optional[Dict],
        frame_image: Optional[QtGui.QImage] = None,
        plate_image: Optional[QtGui.QImage] = None,
    ) -> None:
        if event is None:
            self.clear()
            return

        self.time_label.setText(event.get("timestamp", "—"))
        self.channel_label.setText(event.get("channel", "—"))
        self.country_label.setText(event.get("country") or "—")
        plate = event.get("plate") or "—"
        self.plate_label.setText(plate)
        conf = event.get("confidence")
        self.conf_label.setText(f"{float(conf):.2f}" if conf is not None else "—")
        self.direction_label.setText(event.get("direction", "—"))

        self._set_image(self.frame_preview, frame_image, keep_aspect=True)
        self._set_image(self.plate_preview, plate_image, keep_aspect=True)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if event.type() == QtCore.QEvent.Resize:
            if obj is getattr(self.frame_preview, "display_label", None):
                self._apply_image_to_label(self.frame_preview, self._frame_image, keep_aspect=True)
            elif obj is getattr(self.plate_preview, "display_label", None):
                self._apply_image_to_label(self.plate_preview, self._plate_image, keep_aspect=True)
        return super().eventFilter(obj, event)

    def _set_image(
        self,
        group: QtWidgets.QGroupBox,
        image: Optional[QtGui.QImage],
        keep_aspect: bool = False,
    ) -> None:
        if group is self.frame_preview:
            self._frame_image = image
        elif group is self.plate_preview:
            self._plate_image = image

        self._apply_image_to_label(group, image, keep_aspect)

    def _apply_image_to_label(
        self,
        group: QtWidgets.QGroupBox,
        image: Optional[QtGui.QImage],
        keep_aspect: bool = False,
    ) -> None:
        label: QtWidgets.QLabel = group.display_label  # type: ignore[attr-defined]
        if image is None:
            label.setPixmap(QtGui.QPixmap())
            label.setText("Нет изображения")
            return

        label.setText("")
        target_size = label.contentsRect().size()
        if target_size.width() == 0 or target_size.height() == 0:
            return
        pixmap = QtGui.QPixmap.fromImage(image)
        if keep_aspect:
            pixmap = pixmap.scaled(
                target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        else:
            pixmap = pixmap.scaled(target_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_image_to_label(self.frame_preview, self._frame_image, keep_aspect=True)
        self._apply_image_to_label(self.plate_preview, self._plate_image, keep_aspect=True)


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ANPR с вкладками наблюдения, поиска и настроек."""

    THEME_PALETTES = {
        "dark": {
            "accent": "#22d3ee",
            "background": "#0b0c10",
            "surface": "#16181d",
        "panel": "#0f1115",
        "border": "#20242c",
        "text_primary": "#e5e7eb",
        "text_secondary": "#cbd5e1",
        "text_muted": "#9ca3af",
        "text_inverse": "#0b0c10",
        "field_bg": "#0b0c10",
        "header_bg": "#0b0c10",
        "table_header_bg": "#11131a",
        "table_row_bg": "#0b0c10",
        "overlay_bg": "rgba(0, 0, 0, 0.55)",
    },
    "light": {
        "accent": "#0ea5e9",
        "background": "#f5f7fb",
        "surface": "#ffffff",
            "panel": "#eef2f7",
            "border": "#d4d4d8",
            "text_primary": "#0f172a",
            "text_secondary": "#1f2937",
        "text_muted": "#475569",
        "text_inverse": "#ffffff",
        "field_bg": "#ffffff",
        "header_bg": "#e5e7eb",
        "table_header_bg": "#e5e7eb",
        "table_row_bg": "#ffffff",
        "overlay_bg": "rgba(255, 255, 255, 0.8)",
    },
}
    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3"]
    MAX_IMAGE_CACHE = 200
    MAX_IMAGE_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB
    FIELD_MAX_WIDTH = 520
    COMPACT_FIELD_WIDTH = 180
    FIELD_MIN_WIDTH = 360
    BUTTON_HEIGHT = 36
    LABEL_MIN_WIDTH = 180
    ACTION_BUTTON_WIDTH = 180

    @staticmethod
    def _default_roi_region() -> Dict[str, Any]:
        return {"unit": "px", "points": [point.copy() for point in DEFAULT_ROI_POINTS]}

    def __init__(self, settings: Optional[Config] = None) -> None:
        super().__init__()
        self._theme_setters: List[Callable[[], None]] = []
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)
        screen = QtWidgets.QApplication.primaryScreen()
        self._top_left = (
            screen.availableGeometry().topLeft()
            if screen is not None
            else QtCore.QPoint(0, 0)
        )
        self.move(self._top_left)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        self._window_drag_pos: Optional[QtCore.QPoint] = None

        self.settings = settings or Config()
        self.theme = self.settings.get_theme()
        self._apply_theme_palette(self.theme)
        self.current_grid = self.settings.get_grid()
        if self.current_grid not in self.GRID_VARIANTS:
            self.current_grid = self.GRID_VARIANTS[0]
        self.db = EventDatabase(self.settings.get_db_path())

        self._pixmap_pool = PixmapPool()
        self._log_history: deque[str] = deque(maxlen=500)
        self._log_emitter = LogSignalEmitter()
        self._log_emitter.message.connect(self._append_log_message)
        self._log_handler = QtLogHandler(self._log_emitter)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        self._log_handler_attached = False
        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, ChannelView] = {}
        self.focused_channel_name: Optional[str] = None
        self._previous_grid: Optional[str] = None
        self._debug_settings_cache: Optional[Dict[str, Any]] = None
        self.event_images: "OrderedDict[int, Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]]" = OrderedDict()
        self._image_cache_bytes = 0
        self.event_cache: Dict[int, Dict] = {}
        self.search_results: Dict[int, Dict] = {}
        self.flag_cache: Dict[str, Optional[QtGui.QIcon]] = {}
        self.flag_dir = Path(__file__).resolve().parents[2] / "images" / "flags"
        self.country_display_names = self._load_country_names()
        self._pending_channels: Optional[List[Dict[str, Any]]] = None
        self._channel_save_timer = QtCore.QTimer(self)
        self._channel_save_timer.setSingleShot(True)
        self._channel_save_timer.timeout.connect(self._flush_pending_channels)
        self._pending_channel_restarts: set[int] = set()
        self._drag_counter = 0
        self._skip_frame_updates = False
        self._latest_frames: Dict[str, QtGui.QImage] = {}

        self.tabs = QtWidgets.QTabWidget()
        self._apply_stylesheet(self.tabs, lambda: self.tabs_style)
        self.observation_tab = self._build_observation_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.observation_tab, "Наблюдение")
        self.tabs.addTab(self.search_tab, "Журнал")
        self.tabs.addTab(self.settings_tab, "Настройки")

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        header = self._build_main_header()
        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)
        self._apply_stylesheet(self, lambda: self.main_style)
        self._build_status_bar()
        self._start_system_monitoring()
        self._refresh_events_table()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_app_about_to_quit)
        self._start_channels()
        # Повторно применяем тему после инициализации всех виджетов,
        # чтобы зарегистрированные сеттеры обновили стили при стартовом запуске.
        self._apply_theme_styles()

    def _apply_theme_palette(self, theme: str) -> None:
        palette = self.THEME_PALETTES.get(theme, self.THEME_PALETTES["dark"])
        self.theme = theme if theme in self.THEME_PALETTES else "dark"
        self.colors = palette
        self._build_styles()
        self._apply_theme_styles()

    def _build_styles(self) -> None:
        c = self.colors
        accent_color = QtGui.QColor(c["accent"])
        accent_lighter = accent_color.lighter(115).name()
        accent_darker = accent_color.darker(110).name()
        accent_rgba = f"rgba({accent_color.red()}, {accent_color.green()}, {accent_color.blue()}, 38)"
        self.group_box_style = (
            f"QGroupBox {{ background-color: {c['surface']}; color: {c['text_primary']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 12px; margin-top: 10px; }}"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; font-weight: 700; }"
            f"QLabel {{ color: {c['text_secondary']}; font-size: 13px; }}"
            f"QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {{ background-color: {c['field_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 8px; }}"
            f"QPushButton {{ background-color: {c['accent']}; color: {c['text_inverse']}; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }}"
            f"QPushButton:hover {{ background-color: {accent_lighter}; }}"
            f"QCheckBox {{ color: {c['text_primary']}; font-size: 13px; }}"
        )
        self.form_style = (
            f"QLabel {{ color: {c['text_secondary']}; font-size: 13px; }}"
            f"QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {{ background-color: {c['field_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 8px; }}"
            f"QPushButton {{ background-color: {c['accent']}; color: {c['text_inverse']}; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }}"
            f"QPushButton:hover {{ background-color: {accent_lighter}; }}"
            f"QCheckBox {{ color: {c['text_primary']}; font-size: 13px; }}"
            f"QWidget {{ background-color: {c['surface']}; }}"
        )
        self.primary_hollow_button = (
            f"QPushButton {{ background-color: transparent; color: {c['text_primary']}; border: 1px solid {c['text_primary']}; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }}"
            f"QPushButton:hover {{ background-color: {c['accent']}; color: {c['text_inverse']}; }}"
            f"QPushButton:pressed {{ background-color: {accent_darker}; color: {c['text_inverse']}; }}"
        )
        self.table_style = (
            f"QHeaderView {{ background-color: {c['table_header_bg']}; border: none; }}"
            f"QHeaderView::section {{ background-color: {c['table_header_bg']}; color: {c['text_secondary']}; padding: 8px; font-weight: 700; border: none; }}"
            f"QTableWidget {{ background-color: {c['table_row_bg']}; color: {c['text_primary']}; gridline-color: {c['border']}; selection-background-color: {accent_rgba}; }}"
            f"QTableWidget::item {{ border-bottom: 1px solid {c['border']}; padding: 6px; }}"
            f"QTableWidget::item:selected {{ background-color: {accent_rgba}; color: {c['accent']}; border: 1px solid {c['accent']}; }}"
            f"QScrollBar:vertical {{ background: {c['field_bg']}; width: 12px; margin: 0px; border: 1px solid {c['border']}; border-radius: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {c['border']}; min-height: 24px; border-radius: 6px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
            f"QScrollBar:horizontal {{ background: {c['field_bg']}; height: 12px; margin: 0px; border: 1px solid {c['border']}; border-radius: 6px; }}"
            f"QScrollBar::handle:horizontal {{ background: {c['border']}; min-width: 24px; border-radius: 6px; }}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }"
        )
        self.list_style = (
            f"QListWidget {{ background-color: {c['field_bg']}; color: {c['text_secondary']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 6px; }}"
            f"QListWidget::item:selected {{ background-color: rgba({accent_color.red()}, {accent_color.green()}, {accent_color.blue()}, 32); color: {c['accent']}; border-radius: 6px; }}"
            "QListWidget::item { padding: 8px 10px; margin: 2px 0; }"
        )
        self.combo_style = (
            f"QComboBox {{ background-color: {c['field_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 6px 10px; min-width: 0px; }}"
            "QComboBox::drop-down { border: 0px; width: 28px; }"
            "QComboBox::down-arrow { image: url(:/qt-project.org/styles/commonstyle/images/arrowdown.png); width: 12px; height: 12px; margin-right: 6px; }"
            f"QComboBox QAbstractItemView {{ background-color: {c['field_bg']}; color: {c['text_primary']}; selection-background-color: {c['accent']}; border: 1px solid {c['border']}; padding: 6px; }}"
        )
        self.combo_plain_style = (
            f"QComboBox {{ background-color: {c['field_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; }}"
            f"QComboBox QAbstractItemView {{ background-color: {c['field_bg']}; color: {c['text_primary']}; selection-background-color: {c['accent']}; }}"
            "QComboBox:on { padding-top: 3px; padding-left: 4px; }"
        )
        self.tabs_style = (
            "QTabBar { font-weight: 700; }"
            f"QTabBar::tab {{ background: {c['field_bg']}; color: {c['text_muted']}; padding: 10px 18px; border: 1px solid {c['border']}; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 6px; }}"
            f"QTabBar::tab:selected {{ background: {c['surface']}; color: {c['accent']}; border: 1px solid {c['border']}; border-bottom: 2px solid {c['accent']}; }}"
            f"QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 10px; background-color: {c['surface']}; top: -1px; }}"
        )
        self.main_style = (
            f"QMainWindow {{ background-color: {c['background']}; }}"
            f"QStatusBar {{ background-color: {c['background']}; color: {c['text_primary']}; padding: 4px; border-top: 1px solid {c['border']}; }}"
            f"QToolButton[windowControl='true'] {{ background-color: transparent; border: none; color: {c['text_primary']}; padding: 6px; }}"
            f"QToolButton[windowControl='true']:hover {{ background-color: rgba({QtGui.QColor(c['text_primary']).red()}, {QtGui.QColor(c['text_primary']).green()}, {QtGui.QColor(c['text_primary']).blue()}, 20); border-radius: 6px; }}"
        )

    def _apply_theme_styles(self) -> None:
        for setter in getattr(self, "_theme_setters", []):
            setter()

    def _register_theme_setter(self, setter: Callable[[], None]) -> None:
        self._theme_setters.append(setter)
        setter()

    def _apply_stylesheet(self, widget: QtWidgets.QWidget, stylesheet: Any) -> None:
        def apply_style() -> None:
            style_value = stylesheet() if callable(stylesheet) else stylesheet
            widget.setStyleSheet(style_value)

        self._register_theme_setter(apply_style)

    def _style_title_label(self, label: QtWidgets.QLabel) -> None:
        self._apply_stylesheet(label, f"color: {self.colors['text_primary']}; font-weight: 800;")

    def _style_combo(self, combo: QtWidgets.QComboBox, rounded: bool = False) -> None:
        style = lambda: self.combo_style if rounded else self.combo_plain_style
        self._apply_stylesheet(combo, style)

    def _toggle_theme(self) -> None:
        new_theme = "light" if self.theme == "dark" else "dark"
        self.settings.save_theme(new_theme)
        self._apply_theme_palette(new_theme)
        self._update_theme_button_icon()

    def _update_theme_button_icon(self) -> None:
        if hasattr(self, "theme_btn"):
            self.theme_btn.setText("🌙" if self.theme == "light" else "☀")
    def _build_main_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header.setFixedHeight(48)
        self._apply_stylesheet(
            header,
            lambda: f"QFrame {{ background-color: {self.colors['header_bg']}; border-bottom: 1px solid {self.colors['border']}; }}",
        )
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("ANPR Desktop")
        self._style_title_label(title)
        layout.addWidget(title)
        layout.addStretch()

        self.theme_btn = QtWidgets.QToolButton()
        self.theme_btn.setProperty("windowControl", True)
        self.theme_btn.setToolTip("Переключить тему")
        self.theme_btn.clicked.connect(self._toggle_theme)
        self.theme_btn.setMinimumWidth(34)
        self._apply_stylesheet(
            self.theme_btn,
            lambda: (
                f"QToolButton {{ background-color: transparent; border: 1px solid {self.colors['border']}; color: {self.colors['text_primary']}; padding: 6px; border-radius: 8px; }}"
                f"QToolButton:hover {{ background-color: rgba({QtGui.QColor(self.colors['text_primary']).red()}, {QtGui.QColor(self.colors['text_primary']).green()}, {QtGui.QColor(self.colors['text_primary']).blue()}, 18); }}"
            ),
        )
        layout.addWidget(self.theme_btn)

        minimize_btn = QtWidgets.QToolButton()
        minimize_btn.setProperty("windowControl", True)
        minimize_btn.setText("–")
        minimize_btn.setToolTip("Свернуть")
        minimize_btn.clicked.connect(self.showMinimized)
        layout.addWidget(minimize_btn)

        maximize_btn = QtWidgets.QToolButton()
        maximize_btn.setProperty("windowControl", True)
        maximize_btn.setText("⛶")
        maximize_btn.setToolTip("Развернуть/свернуть окно")

        def toggle_maximize() -> None:
            if self.isFullScreen():
                self.showNormal()
                self.move(self._top_left)
            else:
                self.showFullScreen()

        maximize_btn.clicked.connect(toggle_maximize)
        layout.addWidget(maximize_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setProperty("windowControl", True)
        close_btn.setText("✕")
        close_btn.setToolTip("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        def start_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton and not self.isFullScreen():
                self._window_drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()

        def move_window(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if (
                event.buttons() & QtCore.Qt.LeftButton
                and self._window_drag_pos is not None
                and not self.isFullScreen()
            ):
                self.move(event.globalPos() - self._window_drag_pos)
                event.accept()

        def end_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            self._window_drag_pos = None
            event.accept()

        def double_click(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton:
                toggle_maximize()
                event.accept()

        header.mousePressEvent = start_move  # type: ignore[assignment]
        header.mouseMoveEvent = move_window  # type: ignore[assignment]
        header.mouseReleaseEvent = end_move  # type: ignore[assignment]
        header.mouseDoubleClickEvent = double_click  # type: ignore[assignment]
        self._register_theme_setter(self._update_theme_button_icon)
        return header

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        self._apply_stylesheet(
            status,
            lambda: f"background-color: {self.colors['background']}; color: {self.colors['text_primary']}; padding: 6px; border-top: 1px solid {self.colors['border']};",
        )
        status.setSizeGripEnabled(False)
        self.cpu_label = QtWidgets.QLabel("CPU: —")
        self.ram_label = QtWidgets.QLabel("RAM: —")
        status.addPermanentWidget(self.cpu_label)
        status.addPermanentWidget(self.ram_label)

    def _start_system_monitoring(self) -> None:
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self._update_system_stats)
        self.stats_timer.start()
        self._update_system_stats()

    def _update_system_stats(self) -> None:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        self.cpu_label.setText(f"CPU: {cpu_percent:.0f}%")
        self.ram_label.setText(f"RAM: {ram_percent:.0f}%")

    def _enable_log_streaming(self) -> None:
        if self._log_handler_attached:
            return
        logging.getLogger().addHandler(self._log_handler)
        self._log_handler_attached = True

    def _disable_log_streaming(self) -> None:
        if not self._log_handler_attached:
            return
        try:
            logging.getLogger().removeHandler(self._log_handler)
        except ValueError:
            pass
        self._log_handler_attached = False

    def _set_log_panel_visible(self, visible: bool) -> None:
        if hasattr(self, "log_group"):
            self.log_group.setVisible(visible)
        if visible:
            self._enable_log_streaming()
            self._refresh_log_view()
        else:
            self._disable_log_streaming()

    def _append_log_message(self, message: str) -> None:
        self._log_history.append(message)
        if getattr(self, "log_group", None) and self.log_group.isVisible():
            self.log_view.append(message)
            self.log_view.moveCursor(QtGui.QTextCursor.End)

    def _refresh_log_view(self) -> None:
        if not getattr(self, "log_view", None):
            return
        self.log_view.clear()
        for line in self._log_history:
            self.log_view.append(line)
        self.log_view.moveCursor(QtGui.QTextCursor.End)

    # ------------------ Наблюдение ------------------
    def _build_observation_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setSpacing(4)

        left_widget = QtWidgets.QWidget()
        left_column = QtWidgets.QVBoxLayout(left_widget)
        left_column.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        chooser_layout = QtWidgets.QHBoxLayout()
        chooser_layout.setSpacing(8)
        chooser_layout.setContentsMargins(4, 4, 4, 4)
        chooser_label = QtWidgets.QLabel("Сетка")
        self._style_title_label(chooser_label)
        chooser_layout.addWidget(chooser_label)
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        self.grid_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self._style_combo(self.grid_combo, rounded=True)
        for variant in self.GRID_VARIANTS:
            self.grid_combo.addItem(variant.replace("x", "х"), variant)
        combo_width = max(self.fontMetrics().horizontalAdvance("3х3") + 26, 80)
        self.grid_combo.setMinimumWidth(combo_width)
        self.grid_combo.setMinimumContentsLength(3)
        current_index = self.grid_combo.findData(self.current_grid)
        if current_index >= 0:
            self.grid_combo.setCurrentIndex(current_index)
        self.grid_combo.currentIndexChanged.connect(self._on_grid_combo_changed)
        chooser_layout.addWidget(self.grid_combo)

        controls.addLayout(chooser_layout)
        controls.addStretch()
        left_column.addLayout(controls)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        left_column.addWidget(self.grid_widget, stretch=4)

        self.log_group = QtWidgets.QGroupBox("Логи (Debug)")
        self._apply_stylesheet(self.log_group, lambda: self.group_box_style)
        log_layout = QtWidgets.QVBoxLayout(self.log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        self.log_view.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self._apply_stylesheet(
            self.log_view,
            lambda: (
                f"QTextEdit {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 8px; }}"
            ),
        )
        log_layout.addWidget(self.log_view)
        self.log_group.setVisible(False)
        left_column.addWidget(self.log_group, stretch=1)

        right_column = QtWidgets.QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)

        right_header = QtWidgets.QHBoxLayout()
        right_header.setContentsMargins(0, 0, 0, 0)
        right_title = QtWidgets.QLabel("Детали")
        self._style_title_label(right_title)
        right_header.addWidget(right_title)
        right_header.addStretch()
        right_column.addLayout(right_header)
        details_group = QtWidgets.QGroupBox("Информация о событии")
        self._apply_stylesheet(details_group, lambda: self.group_box_style)
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.event_detail = EventDetailView()
        self._register_theme_setter(lambda: self.event_detail.set_theme(self.colors))
        details_layout.addWidget(self.event_detail)
        right_column.addWidget(details_group, stretch=3)

        events_group = QtWidgets.QGroupBox("События")
        self._apply_stylesheet(events_group, lambda: self.group_box_style)
        events_layout = QtWidgets.QVBoxLayout(events_group)
        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(["Дата/Время", "Гос. номер", "Страна", "Канал", "Направление"])
        self._apply_stylesheet(self.events_table, lambda: self.table_style)
        header = self.events_table.horizontalHeader()
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(False)
        self.events_table.setColumnWidth(0, 190)
        self.events_table.setColumnWidth(1, 130)
        self.events_table.setColumnWidth(2, 90)
        self.events_table.setColumnWidth(3, 140)
        self.events_table.setColumnWidth(4, 130)
        self.events_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.itemSelectionChanged.connect(self._on_event_selected)
        events_layout.addWidget(self.events_table)
        right_column.addWidget(events_group, stretch=1)

        toggle_details_btn = QtWidgets.QToolButton()
        toggle_details_btn.setCheckable(True)
        toggle_details_btn.setChecked(False)
        toggle_details_btn.setText("▶")
        toggle_details_btn.setToolTip("Скрыть панель деталей")
        toggle_details_btn.setFixedSize(26, 26)
        self._apply_stylesheet(
            toggle_details_btn,
            lambda: (
                f"QToolButton {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 6px; }}"
                f"QToolButton:hover {{ background-color: rgba({QtGui.QColor(self.colors['text_primary']).red()}, {QtGui.QColor(self.colors['text_primary']).green()}, {QtGui.QColor(self.colors['text_primary']).blue()}, 12); }}"
            ),
        )

        toggle_rail = QtWidgets.QFrame()
        toggle_rail.setFixedWidth(max(toggle_details_btn.sizeHint().width() + 2, 26))
        toggle_rail.setStyleSheet("QFrame { background-color: transparent; }")
        rail_layout = QtWidgets.QVBoxLayout(toggle_rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.addStretch()
        rail_layout.addWidget(toggle_details_btn, 0, QtCore.Qt.AlignCenter)
        rail_layout.addStretch()

        details_content = QtWidgets.QWidget()
        details_content.setLayout(right_column)

        details_container = QtWidgets.QWidget()
        details_layout = QtWidgets.QHBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(0)
        details_layout.addWidget(toggle_rail)
        details_layout.addWidget(details_content, 1)

        def toggle_details_panel(checked: bool) -> None:
            if checked:
                right_title.hide()
                details_group.hide()
                events_group.hide()
                details_content.hide()
                details_container.setMinimumWidth(toggle_rail.width())
                details_container.setMaximumWidth(toggle_rail.width())
                toggle_details_btn.setText("◀")
                toggle_details_btn.setToolTip("Показать панель деталей")
            else:
                right_title.show()
                details_group.show()
                events_group.show()
                details_content.show()
                details_container.setMinimumWidth(0)
                details_container.setMaximumWidth(16777215)
                toggle_details_btn.setText("▶")
                toggle_details_btn.setToolTip("Скрыть панель деталей")

        toggle_details_btn.toggled.connect(toggle_details_panel)

        layout.addWidget(left_widget, stretch=3)
        layout.addWidget(details_container, stretch=2)

        self._draw_grid()
        return widget

    def _build_debug_settings_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._apply_stylesheet(
            scroll,
            lambda: (
                f"QScrollArea {{ background: transparent; border: none; }}"
                f"QScrollArea > QWidget > QWidget {{ background-color: {self.colors['surface']}; }}"
            ),
        )

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self._apply_stylesheet(widget, lambda: self.form_style)

        debug_group = QtWidgets.QGroupBox("Глобальный Debug")
        self._apply_stylesheet(debug_group, lambda: self.group_box_style)
        debug_form = QtWidgets.QFormLayout(debug_group)
        self._tune_form_layout(debug_form)

        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.setSpacing(10)
        self.debug_detection_global_checkbox = QtWidgets.QCheckBox("Рамки детекции")
        self.debug_ocr_global_checkbox = QtWidgets.QCheckBox("Символы OCR")
        self.debug_direction_global_checkbox = QtWidgets.QCheckBox("Трек движения")
        for checkbox in (
            self.debug_detection_global_checkbox,
            self.debug_ocr_global_checkbox,
            self.debug_direction_global_checkbox,
        ):
            overlay_row.addWidget(checkbox)
        overlay_row.addStretch(1)
        debug_form.addRow("Оверлеи:", overlay_row)

        self.debug_log_checkbox = QtWidgets.QCheckBox("Лог")
        self.debug_log_checkbox.setToolTip("Показывать поток логов под сеткой наблюдения во вкладке «Наблюдение»")
        debug_form.addRow("Логирование:", self.debug_log_checkbox)

        for checkbox in (
            self.debug_detection_global_checkbox,
            self.debug_ocr_global_checkbox,
            self.debug_direction_global_checkbox,
            self.debug_log_checkbox,
        ):
            checkbox.stateChanged.connect(self._on_debug_settings_changed)

        layout.addWidget(debug_group)
        layout.addStretch(1)
        scroll.setWidget(widget)

        self._load_debug_settings()
        return scroll

    def _polish_button(self, button: QtWidgets.QPushButton, min_width: int = 140) -> None:
        def apply_style() -> None:
            button.setStyleSheet(self.primary_hollow_button)

        self._register_theme_setter(apply_style)
        button.setMinimumWidth(min_width)
        button.setMinimumHeight(MainWindow.BUTTON_HEIGHT)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def _style_channel_tool_button(self, button: QtWidgets.QToolButton) -> None:
        def apply_style() -> None:
            button.setStyleSheet(
                f"QToolButton {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 6px; }} "
                f"QToolButton:hover {{ background-color: rgba({QtGui.QColor(self.colors['text_primary']).red()}, {QtGui.QColor(self.colors['text_primary']).green()}, {QtGui.QColor(self.colors['text_primary']).blue()}, 14); }} "
                f"QToolButton:checked {{ background-color: {self.colors['accent']}; color: {self.colors['text_inverse']}; }}"
            )

        self._register_theme_setter(apply_style)
        button.setAutoRaise(False)
        button.setCursor(QtCore.Qt.PointingHandCursor)

    @staticmethod
    def _tune_form_layout(form: QtWidgets.QFormLayout) -> None:
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

    @staticmethod
    def _ensure_label_width(label: QtWidgets.QLabel) -> None:
        label.setMinimumWidth(MainWindow.LABEL_MIN_WIDTH)

    @staticmethod
    def _configure_line_edit(line_edit: QtWidgets.QLineEdit, max_width: Optional[int] = None) -> None:
        line_edit.setMinimumWidth(MainWindow.FIELD_MIN_WIDTH)
        if max_width is not None:
            line_edit.setMaximumWidth(max_width)

    @staticmethod
    def _configure_combo(combo: QtWidgets.QComboBox, max_width: Optional[int] = None) -> None:
        combo.setMinimumWidth(MainWindow.COMPACT_FIELD_WIDTH + 80)
        if max_width is not None:
            combo.setMaximumWidth(max_width)

    @staticmethod
    def _prepare_optional_datetime(widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setCalendarPopup(True)
        widget.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        min_dt = QtCore.QDateTime.fromSecsSinceEpoch(0)
        widget.setMinimumDateTime(min_dt)
        widget.setSpecialValueText("Не выбрано")
        widget.setDateTime(min_dt)

    def _apply_calendar_style(self, widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setStyleSheet(
            f"QDateTimeEdit {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 8px; padding: 6px; }}"
            "QDateTimeEdit::drop-down { width: 24px; }"
        )
        calendar = widget.calendarWidget()
        accent = self.colors["accent"]
        calendar.setStyleSheet(
            f"QCalendarWidget QWidget {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; }}"
            f"QCalendarWidget QToolButton {{ background-color: {self.colors['border']}; color: {self.colors['text_primary']}; border: none; padding: 6px; }}"
            "QCalendarWidget QToolButton#qt_calendar_prevmonth, QCalendarWidget QToolButton#qt_calendar_nextmonth { width: 24px; }"
            f"QCalendarWidget QSpinBox {{ background-color: {self.colors['border']}; color: {self.colors['text_primary']}; }}"
            f"QCalendarWidget QTableView {{ selection-background-color: {accent}; selection-color: {self.colors['text_inverse']}; alternate-background-color: {self.colors['surface']}; }}"
        )

    @staticmethod
    def _offset_label(minutes: int) -> str:
        sign = "+" if minutes >= 0 else "-"
        total = abs(minutes)
        hours = total // 60
        mins = total % 60
        return f"UTC{sign}{hours:02d}:{mins:02d}"

    @staticmethod
    def _available_offset_labels() -> List[str]:
        now = QtCore.QDateTime.currentDateTime()
        offsets = {0}
        for tz_id in QtCore.QTimeZone.availableTimeZoneIds():
            try:
                offset = QtCore.QTimeZone(tz_id).offsetFromUtc(now) // 60
                offsets.add(offset)
            except Exception:
                continue
        return [MainWindow._offset_label(minutes) for minutes in sorted(offsets)]

    @staticmethod
    def _get_datetime_value(widget: QtWidgets.QDateTimeEdit) -> Optional[str]:
        if widget.dateTime() == widget.minimumDateTime():
            return None
        return widget.dateTime().toString(QtCore.Qt.ISODate)

    @staticmethod
    def _format_timestamp(value: str, target_zone: Optional[ZoneInfo], offset_minutes: int) -> str:
        if not value:
            return "—"
        cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return value

        if parsed.tzinfo is None:
            assumed_zone = target_zone
            if assumed_zone is None:
                try:
                    assumed_zone = datetime.now().astimezone().tzinfo
                except Exception:
                    assumed_zone = None
            if assumed_zone:
                parsed = parsed.replace(tzinfo=assumed_zone)
            elif target_zone:
                parsed = parsed.replace(tzinfo=target_zone)

        if offset_minutes:
            parsed = parsed + timedelta(minutes=offset_minutes)

        if target_zone:
            try:
                parsed = parsed.astimezone(target_zone)
            except Exception:
                pass
        return parsed.strftime("%d.%m.%Y %H:%M:%S")

    @staticmethod
    def _parse_utc_offset_minutes(label: str) -> Optional[int]:
        if not label or not label.upper().startswith("UTC"):
            return None
        raw = label[3:].strip()
        if not raw:
            return 0
        try:
            sign = 1
            if raw.startswith("-"):
                sign = -1
                raw = raw[1:]
            elif raw.startswith("+"):
                raw = raw[1:]
            if not raw:
                return 0
            parts = raw.split(":")
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return sign * (hours * 60 + minutes)
        except (ValueError, TypeError):
            return None

    def _get_target_zone(self) -> Optional[timezone]:
        tz_value = self.settings.get_timezone()
        offset_minutes = self._parse_utc_offset_minutes(tz_value)
        if offset_minutes is not None:
            try:
                return timezone(timedelta(minutes=offset_minutes))
            except Exception:
                pass
        try:
            return ZoneInfo(tz_value)
        except Exception:
            return None

    def _draw_grid(self) -> None:
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.channel_labels.clear()
        self.grid_cells: Dict[int, ChannelView] = {}
        for col in range(3):
            self.grid_layout.setColumnStretch(col, 0)
        for row in range(3):
            self.grid_layout.setRowStretch(row, 0)
        channels = self.settings.get_channels()
        if self.current_grid == "1x1" and self.focused_channel_name:
            focused = [
                channel
                for channel in channels
                if channel.get("name") == self.focused_channel_name
            ]
            if focused:
                channels = focused
        rows, cols = map(int, self.current_grid.split("x"))
        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid_layout.setRowStretch(row, 1)
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = ChannelView(f"Канал {index+1}", self._pixmap_pool, self.colors)
                label.set_grid_position(index)
                channel_name: Optional[str] = None
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    self.channel_labels[channel_name] = label
                label.set_channel_name(channel_name)
                label.channelDropped.connect(self._on_channel_dropped)
                label.channelActivated.connect(self._on_channel_activated)
                label.dragStarted.connect(self._on_drag_started)
                label.dragFinished.connect(self._on_drag_finished)
                self._register_theme_setter(lambda l=label: l.set_theme(self.colors))
                self.grid_layout.addWidget(label, row, col)
                self.grid_cells[index] = label
                index += 1

        self.grid_layout.invalidate()
        self.grid_widget.updateGeometry()

    def _on_grid_combo_changed(self, index: int) -> None:
        variant = self.grid_combo.itemData(index)
        if variant:
            self._select_grid(str(variant))

    def _on_drag_started(self) -> None:
        self._drag_counter += 1
        if self._drag_counter == 1:
            self._skip_frame_updates = True
            self.grid_widget.setUpdatesEnabled(False)

    def _on_drag_finished(self) -> None:
        if self._drag_counter:
            self._drag_counter -= 1
        if self._drag_counter == 0:
            self._skip_frame_updates = False
            self.grid_widget.setUpdatesEnabled(True)

    def _on_channel_dropped(self, source_index: int, target_index: int) -> None:
        channels = self.settings.get_channels()
        if (
            source_index == target_index
            or source_index < 0
            or source_index >= len(channels)
        ):
            return

        if target_index >= len(channels):
            channel = channels.pop(source_index)
            channels.append(channel)
        else:
            channels[source_index], channels[target_index] = (
                channels[target_index],
                channels[source_index],
            )

        self._update_channels_list_names(channels)
        self._schedule_channels_save(channels)
        self.grid_widget.setUpdatesEnabled(False)
        try:
            if not self._swap_channel_views(source_index, target_index):
                self._draw_grid()
        finally:
            self.grid_widget.setUpdatesEnabled(True)

    def _swap_channel_views(self, source_index: int, target_index: int) -> bool:
        source_view = getattr(self, "grid_cells", {}).get(source_index)
        target_view = getattr(self, "grid_cells", {}).get(target_index)
        if not source_view or not target_view:
            return False

        source_name = source_view.channel_name()
        target_name = target_view.channel_name()
        source_view.set_channel_name(target_name)
        target_view.set_channel_name(source_name)

        self.channel_labels.clear()
        for view in self.grid_cells.values():
            name = view.channel_name()
            if name:
                self.channel_labels[name] = view

        for index, view in self.grid_cells.items():
            view.set_grid_position(index)
        return True

    def _update_channels_list_names(self, channels: List[Dict[str, Any]]) -> None:
        current_row = self.channels_list.currentRow()
        self.channels_list.blockSignals(True)
        self.channels_list.clear()
        for channel in channels:
            self.channels_list.addItem(self._channel_item_label(channel))
        self.channels_list.blockSignals(False)
        if self.channels_list.count():
            target_row = min(current_row if current_row >= 0 else 0, self.channels_list.count() - 1)
            self.channels_list.setCurrentRow(target_row)
        self._update_channel_action_states()

    def _schedule_channels_save(self, channels: List[Dict[str, Any]]) -> None:
        self._pending_channels = [dict(channel) for channel in channels]
        self._channel_save_timer.start(150)

    def _flush_pending_channels(self) -> None:
        if self._pending_channels is None:
            return
        self.settings.save_channels(self._pending_channels)
        self._pending_channels = None

    def _on_channel_activated(self, channel_name: str) -> None:
        if self.current_grid == "1x1" and self._previous_grid:
            previous = self._previous_grid
            self._previous_grid = None
            self.focused_channel_name = None
            self._select_grid(previous)
            return

        self.focused_channel_name = channel_name
        self._select_grid("1x1", focused=True)

    def _select_grid(self, grid: str, focused: bool = False) -> None:
        if grid != "1x1":
            self.focused_channel_name = None if not focused else self.focused_channel_name
            self._previous_grid = None
        elif not self._previous_grid and self.current_grid != "1x1":
            self._previous_grid = self.current_grid
        self.current_grid = grid
        if hasattr(self, "grid_combo"):
            combo_index = self.grid_combo.findData(grid)
            if combo_index >= 0 and combo_index != self.grid_combo.currentIndex():
                self.grid_combo.blockSignals(True)
                self.grid_combo.setCurrentIndex(combo_index)
                self.grid_combo.blockSignals(False)
        self.settings.save_grid(grid)
        self._draw_grid()

    def _start_channels(self) -> None:
        self._stop_workers()
        self.channel_workers = []
        reconnect_conf = self.settings.get_reconnect()
        plate_settings = self.settings.get_plate_settings()
        for channel_conf in self.settings.get_channels():
            self._start_channel_worker(channel_conf, reconnect_conf, plate_settings)

    def _start_channel_worker(
        self,
        channel_conf: Dict[str, Any],
        reconnect_conf: Optional[Dict[str, Any]] = None,
        plate_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        source = str(channel_conf.get("source", "")).strip()
        channel_name = channel_conf.get("name", "Канал")
        if not source:
            label = self.channel_labels.get(channel_name)
            if label:
                label.set_status("Нет источника")
            return
        reconnect_conf = reconnect_conf or self.settings.get_reconnect()
        plate_settings = plate_settings or self.settings.get_plate_settings()
        debug_settings = self.settings.get_debug_settings()
        worker = self._create_channel_worker(channel_conf, reconnect_conf, plate_settings, debug_settings)
        self.channel_workers.append(worker)
        worker.start()

    def _create_channel_worker(
        self,
        channel_conf: Dict[str, Any],
        reconnect_conf: Dict[str, Any],
        plate_settings: Dict[str, Any],
        debug_settings: Dict[str, Any],
    ) -> ChannelWorker:
        worker = ChannelWorker(
            channel_conf,
            self.settings.get_db_path(),
            self.settings.get_screenshot_dir(),
            reconnect_conf,
            plate_settings,
            debug_settings,
        )
        worker.frame_ready.connect(self._update_frame)
        worker.event_ready.connect(self._handle_event)
        worker.status_ready.connect(self._handle_status)
        worker.metrics_ready.connect(self._handle_metrics)
        return worker

    def _find_channel_worker(self, channel_id: int) -> Optional[ChannelWorker]:
        for worker in self.channel_workers:
            if worker.channel_id == channel_id:
                return worker
        return None

    def _restart_channel_worker(self, channel_conf: Dict[str, Any]) -> None:
        channel_id = int(channel_conf.get("id", 0))
        existing = self._find_channel_worker(channel_id)
        if existing:
            if channel_id in self._pending_channel_restarts:
                return
            self._pending_channel_restarts.add(channel_id)
            existing.stop()
            existing.finished.connect(
                lambda conf=channel_conf, worker=existing: self._finalize_channel_restart(
                    worker, conf
                )
            )
            if existing.isFinished():
                self._finalize_channel_restart(existing, channel_conf)
            return

        self._start_channel_worker(channel_conf)

    def _finalize_channel_restart(self, worker: ChannelWorker, channel_conf: Dict[str, Any]) -> None:
        channel_id = int(channel_conf.get("id", 0))
        try:
            worker.frame_ready.disconnect()
            worker.event_ready.disconnect()
            worker.status_ready.disconnect()
            worker.metrics_ready.disconnect()
        except Exception:
            pass
        if worker in self.channel_workers:
            self.channel_workers.remove(worker)
        worker.deleteLater()
        self._pending_channel_restarts.discard(channel_id)
        self._start_channel_worker(channel_conf)

    def _stop_channel_worker(self, worker: ChannelWorker) -> None:
        worker.stop()
        worker.finished.connect(lambda w=worker: self._finalize_channel_stop(w))
        if worker.isFinished():
            self._finalize_channel_stop(worker)

    def _finalize_channel_stop(self, worker: ChannelWorker) -> None:
        try:
            worker.frame_ready.disconnect()
            worker.event_ready.disconnect()
            worker.status_ready.disconnect()
            worker.metrics_ready.disconnect()
        except Exception:
            pass
        if worker in self.channel_workers:
            self.channel_workers.remove(worker)
        worker.deleteLater()

    def _stop_workers(self, *, shutdown_executor: bool = False) -> None:
        for worker in list(self.channel_workers):
            worker_name = getattr(getattr(worker, "config", None), "name", f"#{getattr(worker, 'channel_id', '?')}")
            worker.stop()
            finished = worker.wait(5000)
            if not finished:
                logger.warning(
                    "Канал %s не завершился за 5 секунд, выполняем принудительное завершение потока",
                    worker_name,
                )
                worker.terminate()
                worker.wait(1000)
            try:
                worker.frame_ready.disconnect()
                worker.event_ready.disconnect()
                worker.status_ready.disconnect()
                worker.metrics_ready.disconnect()
            except Exception:
                pass
        self.channel_workers = []
        self._pending_channel_restarts.clear()
        self._latest_frames.clear()
        if shutdown_executor:
            ChannelWorker.shutdown_executor()

    def _on_app_about_to_quit(self) -> None:
        self._disable_log_streaming()
        self._stop_workers(shutdown_executor=True)

    def _update_frame(self, channel_name: str, image: QtGui.QImage) -> None:
        cached_preview = image.scaled(
            QtCore.QSize(960, 540),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self._latest_frames[channel_name] = cached_preview
        if self._skip_frame_updates:
            return
        label = self.channel_labels.get(channel_name)
        if not label:
            return
        target_size = label.video_label.contentsRect().size()
        if target_size.isEmpty():
            return

        scaled_image = image.scaled(
            target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.set_pixmap(QtGui.QPixmap.fromImage(scaled_image))

    @staticmethod
    def _load_image_from_path(path: Optional[str]) -> Optional[QtGui.QImage]:
        if not path or not os.path.exists(path):
            return None
        image = cv2.imread(path)
        if image is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        bytes_per_line = 3 * width
        return QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

    @staticmethod
    def _image_weight(images: Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]) -> int:
        frame_image, plate_image = images
        frame_bytes = frame_image.byteCount() if frame_image else 0
        plate_bytes = plate_image.byteCount() if plate_image else 0
        return frame_bytes + plate_bytes

    def _discard_event_images(self, event_id: int) -> None:
        images = self.event_images.pop(event_id, None)
        if images:
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(images))

    def _store_event_images(
        self,
        event_id: int,
        images: Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]],
    ) -> None:
        existing = self.event_images.pop(event_id, None)
        if existing:
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(existing))

        frame_image, plate_image = images
        if frame_image is None and plate_image is None:
            return

        self.event_images[event_id] = (frame_image, plate_image)
        self.event_images.move_to_end(event_id)
        self._image_cache_bytes += self._image_weight((frame_image, plate_image))
        self._prune_image_cache()

    def _handle_event(self, event: Dict) -> None:
        event_id = int(event.get("id", 0))
        frame_image = event.get("frame_image")
        plate_image = event.get("plate_image")
        if event_id:
            self._store_event_images(event_id, (frame_image, plate_image))
            self.event_cache[event_id] = event
        channel_label = self.channel_labels.get(event.get("channel", ""))
        if channel_label:
            channel_label.set_last_plate(event.get("plate", ""))
        if event_id:
            self._insert_event_row(event, position=0)
            self._trim_events_table()
        else:
            self._refresh_events_table()
        self._show_event_details(event_id)

    def _cleanup_event_images(self, valid_ids: set[int]) -> None:
        for stale_id in list(self.event_images.keys()):
            if stale_id not in valid_ids:
                self._discard_event_images(stale_id)
        self._prune_image_cache()

    def _get_flag_icon(self, country: Optional[str]) -> Optional[QtGui.QIcon]:
        if not country:
            return None
        code = str(country).lower()
        if code in self.flag_cache:
            return self.flag_cache[code]
        flag_path = self.flag_dir / f"{code}.png"
        if flag_path.exists():
            icon = QtGui.QIcon(str(flag_path))
            self.flag_cache[code] = icon
            return icon
        self.flag_cache[code] = None
        return None

    def _get_country_name(self, country: Optional[str]) -> str:
        if not country:
            return "—"
        code = str(country).upper()
        return self.country_display_names.get(code, code)

    @staticmethod
    def _format_direction(direction: Optional[str]) -> str:
        if not direction:
            return "—"
        normalized = str(direction).upper()
        mapping = {"APPROACHING": "К камере", "RECEDING": "От камеры"}
        return mapping.get(normalized, "—")

    def _prune_image_cache(self) -> None:
        """Ограничивает размер кеша изображений, удаляя самые старые записи."""

        valid_ids = set(self.event_cache.keys())
        for event_id in list(self.event_images.keys()):
            if event_id not in valid_ids:
                self._discard_event_images(event_id)

        while self.event_images and (
            len(self.event_images) > self.MAX_IMAGE_CACHE
            or self._image_cache_bytes > self.MAX_IMAGE_CACHE_BYTES
        ):
            stale_id, images = self.event_images.popitem(last=False)
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(images))

    def _insert_event_row(self, event: Dict, position: Optional[int] = None) -> None:
        row_index = position if position is not None else self.events_table.rowCount()
        self.events_table.insertRow(row_index)

        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()
        timestamp = self._format_timestamp(event.get("timestamp", ""), target_zone, offset_minutes)
        plate = event.get("plate", "—")
        channel = event.get("channel", "—")
        event_id = int(event.get("id") or 0)
        country_code = (event.get("country") or "").upper()
        direction = self._format_direction(event.get("direction"))

        id_item = QtWidgets.QTableWidgetItem(timestamp)
        id_item.setData(QtCore.Qt.UserRole, event_id)
        self.events_table.setItem(row_index, 0, id_item)
        self.events_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(plate))
        country_item = QtWidgets.QTableWidgetItem("")
        country_item.setData(QtCore.Qt.UserRole, country_code)
        country_item.setData(QtCore.Qt.TextAlignmentRole, QtCore.Qt.AlignCenter)
        country_icon = self._get_flag_icon(event.get("country"))
        if country_icon:
            country_item.setData(QtCore.Qt.DecorationRole, country_icon)
        country_name = self._get_country_name(country_code)
        if country_name != "—":
            country_item.setToolTip(country_name)
        self.events_table.setItem(row_index, 2, country_item)
        self.events_table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(channel))
        self.events_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(direction))

    def _trim_events_table(self, max_rows: int = 200) -> None:
        while self.events_table.rowCount() > max_rows:
            last_row = self.events_table.rowCount() - 1
            item = self.events_table.item(last_row, 0)
            event_id = int(item.data(QtCore.Qt.UserRole) or 0) if item else 0
            self.events_table.removeRow(last_row)
            if event_id and event_id in self.event_cache:
                self.event_cache.pop(event_id, None)
            if event_id and event_id in self.event_images:
                self._discard_event_images(event_id)

    def _handle_status(self, channel: str, status: str) -> None:
        label = self.channel_labels.get(channel)
        if label:
            normalized = status.lower()
            if "движ" in normalized or "motion" in normalized:
                label.set_status("")
            else:
                label.set_status(status)
            label.set_motion_active("обнаружено" in normalized)

    def _handle_metrics(self, channel: str, metrics: Dict[str, Any]) -> None:
        label = self.channel_labels.get(channel)
        if not label:
            return
        fps = metrics.get("fps")
        latency_ms = metrics.get("latency_ms")
        accuracy = metrics.get("accuracy")

        fps_text = f"{fps:.1f}" if isinstance(fps, (int, float)) else "—"
        latency_text = f"{latency_ms:.0f} ms" if isinstance(latency_ms, (int, float)) else "—"
        accuracy_text = f"{accuracy:.0f}%" if isinstance(accuracy, (int, float)) else "—"
        label.set_metrics(f"FPS: {fps_text} | Lat: {latency_text} | Acc: {accuracy_text}")

    def _on_event_selected(self) -> None:
        row = self.events_table.currentRow()
        if row < 0:
            return
        event_id_item = self.events_table.item(row, 0)
        if event_id_item is None:
            return
        event_id = int(event_id_item.data(QtCore.Qt.UserRole) or 0)
        self._show_event_details(event_id)

    def _show_event_details(self, event_id: int) -> None:
        event = self.event_cache.get(event_id)
        images = self.event_images.get(event_id, (None, None))
        frame_image, plate_image = images
        if event:
            if frame_image is None and event.get("frame_path"):
                frame_image = self._load_image_from_path(event.get("frame_path"))
            if plate_image is None and event.get("plate_path"):
                plate_image = self._load_image_from_path(event.get("plate_path"))
            self._store_event_images(event_id, (frame_image, plate_image))
        display_event = dict(event) if event else None
        if display_event:
            target_zone = self._get_target_zone()
            offset_minutes = self.settings.get_time_offset_minutes()
            display_event["timestamp"] = self._format_timestamp(
                display_event.get("timestamp", ""), target_zone, offset_minutes
            )
            display_event["country"] = self._get_country_name(display_event.get("country"))
            display_event["direction"] = self._format_direction(display_event.get("direction"))
        self.event_detail.set_event(display_event, frame_image, plate_image)

    def _refresh_events_table(self, select_id: Optional[int] = None) -> None:
        rows = self.db.fetch_recent(limit=200)
        self.events_table.setRowCount(0)
        self.event_cache = {row["id"]: dict(row) for row in rows}
        valid_ids = set(self.event_cache.keys())

        for row_data in rows:
            self._insert_event_row(dict(row_data))

        self._cleanup_event_images(valid_ids)

        if select_id:
            for row in range(self.events_table.rowCount()):
                item = self.events_table.item(row, 0)
                if item and int(item.data(QtCore.Qt.UserRole) or 0) == select_id:
                    self.events_table.selectRow(row)
                    break

    # ------------------ Поиск ------------------
    def _build_search_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self._apply_stylesheet(widget, lambda: self.form_style)

        filters_group = QtWidgets.QGroupBox("Фильтры поиска")
        self._apply_stylesheet(filters_group, lambda: self.group_box_style)
        form = QtWidgets.QFormLayout(filters_group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        metrics = self.fontMetrics()
        input_width = metrics.horizontalAdvance("00.00.0000 00:00:00") + 40

        self.search_plate = QtWidgets.QLineEdit()
        self.search_plate.setMinimumWidth(input_width)
        self.search_plate.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self._apply_calendar_style(self.search_from)
        self._register_theme_setter(lambda: self._apply_calendar_style(self.search_from))
        self.search_from.setMinimumWidth(input_width)
        self.search_from.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)
        self._apply_calendar_style(self.search_to)
        self._register_theme_setter(lambda: self._apply_calendar_style(self.search_to))
        self.search_to.setMinimumWidth(input_width)
        self.search_to.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )

        self._reset_journal_range()

        form.addRow("Гос.номер:", self.search_plate)
        form.addRow("Дата с:", self.search_from)
        form.addRow("Дата по:", self.search_to)
        layout.addWidget(filters_group)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)
        search_btn = QtWidgets.QPushButton("Поиск")
        search_btn.clicked.connect(self._run_plate_search)
        self._polish_button(search_btn, 150)
        search_btn.setMinimumWidth(150)
        button_row.addWidget(search_btn)

        reset_btn = QtWidgets.QPushButton("Сбросить фильтр")
        reset_btn.clicked.connect(self._reset_journal_filters)
        self._polish_button(reset_btn, 180)
        reset_btn.setMinimumWidth(180)
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.search_table = QtWidgets.QTableWidget(0, 7)
        self.search_table.setHorizontalHeaderLabels(
            [
                "Дата/Время",
                "Канал",
                "Страна",
                "Направление",
                "Гос. номер",
                "Уверенность",
                "Источник",
            ]
        )
        header = self.search_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(90)
        self.search_table.setColumnWidth(0, 220)
        self.search_table.setColumnWidth(3, 140)
        self._apply_stylesheet(self.search_table, lambda: self.table_style)
        self.search_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.search_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.search_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.search_table.verticalHeader().setVisible(False)
        self.search_table.itemActivated.connect(self._on_journal_event_activated)
        layout.addWidget(self.search_table)

        self._run_plate_search()

        return widget

    def _reset_journal_range(self) -> None:
        today = QtCore.QDate.currentDate()
        start_of_day = QtCore.QDateTime(today, QtCore.QTime(0, 0))
        end_of_day = QtCore.QDateTime(today, QtCore.QTime(23, 59))
        self.search_from.setDateTime(start_of_day)
        self.search_to.setDateTime(end_of_day)

    def _reset_journal_filters(self) -> None:
        self.search_plate.clear()
        self._reset_journal_range()
        self._run_plate_search()

    def _run_plate_search(self) -> None:
        start = self._get_datetime_value(self.search_from)
        end = self._get_datetime_value(self.search_to)
        plate_fragment = self.search_plate.text().strip()
        if plate_fragment:
            rows = self.db.search_by_plate(
                plate_fragment, start=start or None, end=end or None, limit=200
            )
        else:
            rows = self.db.fetch_filtered(start=start or None, end=end or None, limit=50)

        self._populate_journal_table(rows)

    def _populate_journal_table(self, rows: List[Dict]) -> None:
        self.search_table.setRowCount(0)
        self.search_results.clear()
        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()

        for db_row in rows:
            row_data = dict(db_row)
            event_id = int(row_data["id"])
            self.search_results[event_id] = dict(row_data)
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            formatted_time = self._format_timestamp(
                row_data["timestamp"], target_zone, offset_minutes
            )
            time_item = QtWidgets.QTableWidgetItem(formatted_time)
            time_item.setData(QtCore.Qt.UserRole, event_id)
            self.search_table.setItem(row_index, 0, time_item)
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))

            country_item = QtWidgets.QTableWidgetItem(row_data.get("country") or "")
            country_icon = self._get_flag_icon(row_data.get("country"))
            if country_icon:
                country_item.setIcon(country_icon)
            country_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 2, country_item)

            direction = self._format_direction(row_data.get("direction"))
            direction_item = QtWidgets.QTableWidgetItem(direction)
            direction_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 3, direction_item)

            self.search_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 5, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 6, QtWidgets.QTableWidgetItem(row_data["source"]))

    def _on_journal_event_activated(self, item: QtWidgets.QTableWidgetItem) -> None:
        row = item.row()
        if row < 0:
            return
        self._open_journal_event(row)

    def _open_journal_event(self, row: int) -> None:
        event_id_item = self.search_table.item(row, 0)
        if event_id_item is None:
            return
        event_id = int(event_id_item.data(QtCore.Qt.UserRole) or 0)
        event = self.search_results.get(event_id)
        if not event:
            return

        frame_image = self._load_image_from_path(event.get("frame_path"))
        plate_image = self._load_image_from_path(event.get("plate_path"))

        display_event = dict(event)
        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()
        display_event["timestamp"] = self._format_timestamp(
            display_event.get("timestamp", ""), target_zone, offset_minutes
        )
        display_event["country"] = self._get_country_name(display_event.get("country"))
        display_event["direction"] = self._format_direction(display_event.get("direction"))

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Информация о событии")
        dialog.setMinimumSize(720, 640)
        dialog.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        dialog.setStyleSheet(
            f"QDialog {{ background-color: {self.colors['surface']}; border: 1px solid {self.colors['border']}; border-radius: 12px; }}"
            f"QGroupBox {{ background-color: {self.colors['panel']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 12px; padding: 10px; margin-top: 8px; }}"
            f"QLabel {{ color: {self.colors['text_primary']}; }}"
            f"QToolButton {{ background-color: transparent; border: none; color: {self.colors['text_primary']}; padding: 6px; }}"
            f"QToolButton:hover {{ background-color: rgba({QtGui.QColor(self.colors['text_primary']).red()}, {QtGui.QColor(self.colors['text_primary']).green()}, {QtGui.QColor(self.colors['text_primary']).blue()}, 20); border-radius: 6px; }}"
            f"QPushButton {{ background-color: {self.colors['accent']}; color: {self.colors['text_inverse']}; border: none; border-radius: 8px; padding: 8px 16px; font-weight: 700; }}"
            f"QPushButton:hover {{ background-color: {QtGui.QColor(self.colors['accent']).lighter(115).name()}; }}"
        )
        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 12)

        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(
            f"QFrame {{ background-color: {self.colors['header_bg']}; border-bottom: 1px solid {self.colors['border']}; border-top-left-radius: 12px; border-top-right-radius: 12px; }}"
        )
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(10)

        title = QtWidgets.QLabel("Информация о событии")
        title.setStyleSheet(f"color: {self.colors['text_primary']}; font-weight: 800;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        maximize_btn = QtWidgets.QToolButton()
        maximize_btn.setText("⛶")
        maximize_btn.setToolTip("Развернуть/свернуть окно")

        def toggle_maximize() -> None:
            if dialog.isMaximized():
                dialog.showNormal()
            else:
                dialog.showMaximized()

        maximize_btn.clicked.connect(toggle_maximize)
        header_layout.addWidget(maximize_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setText("✕")
        close_btn.setToolTip("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        header_layout.addWidget(close_btn)

        def start_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton:
                header._drag_pos = event.globalPos() - dialog.frameGeometry().topLeft()  # type: ignore[attr-defined]
                event.accept()

        def move_window(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.buttons() & QtCore.Qt.LeftButton and hasattr(header, "_drag_pos") and not dialog.isMaximized():
                dialog.move(event.globalPos() - header._drag_pos)  # type: ignore[attr-defined]
                event.accept()

        header.mousePressEvent = start_move  # type: ignore[assignment]
        header.mouseMoveEvent = move_window  # type: ignore[assignment]

        dialog_layout.addWidget(header)

        details = EventDetailView()
        details.set_theme(self.colors)
        details.set_event(display_event, frame_image, plate_image)
        dialog_layout.addWidget(details)

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.addStretch()
        bottom_close_btn = QtWidgets.QPushButton("Закрыть")
        bottom_close_btn.clicked.connect(dialog.accept)
        footer_row.addWidget(bottom_close_btn)
        dialog_layout.addLayout(footer_row)

        dialog.exec_()

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        content = QtWidgets.QFrame()
        self._apply_stylesheet(
            content,
            lambda: f"QFrame {{ background-color: {self.colors['surface']}; border: none; border-radius: 14px; }}",
        )
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        self.settings_nav = QtWidgets.QListWidget()
        self.settings_nav.setFixedWidth(220)
        self._apply_stylesheet(self.settings_nav, lambda: self.list_style)
        self.settings_nav.addItem("Общие")
        self.settings_nav.addItem("Каналы")
        self.settings_nav.addItem("Debug")
        content_layout.addWidget(self.settings_nav)

        self.settings_stack = QtWidgets.QStackedWidget()
        self.settings_stack.addWidget(self._build_general_settings_tab())
        self.settings_stack.addWidget(self._build_channel_settings_tab())
        self.settings_stack.addWidget(self._build_debug_settings_tab())
        content_layout.addWidget(self.settings_stack, 1)

        layout.addWidget(content, 1)

        self.settings_nav.currentRowChanged.connect(self.settings_stack.setCurrentIndex)
        self.settings_nav.setCurrentRow(0)
        self._load_general_settings()
        self._reload_channels_list()
        return widget

    def _build_general_settings_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._apply_stylesheet(
            scroll,
            lambda: (
                f"QScrollArea {{ background: transparent; border: none; }}"
                f"QScrollArea > QWidget > QWidget {{ background-color: {self.colors['surface']}; }}"
            ),
        )

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self._apply_stylesheet(widget, lambda: self.form_style)

        def make_section(title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QFormLayout]:
            frame = QtWidgets.QFrame()
            self._apply_stylesheet(
                frame,
                lambda: f"QFrame {{ background-color: {self.colors['panel']}; border: none; border-radius: 12px; }}",
            )
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(14, 12, 14, 12)
            frame_layout.setSpacing(10)

            header = QtWidgets.QLabel(title)
            self._apply_stylesheet(
                header,
                lambda: f"font-size: 14px; font-weight: 800; color: {self.colors['text_primary']};",
            )
            self._ensure_label_width(header)
            frame_layout.addWidget(header)

            form = QtWidgets.QFormLayout()
            self._tune_form_layout(form)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            frame_layout.addLayout(form)

            return frame, form

        reconnect_group, reconnect_form = make_section("Стабильность каналов")
        self.reconnect_on_loss_checkbox = QtWidgets.QCheckBox("Переподключение при потере сигнала")
        reconnect_form.addRow(self.reconnect_on_loss_checkbox)

        self.frame_timeout_input = QtWidgets.QSpinBox()
        self.frame_timeout_input.setMaximumWidth(140)
        self.frame_timeout_input.setRange(1, 300)
        self.frame_timeout_input.setSuffix(" с")
        self.frame_timeout_input.setToolTip("Сколько секунд ждать кадр перед попыткой переподключения")
        self.frame_timeout_input.setMinimumHeight(self.BUTTON_HEIGHT)
        reconnect_form.addRow("Таймаут ожидания кадра:", self.frame_timeout_input)

        self.retry_interval_input = QtWidgets.QSpinBox()
        self.retry_interval_input.setMaximumWidth(140)
        self.retry_interval_input.setRange(1, 300)
        self.retry_interval_input.setSuffix(" с")
        self.retry_interval_input.setToolTip("Интервал между попытками переподключения при потере сигнала")
        self.retry_interval_input.setMinimumHeight(self.BUTTON_HEIGHT)
        reconnect_form.addRow("Интервал между попытками:", self.retry_interval_input)

        self.periodic_reconnect_checkbox = QtWidgets.QCheckBox("Переподключение по таймеру")
        reconnect_form.addRow(self.periodic_reconnect_checkbox)

        self.periodic_interval_input = QtWidgets.QSpinBox()
        self.periodic_interval_input.setMaximumWidth(140)
        self.periodic_interval_input.setRange(1, 1440)
        self.periodic_interval_input.setSuffix(" мин")
        self.periodic_interval_input.setToolTip("Плановое переподключение каждые N минут")
        self.periodic_interval_input.setMinimumHeight(self.BUTTON_HEIGHT)
        reconnect_form.addRow("Интервал переподключения:", self.periodic_interval_input)

        storage_group, storage_form = make_section("Хранилище")
        storage_group.setMaximumWidth(self.FIELD_MAX_WIDTH + 220)
        storage_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)

        db_row = QtWidgets.QHBoxLayout()
        db_row.setContentsMargins(0, 0, 0, 0)
        db_row.setSpacing(8)
        self.db_dir_input = QtWidgets.QLineEdit()
        self._configure_line_edit(self.db_dir_input, self.FIELD_MAX_WIDTH)
        self.db_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_db_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_db_btn, 130)
        browse_db_btn.clicked.connect(self._choose_db_dir)
        db_row.addWidget(self.db_dir_input)
        db_row.addWidget(browse_db_btn)
        db_container = QtWidgets.QWidget()
        db_container.setLayout(db_row)
        db_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        db_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        storage_form.addRow("Папка БД:", db_container)

        screenshot_row = QtWidgets.QHBoxLayout()
        screenshot_row.setContentsMargins(0, 0, 0, 0)
        screenshot_row.setSpacing(8)
        self.screenshot_dir_input = QtWidgets.QLineEdit()
        self._configure_line_edit(self.screenshot_dir_input, self.FIELD_MAX_WIDTH)
        self.screenshot_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_screenshot_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_screenshot_btn, 130)
        browse_screenshot_btn.clicked.connect(self._choose_screenshot_dir)
        screenshot_row.addWidget(self.screenshot_dir_input)
        screenshot_row.addWidget(browse_screenshot_btn)
        screenshot_container = QtWidgets.QWidget()
        screenshot_container.setLayout(screenshot_row)
        screenshot_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        screenshot_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        storage_form.addRow("Папка для скриншотов:", screenshot_container)

        model_group, model_form = make_section("Модели")
        model_group.setMaximumWidth(self.FIELD_MAX_WIDTH + 220)

        self.device_combo = QtWidgets.QComboBox()
        self._configure_combo(self.device_combo)
        self._style_combo(self.device_combo)
        self.device_status_label = QtWidgets.QLabel("")
        self._apply_stylesheet(self.device_status_label, lambda: f"color: {self.colors['text_muted']};")
        self.device_status_label.setWordWrap(True)
        self._populate_device_options()
        model_form.addRow("Устройство инференса:", self.device_combo)
        model_form.addRow("", self.device_status_label)

        time_group, time_form = make_section("Дата и время")
        time_group.setMaximumWidth(self.FIELD_MAX_WIDTH + 220)

        self.timezone_combo = QtWidgets.QComboBox()
        self.timezone_combo.setEditable(False)
        self._configure_combo(self.timezone_combo)
        self._style_combo(self.timezone_combo)
        for label in self._available_offset_labels():
            self.timezone_combo.addItem(label, label)
        time_form.addRow("Часовой пояс:", self.timezone_combo)

        time_row = QtWidgets.QHBoxLayout()
        time_row.setSpacing(8)
        time_row.setContentsMargins(0, 0, 0, 0)
        self.time_correction_input = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self.time_correction_input.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        self.time_correction_input.setCalendarPopup(True)
        self.time_correction_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.time_correction_input.setMinimumWidth(self.FIELD_MIN_WIDTH)
        self._apply_calendar_style(self.time_correction_input)
        self._register_theme_setter(lambda: self._apply_calendar_style(self.time_correction_input))
        sync_now_btn = QtWidgets.QPushButton("Текущее время")
        self._polish_button(sync_now_btn, 160)
        sync_now_btn.clicked.connect(self._sync_time_now)
        time_row.addWidget(self.time_correction_input)
        time_row.addWidget(sync_now_btn)
        time_row.addStretch()
        time_container = QtWidgets.QWidget()
        time_container.setLayout(time_row)
        time_form.addRow("Коррекция времени:", time_container)

        plate_group, plate_form = make_section("Валидация номеров")

        plate_dir_row = QtWidgets.QHBoxLayout()
        plate_dir_row.setContentsMargins(0, 0, 0, 0)
        plate_dir_row.setSpacing(8)
        self.country_config_dir_input = QtWidgets.QLineEdit()
        self._configure_line_edit(self.country_config_dir_input, self.FIELD_MAX_WIDTH)
        self.country_config_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_country_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_country_btn, 130)
        browse_country_btn.clicked.connect(self._choose_country_dir)
        self.country_config_dir_input.editingFinished.connect(self._reload_country_templates)
        plate_dir_row.addWidget(self.country_config_dir_input)
        plate_dir_row.addWidget(browse_country_btn)
        plate_dir_container = QtWidgets.QWidget()
        plate_dir_container.setLayout(plate_dir_row)
        plate_dir_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        plate_dir_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        plate_form.addRow("Каталог шаблонов:", plate_dir_container)

        self.country_templates_list = QtWidgets.QListWidget()
        self.country_templates_list.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.country_templates_list.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.country_templates_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._register_theme_setter(
            lambda: self.country_templates_list.setStyleSheet(self.list_style)
        )
        plate_form.addRow("Активные страны:", self.country_templates_list)

        refresh_countries_btn = QtWidgets.QPushButton("Обновить список стран")
        self._polish_button(refresh_countries_btn, 180)
        refresh_countries_btn.clicked.connect(self._reload_country_templates)
        plate_form.addRow("", refresh_countries_btn)

        save_card = QtWidgets.QFrame()
        self._register_theme_setter(
            lambda f=save_card: f.setStyleSheet(
                f"QFrame {{ background-color: {self.colors['panel']}; border: none; border-radius: 12px; }}"
            )
        )
        save_row = QtWidgets.QHBoxLayout(save_card)
        save_row.setContentsMargins(14, 12, 14, 12)
        save_row.setSpacing(10)
        save_general_btn = QtWidgets.QPushButton("Сохранить")
        self._polish_button(save_general_btn, 220)
        save_general_btn.clicked.connect(self._save_general_settings)
        save_row.addWidget(save_general_btn, 0)
        save_row.addStretch(1)

        layout.addWidget(reconnect_group)
        layout.addWidget(storage_group)
        layout.addWidget(model_group)
        layout.addWidget(time_group)
        layout.addWidget(plate_group)
        layout.addWidget(save_card)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _build_channel_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        self._apply_stylesheet(widget, lambda: self.group_box_style)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(8)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)
        self.add_channel_btn = QtWidgets.QToolButton()
        self.add_channel_btn.setText("+")
        self.add_channel_btn.setToolTip("Добавить канал")
        self.add_channel_btn.setFixedSize(28, 28)
        self._style_channel_tool_button(self.add_channel_btn)
        self.add_channel_btn.clicked.connect(self._add_channel)
        toolbar.addWidget(self.add_channel_btn)

        self.remove_channel_btn = QtWidgets.QToolButton()
        self.remove_channel_btn.setText("-")
        self.remove_channel_btn.setToolTip("Удалить выбранный канал")
        self.remove_channel_btn.setFixedSize(28, 28)
        self._style_channel_tool_button(self.remove_channel_btn)
        self.remove_channel_btn.clicked.connect(self._remove_channel)
        toolbar.addWidget(self.remove_channel_btn)

        toolbar.addStretch(1)
        left_panel.addLayout(toolbar)

        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.setFixedWidth(180)
        self._apply_stylesheet(self.channels_list, lambda: self.list_style)
        self.channels_list.currentRowChanged.connect(self._on_channel_selected)
        left_panel.addWidget(self.channels_list)
        layout.addLayout(left_panel)

        self.channel_details_container = QtWidgets.QWidget()
        details_layout = QtWidgets.QHBoxLayout(self.channel_details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)

        center_panel = QtWidgets.QVBoxLayout()
        self.preview = ROIEditor()
        self.preview.roi_changed.connect(self._on_roi_drawn)
        self.preview.plate_size_selected.connect(self._on_plate_size_selected)
        self._register_theme_setter(lambda: self.preview.set_theme(self.colors))
        center_panel.addWidget(self.preview)
        details_layout.addLayout(center_panel, 2)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)

        tabs = QtWidgets.QTabWidget()
        self._apply_stylesheet(
            tabs,
            lambda: (
                "QTabBar { font-weight: 700; }"
                f"QTabBar::tab {{ background: {self.colors['field_bg']}; color: {self.colors['text_muted']}; padding: 8px 14px; border: 1px solid {self.colors['border']}; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 6px; }}"
                f"QTabBar::tab:selected {{ background: {self.colors['surface']}; color: {self.colors['accent']}; border: 1px solid {self.colors['border']}; border-bottom: 2px solid {self.colors['accent']}; }}"
                f"QTabWidget::pane {{ border: 1px solid {self.colors['border']}; border-radius: 10px; background-color: {self.colors['surface']}; top: -1px; }}"
                f"QWidget {{ background-color: {self.colors['surface']}; color: {self.colors['text_secondary']}; }}"
                f"QLabel {{ color: {self.colors['text_secondary']}; font-size: 13px; }}"
                f"QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color: {self.colors['field_bg']}; color: {self.colors['text_primary']}; border: 1px solid {self.colors['border']}; border-radius: 8px; padding: 8px; }}"
            ),
        )

        def make_form_tab() -> QtWidgets.QFormLayout:
            tab_widget = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab_widget)
            self._tune_form_layout(form)
            form.setContentsMargins(12, 12, 12, 12)
            tabs.addTab(tab_widget, "")
            return form

        channel_form = make_form_tab()
        tabs.setTabText(0, "Канал")
        self.channel_name_input = QtWidgets.QLineEdit()
        self._configure_line_edit(self.channel_name_input, self.FIELD_MAX_WIDTH)
        self.channel_source_input = QtWidgets.QLineEdit()
        self._configure_line_edit(self.channel_source_input, self.FIELD_MAX_WIDTH)
        channel_form.addRow("Название:", self.channel_name_input)
        channel_form.addRow("Источник/RTSP:", self.channel_source_input)

        recognition_form = make_form_tab()
        tabs.setTabText(1, "Распознавание")
        self.best_shots_input = QtWidgets.QSpinBox()
        self.best_shots_input.setRange(1, 50)
        self.best_shots_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.best_shots_input.setMinimumWidth(120)
        self.best_shots_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.best_shots_input.setToolTip("Количество бстшотов, участвующих в консенсусе трека")
        recognition_form.addRow("Бестшоты на трек:", self.best_shots_input)

        self.cooldown_input = QtWidgets.QSpinBox()
        self.cooldown_input.setRange(0, 3600)
        self.cooldown_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.cooldown_input.setMinimumWidth(120)
        self.cooldown_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.cooldown_input.setToolTip(
            "Интервал (в секундах), в течение которого не создается повторное событие для того же номера"
        )
        recognition_form.addRow("Пауза повтора (сек):", self.cooldown_input)

        self.min_conf_input = QtWidgets.QDoubleSpinBox()
        self.min_conf_input.setRange(0.0, 1.0)
        self.min_conf_input.setSingleStep(0.05)
        self.min_conf_input.setDecimals(2)
        self.min_conf_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_conf_input.setMinimumWidth(120)
        self.min_conf_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.min_conf_input.setToolTip(
            "Минимальная уверенность OCR (0-1) для приема результата; ниже — помечается как нечитаемое"
        )
        recognition_form.addRow("Мин. уверенность OCR:", self.min_conf_input)

        motion_form = make_form_tab()
        tabs.setTabText(2, "Детектор движения")
        self.detection_mode_input = QtWidgets.QComboBox()
        self.detection_mode_input.addItem("Постоянное", "continuous")
        self.detection_mode_input.addItem("Детектор движения", "motion")
        self.detection_mode_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.detection_mode_input.setMinimumWidth(self.FIELD_MIN_WIDTH)
        motion_form.addRow("Обнаружение ТС:", self.detection_mode_input)

        self.detector_stride_input = QtWidgets.QSpinBox()
        self.detector_stride_input.setRange(1, 12)
        self.detector_stride_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.detector_stride_input.setMinimumWidth(120)
        self.detector_stride_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.detector_stride_input.setToolTip(
            "Запускать YOLO на каждом N-м кадре в зоне распознавания, чтобы снизить нагрузку"
        )
        motion_form.addRow("Шаг инференса (кадр):", self.detector_stride_input)

        self.motion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.motion_threshold_input.setRange(0.0, 1.0)
        self.motion_threshold_input.setDecimals(3)
        self.motion_threshold_input.setSingleStep(0.005)
        self.motion_threshold_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_threshold_input.setMinimumWidth(120)
        self.motion_threshold_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.motion_threshold_input.setToolTip("Порог чувствительности по площади изменения внутри ROI")
        motion_form.addRow("Порог движения:", self.motion_threshold_input)

        self.motion_stride_input = QtWidgets.QSpinBox()
        self.motion_stride_input.setRange(1, 30)
        self.motion_stride_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_stride_input.setMinimumWidth(120)
        self.motion_stride_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.motion_stride_input.setToolTip("Обрабатывать каждый N-й кадр для поиска движения")
        motion_form.addRow("Частота анализа (кадр):", self.motion_stride_input)

        self.motion_activation_frames_input = QtWidgets.QSpinBox()
        self.motion_activation_frames_input.setRange(1, 60)
        self.motion_activation_frames_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_activation_frames_input.setMinimumWidth(120)
        self.motion_activation_frames_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.motion_activation_frames_input.setToolTip("Сколько кадров подряд должно быть движение, чтобы включить распознавание")
        motion_form.addRow("Мин. кадров с движением:", self.motion_activation_frames_input)

        self.motion_release_frames_input = QtWidgets.QSpinBox()
        self.motion_release_frames_input.setRange(1, 120)
        self.motion_release_frames_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_release_frames_input.setMinimumWidth(120)
        self.motion_release_frames_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.motion_release_frames_input.setToolTip("Сколько кадров без движения нужно, чтобы остановить распознавание")
        motion_form.addRow("Мин. кадров без движения:", self.motion_release_frames_input)

        roi_form = make_form_tab()
        tabs.setTabText(3, "Зона распознавания")

        self.size_filter_checkbox = QtWidgets.QCheckBox("Использовать фильтр по размеру")
        self.size_filter_checkbox.setChecked(True)
        self.size_filter_checkbox.setToolTip("При отключении детектор не будет отбрасывать рамки по размеру")
        self.size_filter_checkbox.toggled.connect(self._on_size_filter_toggled)
        roi_form.addRow("Фильтрация размеров:", self.size_filter_checkbox)

        size_group = QtWidgets.QGroupBox("Фильтр по размеру рамки")
        size_layout = QtWidgets.QGridLayout()
        size_layout.setHorizontalSpacing(14)
        size_layout.setVerticalSpacing(10)

        self.min_plate_width_input = QtWidgets.QSpinBox()
        self.min_plate_width_input.setRange(0, 5000)
        self.min_plate_width_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_plate_width_input.setMinimumWidth(120)
        self.min_plate_width_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.min_plate_width_input.setToolTip("Минимальная ширина рамки, меньшие детекции будут отброшены")

        self.min_plate_width_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.min_plate_height_input = QtWidgets.QSpinBox()
        self.min_plate_height_input.setRange(0, 3000)
        self.min_plate_height_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_plate_height_input.setMinimumWidth(120)
        self.min_plate_height_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.min_plate_height_input.setToolTip("Минимальная высота рамки, меньшие детекции будут отброшены")

        self.min_plate_height_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.max_plate_width_input = QtWidgets.QSpinBox()
        self.max_plate_width_input.setRange(0, 8000)
        self.max_plate_width_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.max_plate_width_input.setMinimumWidth(120)
        self.max_plate_width_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.max_plate_width_input.setToolTip("Максимальная ширина рамки, более крупные детекции будут отброшены")

        self.max_plate_width_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.max_plate_height_input = QtWidgets.QSpinBox()
        self.max_plate_height_input.setRange(0, 4000)
        self.max_plate_height_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.max_plate_height_input.setMinimumWidth(120)
        self.max_plate_height_input.setMinimumHeight(self.BUTTON_HEIGHT)
        self.max_plate_height_input.setToolTip("Максимальная высота рамки, более крупные детекции будут отброшены")

        self.max_plate_height_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        min_width_label = QtWidgets.QLabel("Мин. ширина (px):")
        min_height_label = QtWidgets.QLabel("Мин. высота (px):")
        max_width_label = QtWidgets.QLabel("Макс. ширина (px):")
        max_height_label = QtWidgets.QLabel("Макс. высота (px):")
        for label in (min_width_label, min_height_label, max_width_label, max_height_label):
            self._ensure_label_width(label)

        size_layout.addWidget(min_width_label, 0, 0)
        size_layout.addWidget(self.min_plate_width_input, 0, 1)
        size_layout.addWidget(min_height_label, 0, 2)
        size_layout.addWidget(self.min_plate_height_input, 0, 3)
        size_layout.addWidget(max_width_label, 1, 0)
        size_layout.addWidget(self.max_plate_width_input, 1, 1)
        size_layout.addWidget(max_height_label, 1, 2)
        size_layout.addWidget(self.max_plate_height_input, 1, 3)

        self.plate_size_hint = QtWidgets.QLabel(
            "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
        )
        self._apply_stylesheet(
            self.plate_size_hint,
            lambda: f"color: {self.colors['text_muted']}; padding-top: 6px;",
        )
        size_layout.addWidget(self.plate_size_hint, 2, 0, 1, 4)
        size_group.setLayout(size_layout)

        self._apply_stylesheet(size_group, lambda: self.group_box_style)
        roi_form.addRow("", size_group)

        self.roi_enabled_checkbox = QtWidgets.QCheckBox("Ограничивать поиск зоной ROI")
        self.roi_enabled_checkbox.setChecked(True)
        self.roi_enabled_checkbox.setToolTip("При отключении поиск номеров выполняется по всему кадру")
        self.roi_enabled_checkbox.toggled.connect(self._on_roi_usage_toggled)
        roi_form.addRow("Использование ROI:", self.roi_enabled_checkbox)

        roi_group = QtWidgets.QGroupBox("Точки ROI")
        self._apply_stylesheet(roi_group, lambda: self.group_box_style)
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        roi_layout.setContentsMargins(12, 12, 12, 12)
        roi_layout.setSpacing(10)

        self.roi_points_table = QtWidgets.QTableWidget()
        self.roi_points_table.setColumnCount(2)
        self.roi_points_table.setHorizontalHeaderLabels(["X (px)", "Y (px)"])
        header = self.roi_points_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        header.setDefaultSectionSize(90)
        header.setMinimumSectionSize(80)
        self._apply_stylesheet(self.roi_points_table, lambda: self.table_style)
        self.roi_points_table.verticalHeader().setVisible(False)
        self.roi_points_table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.roi_points_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.roi_points_table.setMaximumHeight(220)
        self.roi_points_table.itemChanged.connect(self._on_roi_table_changed)

        roi_buttons = QtWidgets.QHBoxLayout()
        self.add_point_btn = QtWidgets.QPushButton("Добавить точку")
        self._polish_button(self.add_point_btn, 150)
        self.add_point_btn.clicked.connect(self._add_roi_point)
        self.remove_point_btn = QtWidgets.QPushButton("Удалить точку")
        self._polish_button(self.remove_point_btn, 150)
        self.remove_point_btn.clicked.connect(self._remove_roi_point)
        self.clear_roi_btn = QtWidgets.QPushButton("Очистить зону")
        self._polish_button(self.clear_roi_btn, 150)
        self.clear_roi_btn.clicked.connect(self._clear_roi_points)
        roi_buttons.addWidget(self.add_point_btn)
        roi_buttons.addWidget(self.remove_point_btn)
        roi_buttons.addWidget(self.clear_roi_btn)

        roi_layout.addWidget(self.roi_points_table)
        roi_layout.addLayout(roi_buttons)
        self.roi_hint_label = QtWidgets.QLabel("Перетаскивайте вершины ROI на предпросмотре слева")
        self._apply_stylesheet(self.roi_hint_label, lambda: f"color: {self.colors['text_muted']}; padding-top: 2px;")
        roi_layout.addWidget(self.roi_hint_label)
        roi_form.addRow("", roi_group)

        right_panel.addWidget(tabs)

        refresh_btn = QtWidgets.QPushButton("Обновить кадр")
        self._polish_button(refresh_btn, self.ACTION_BUTTON_WIDTH)
        refresh_btn.clicked.connect(self._refresh_preview_frame)
        save_btn = QtWidgets.QPushButton("Сохранить канал")
        self._polish_button(save_btn, self.ACTION_BUTTON_WIDTH)
        save_btn.clicked.connect(self._save_channel)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(save_btn, 0, QtCore.Qt.AlignLeft)
        action_row.addWidget(refresh_btn, 0, QtCore.Qt.AlignLeft)
        action_row.addStretch(1)
        right_panel.addLayout(action_row)
        right_panel.addStretch()

        details_layout.addLayout(right_panel, 2)

        layout.addWidget(self.channel_details_container, 1)

        return widget

    def _channel_item_label(self, channel: Dict[str, Any]) -> str:
        return channel.get("name", "Канал")

    def _on_channel_selected(self, index: int) -> None:
        self._load_channel_form(index)
        self._update_channel_action_states()

    def _reload_channels_list(self, target_index: Optional[int] = None) -> None:
        channels = self.settings.get_channels()
        current_row = self.channels_list.currentRow()
        self.channels_list.blockSignals(True)
        self.channels_list.clear()
        for channel in channels:
            self.channels_list.addItem(self._channel_item_label(channel))
        self.channels_list.blockSignals(False)
        if self.channels_list.count():
            if target_index is None:
                target_index = current_row
            target_row = min(max(target_index if target_index is not None else 0, 0), self.channels_list.count() - 1)
            self.channels_list.setCurrentRow(target_row)
        else:
            self._set_channel_settings_visible(False)
        self._update_channel_action_states()

    def _set_channel_settings_visible(self, visible: bool) -> None:
        self.channel_details_container.setVisible(visible)
        self.channel_details_container.setEnabled(visible)
        if not visible:
            self._clear_channel_form()

    def _update_channel_action_states(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        has_selection = 0 <= index < len(channels)
        self.remove_channel_btn.setEnabled(has_selection)

    def _clear_channel_form(self) -> None:
        self.channel_name_input.clear()
        self.channel_source_input.clear()
        self.best_shots_input.setValue(self.settings.get_best_shots())
        self.cooldown_input.setValue(self.settings.get_cooldown_seconds())
        self.min_conf_input.setValue(self.settings.get_min_confidence())
        self.detection_mode_input.setCurrentIndex(
            max(0, self.detection_mode_input.findData("motion"))
        )
        self.detector_stride_input.setValue(2)
        self.motion_threshold_input.setValue(0.01)
        self.motion_stride_input.setValue(1)
        self.motion_activation_frames_input.setValue(3)
        self.motion_release_frames_input.setValue(6)
        self.roi_enabled_checkbox.blockSignals(True)
        self.roi_enabled_checkbox.setChecked(True)
        self.roi_enabled_checkbox.blockSignals(False)
        self.size_filter_checkbox.blockSignals(True)
        self.size_filter_checkbox.setChecked(True)
        self.size_filter_checkbox.blockSignals(False)
        size_defaults = self.settings.get_plate_size_defaults()
        min_size = size_defaults.get("min_plate_size", {})
        max_size = size_defaults.get("max_plate_size", {})
        self.min_plate_width_input.setValue(int(min_size.get("width", 0)))
        self.min_plate_height_input.setValue(int(min_size.get("height", 0)))
        self.max_plate_width_input.setValue(int(max_size.get("width", 0)))
        self.max_plate_height_input.setValue(int(max_size.get("height", 0)))
        self.plate_size_hint.setText(
            "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
        )
        self.preview.set_plate_sizes(
            self.min_plate_width_input.value(),
            self.min_plate_height_input.value(),
            self.max_plate_width_input.value(),
            self.max_plate_height_input.value(),
        )
        self._on_size_filter_toggled(True)
        default_roi = self._default_roi_region()
        self.preview.setPixmap(None)
        self.preview.set_roi(default_roi)
        self._sync_roi_table(default_roi)
        self._on_roi_usage_toggled(True)

    def _load_general_settings(self) -> None:
        reconnect = self.settings.get_reconnect()
        signal_loss = reconnect.get("signal_loss", {})
        periodic = reconnect.get("periodic", {})
        self.db_dir_input.setText(self.settings.get_db_dir())
        self.screenshot_dir_input.setText(self.settings.get_screenshot_dir())
        plate_settings = self.settings.get_plate_settings()
        self.country_config_dir_input.setText(plate_settings.get("config_dir", "config/countries"))
        self._reload_country_templates(plate_settings.get("enabled_countries", []))

        self.reconnect_on_loss_checkbox.setChecked(bool(signal_loss.get("enabled", True)))
        self.frame_timeout_input.setValue(int(signal_loss.get("frame_timeout_seconds", 5)))
        self.retry_interval_input.setValue(int(signal_loss.get("retry_interval_seconds", 5)))

        self.periodic_reconnect_checkbox.setChecked(bool(periodic.get("enabled", False)))
        self.periodic_interval_input.setValue(int(periodic.get("interval_minutes", 60)))

        model_settings = self.settings.get_model_settings()
        device_value = str(model_settings.get("device") or "cpu").strip().lower()
        if device_value == "gpu":
            device_value = "cuda"
        cuda_available = torch.cuda.is_available()
        if device_value.startswith("cuda") and not cuda_available:
            device_value = "cpu"
            self.device_status_label.setText("GPU не обнаружена. Выбрано использование CPU.")
        else:
            self._update_device_status_label(cuda_available)
        index = self.device_combo.findData(device_value)
        if index < 0:
            index = self.device_combo.findData("cpu")
        if index >= 0:
            self.device_combo.setCurrentIndex(index)

        time_settings = self.settings.get_time_settings()
        tz_value = time_settings.get("timezone") or "UTC+00:00"
        offset_label = tz_value
        parsed_offset = self._parse_utc_offset_minutes(tz_value)
        if parsed_offset is None:
            try:
                qtz = QtCore.QTimeZone(tz_value.encode())
                parsed_offset = qtz.offsetFromUtc(QtCore.QDateTime.currentDateTime()) // 60
            except Exception:
                parsed_offset = 0
        if parsed_offset is not None:
            offset_label = self._offset_label(parsed_offset)
        index = self.timezone_combo.findData(offset_label)
        if index < 0:
            self.timezone_combo.addItem(offset_label, offset_label)
            index = self.timezone_combo.findData(offset_label)
        if index >= 0:
            self.timezone_combo.setCurrentIndex(index)
        offset_minutes = int(time_settings.get("offset_minutes", 0) or 0)
        adjusted_dt = QtCore.QDateTime.currentDateTime().addSecs(offset_minutes * 60)
        self.time_correction_input.setDateTime(adjusted_dt)
        self._load_debug_settings()

    def _load_debug_settings(self) -> None:
        debug_settings = self.settings.get_debug_settings()
        self._debug_settings_cache = debug_settings
        self._apply_debug_settings_to_ui()
        self._set_log_panel_visible(debug_settings.get("log_panel_enabled", False))

    def _sync_time_now(self) -> None:
        self.time_correction_input.setDateTime(QtCore.QDateTime.currentDateTime())

    def _choose_screenshot_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки для скриншотов")
        if directory:
            self.screenshot_dir_input.setText(directory)

    def _choose_db_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки базы данных")
        if directory:
            self.db_dir_input.setText(directory)

    def _choose_country_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор каталога шаблонов номеров")
        if directory:
            self.country_config_dir_input.setText(directory)
            self._reload_country_templates()

    def _load_country_names(self, config_dir: Optional[str] = None) -> Dict[str, str]:
        config_path = config_dir or self.settings.get_plate_settings().get("config_dir", "config/countries")
        loader = CountryConfigLoader(config_path)
        loader.ensure_dir()
        names: Dict[str, str] = {}
        for cfg in loader.available_configs():
            code = (cfg.get("code") or "").upper()
            name = cfg.get("name") or code
            if code:
                names[code] = name
        return names

    def _reload_country_templates(self, enabled: Optional[List[str]] = None) -> None:
        plate_settings = self.settings.get_plate_settings()
        config_dir = self.country_config_dir_input.text().strip() or plate_settings.get("config_dir", "config/countries")
        loader = CountryConfigLoader(config_dir)
        loader.ensure_dir()
        available = loader.available_configs()
        enabled_codes = set(enabled or plate_settings.get("enabled_countries", []))
        self.country_display_names = {cfg["code"].upper(): cfg.get("name") or cfg["code"] for cfg in available if cfg.get("code")}

        self.country_templates_list.clear()
        if not available:
            item = QtWidgets.QListWidgetItem("Конфигурации стран не найдены")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.country_templates_list.addItem(item)
            return

        for cfg in available:
            item = QtWidgets.QListWidgetItem(f"{cfg['code']} — {cfg['name']}")
            item.setData(QtCore.Qt.UserRole, cfg["code"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if cfg["code"] in enabled_codes else QtCore.Qt.Unchecked)
            self.country_templates_list.addItem(item)

    def _collect_enabled_countries(self) -> List[str]:
        codes: List[str] = []
        for idx in range(self.country_templates_list.count()):
            item = self.country_templates_list.item(idx)
            if item and item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState() == QtCore.Qt.Checked:
                codes.append(str(item.data(QtCore.Qt.UserRole)))
        return codes

    def _save_general_settings(self) -> None:
        reconnect = {
            "signal_loss": {
                "enabled": self.reconnect_on_loss_checkbox.isChecked(),
                "frame_timeout_seconds": int(self.frame_timeout_input.value()),
                "retry_interval_seconds": int(self.retry_interval_input.value()),
            },
            "periodic": {
                "enabled": self.periodic_reconnect_checkbox.isChecked(),
                "interval_minutes": int(self.periodic_interval_input.value()),
            },
        }
        self.settings.save_reconnect(reconnect)
        db_dir = self.db_dir_input.text().strip() or "data/db"
        os.makedirs(db_dir, exist_ok=True)
        self.settings.save_db_dir(db_dir)
        screenshot_dir = self.screenshot_dir_input.text().strip() or "data/screenshots"
        self.settings.save_screenshot_dir(screenshot_dir)
        os.makedirs(screenshot_dir, exist_ok=True)
        tz_name = self.timezone_combo.currentData() or self.timezone_combo.currentText().strip() or "UTC+00:00"
        offset_minutes = int(
            QtCore.QDateTime.currentDateTime().secsTo(self.time_correction_input.dateTime()) / 60
        )
        self.settings.save_time_settings({"timezone": tz_name, "offset_minutes": offset_minutes})
        plate_settings = {
            "config_dir": self.country_config_dir_input.text().strip() or "config/countries",
            "enabled_countries": self._collect_enabled_countries(),
        }
        os.makedirs(plate_settings["config_dir"], exist_ok=True)
        self.settings.save_plate_settings(plate_settings)
        device_value = self.device_combo.currentData() or "cpu"
        self.settings.save_model_device(device_value)
        self.db = EventDatabase(self.settings.get_db_path())
        self._refresh_events_table()
        self._start_channels()

    def _populate_device_options(self) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("CPU", "cpu")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_label = f"GPU (CUDA) — {gpu_name}"
        else:
            gpu_label = "GPU (CUDA) — недоступно"
        self.device_combo.addItem(gpu_label, "cuda")
        if not cuda_available:
            model = self.device_combo.model()
            gpu_index = self.device_combo.count() - 1
            if model is not None and gpu_index >= 0:
                item = model.item(gpu_index)
                if item is not None:
                    item.setEnabled(False)
        self._update_device_status_label(cuda_available)

    def _update_device_status_label(self, cuda_available: Optional[bool] = None) -> None:
        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            self.device_status_label.setText(f"Доступна GPU: {gpu_name}")
        else:
            self.device_status_label.setText("GPU не обнаружена. Используйте CPU.")

    def _load_channel_form(self, index: int) -> None:
        channels = self.settings.get_channels()
        has_channel = 0 <= index < len(channels)
        self._set_channel_settings_visible(has_channel)
        if has_channel:
            channel = channels[index]
            self.channel_name_input.setText(channel.get("name", ""))
            self.channel_source_input.setText(channel.get("source", ""))
            self.best_shots_input.setValue(int(channel.get("best_shots", self.settings.get_best_shots())))
            self.cooldown_input.setValue(int(channel.get("cooldown_seconds", self.settings.get_cooldown_seconds())))
            self.min_conf_input.setValue(float(channel.get("ocr_min_confidence", self.settings.get_min_confidence())))
            self.detection_mode_input.setCurrentIndex(
                max(0, self.detection_mode_input.findData(channel.get("detection_mode", "motion")))
            )
            self.detector_stride_input.setValue(int(channel.get("detector_frame_stride", 2)))
            self.motion_threshold_input.setValue(float(channel.get("motion_threshold", 0.01)))
            self.motion_stride_input.setValue(int(channel.get("motion_frame_stride", 1)))
            self.motion_activation_frames_input.setValue(int(channel.get("motion_activation_frames", 3)))
            self.motion_release_frames_input.setValue(int(channel.get("motion_release_frames", 6)))
            min_size = channel.get("min_plate_size", self.settings.get_plate_size_defaults().get("min_plate_size", {}))
            max_size = channel.get("max_plate_size", self.settings.get_plate_size_defaults().get("max_plate_size", {}))
            self.min_plate_width_input.setValue(int(min_size.get("width", 0)))
            self.min_plate_height_input.setValue(int(min_size.get("height", 0)))
            self.max_plate_width_input.setValue(int(max_size.get("width", 0)))
            self.max_plate_height_input.setValue(int(max_size.get("height", 0)))
            self.plate_size_hint.setText(
                "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
            )
            size_filter_enabled = bool(channel.get("size_filter_enabled", True))
            self.size_filter_checkbox.blockSignals(True)
            self.size_filter_checkbox.setChecked(size_filter_enabled)
            self.size_filter_checkbox.blockSignals(False)
            self._sync_plate_rects_from_inputs()
            self._on_size_filter_toggled(size_filter_enabled)

            region = channel.get("region") or self._default_roi_region()
            if not region.get("points"):
                region = {"unit": region.get("unit", "px"), "points": self._default_roi_region()["points"]}
            self.preview.setPixmap(None)
            self.preview.set_roi(region)
            self._sync_roi_table(region)
            roi_enabled = bool(channel.get("roi_enabled", True))
            self.roi_enabled_checkbox.blockSignals(True)
            self.roi_enabled_checkbox.setChecked(roi_enabled)
            self.roi_enabled_checkbox.blockSignals(False)
            self._on_roi_usage_toggled(roi_enabled)

    def _add_channel(self) -> None:
        channels = self.settings.get_channels()
        new_id = max([c.get("id", 0) for c in channels] + [0]) + 1
        channels.append(
            {
                "id": new_id,
                "name": f"Канал {new_id}",
                "source": "",
                "best_shots": self.settings.get_best_shots(),
                "cooldown_seconds": self.settings.get_cooldown_seconds(),
                "ocr_min_confidence": self.settings.get_min_confidence(),
                "region": self._default_roi_region(),
                "detection_mode": "motion",
                "detector_frame_stride": 2,
                "motion_threshold": 0.01,
                "motion_frame_stride": 1,
                "motion_activation_frames": 3,
                "motion_release_frames": 6,
                "roi_enabled": True,
                "min_plate_size": self.settings.get_plate_size_defaults().get("min_plate_size"),
                "max_plate_size": self.settings.get_plate_size_defaults().get("max_plate_size"),
                "size_filter_enabled": True,
            }
        )
        self.settings.save_channels(channels)
        self._reload_channels_list(len(channels) - 1)
        self._draw_grid()
        self._start_channels()

    def _remove_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels.pop(index)
            self.settings.save_channels(channels)
            self._reload_channels_list(index)
            self._draw_grid()
            self._start_channels()

    def _save_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            try:
                channel_id = channels[index].get("id")
                previous_name = channels[index].get("name", "Канал")
                previous_source = str(channels[index].get("source", "")).strip()
                logger.info("Сохранение настроек канала: id=%s", channel_id)
                channels[index]["name"] = self.channel_name_input.text()
                channels[index]["source"] = self.channel_source_input.text()
                channels[index]["best_shots"] = int(self.best_shots_input.value())
                channels[index]["cooldown_seconds"] = int(self.cooldown_input.value())
                channels[index]["ocr_min_confidence"] = float(self.min_conf_input.value())
                channels[index]["detection_mode"] = self.detection_mode_input.currentData()
                channels[index]["detector_frame_stride"] = int(self.detector_stride_input.value())
                channels[index]["motion_threshold"] = float(self.motion_threshold_input.value())
                channels[index]["motion_frame_stride"] = int(self.motion_stride_input.value())
                channels[index]["motion_activation_frames"] = int(self.motion_activation_frames_input.value())
                channels[index]["motion_release_frames"] = int(self.motion_release_frames_input.value())
                channels[index]["roi_enabled"] = self.roi_enabled_checkbox.isChecked()
                channels[index]["min_plate_size"] = {
                    "width": int(self.min_plate_width_input.value()),
                    "height": int(self.min_plate_height_input.value()),
                }
                channels[index]["max_plate_size"] = {
                    "width": int(self.max_plate_width_input.value()),
                    "height": int(self.max_plate_height_input.value()),
                }
                channels[index]["size_filter_enabled"] = self.size_filter_checkbox.isChecked()
                channels[index]["region"] = self._build_region_payload()
                channels[index].pop("debug", None)

                debug_settings = self._save_debug_settings()

                self.settings.save_channels(channels)
                logger.info(
                    "Настройки канала сохранены: id=%s name=%s",
                    channel_id,
                    channels[index].get("name", "Канал"),
                )
                self._reload_channels_list(index)
                self._draw_grid()
                existing = self._find_channel_worker(int(channel_id or 0))
                new_source = str(channels[index].get("source", "")).strip()
                if existing:
                    if not new_source:
                        self._stop_channel_worker(existing)
                    else:
                        if previous_name != channels[index].get("name"):
                            self._latest_frames.pop(previous_name, None)
                        if new_source != previous_source:
                            self._restart_channel_worker(channels[index])
                        else:
                            existing.update_runtime_config(
                                channels[index],
                                self.settings.get_reconnect(),
                                self.settings.get_plate_settings(),
                                debug_settings,
                            )
                elif new_source:
                    self._start_channel_worker(channels[index])
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось сохранить настройки канала")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Ошибка",
                    f"Не удалось сохранить настройки канала: {exc}",
                )

    def _on_debug_settings_changed(self) -> None:
        self._save_debug_settings()

    def _save_debug_settings(self) -> Dict[str, Any]:
        debug_settings = {
            "show_detection_boxes": self.debug_detection_global_checkbox.isChecked(),
            "show_ocr_text": self.debug_ocr_global_checkbox.isChecked(),
            "show_direction_tracks": self.debug_direction_global_checkbox.isChecked(),
            "log_panel_enabled": self.debug_log_checkbox.isChecked(),
        }
        self._debug_settings_cache = debug_settings
        self.settings.save_debug_settings(debug_settings)
        self._apply_debug_settings_to_workers(debug_settings)
        self._set_log_panel_visible(debug_settings.get("log_panel_enabled", False))
        return debug_settings

    def _apply_debug_settings_to_workers(self, debug_settings: Dict[str, Any]) -> None:
        reconnect_conf = self.settings.get_reconnect()
        plate_settings = self.settings.get_plate_settings()
        channels = {int(channel.get("id", 0)): channel for channel in self.settings.get_channels()}
        for worker in list(self.channel_workers):
            channel_conf = channels.get(worker.channel_id)
            if channel_conf is None:
                continue
            worker.update_runtime_config(
                channel_conf,
                reconnect_conf,
                plate_settings,
                debug_settings,
            )

    def _apply_debug_settings_to_ui(self) -> None:
        if not self._debug_settings_cache:
            return
        if not hasattr(self, "debug_detection_global_checkbox"):
            return
        mapping = (
            (self.debug_detection_global_checkbox, "show_detection_boxes"),
            (self.debug_ocr_global_checkbox, "show_ocr_text"),
            (self.debug_direction_global_checkbox, "show_direction_tracks"),
            (self.debug_log_checkbox, "log_panel_enabled"),
        )
        for checkbox, key in mapping:
            if checkbox is None:
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(self._debug_settings_cache.get(key, False)))
            checkbox.blockSignals(False)

    def _sync_roi_table(self, roi: Dict[str, Any]) -> None:
        points = roi.get("points") or []
        unit = str(roi.get("unit", "px")).lower()
        if unit == "percent":
            img_size = self.preview.image_size()
            if img_size:
                points = [
                    {
                        "x": float(point.get("x", 0.0)) * img_size.width() / 100.0,
                        "y": float(point.get("y", 0.0)) * img_size.height() / 100.0,
                    }
                    for point in points
                ]
        self.roi_points_table.blockSignals(True)
        self.roi_points_table.setRowCount(len(points))
        for row, point in enumerate(points):
            x_item = QtWidgets.QTableWidgetItem(f"{float(point.get('x', 0.0)):.2f}")
            y_item = QtWidgets.QTableWidgetItem(f"{float(point.get('y', 0.0)):.2f}")
            self.roi_points_table.setItem(row, 0, x_item)
            self.roi_points_table.setItem(row, 1, y_item)
        self.roi_points_table.blockSignals(False)

    def _collect_roi_points_from_table(self) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for row in range(self.roi_points_table.rowCount()):
            x_item = self.roi_points_table.item(row, 0)
            y_item = self.roi_points_table.item(row, 1)
            try:
                x_val = float(x_item.text()) if x_item else 0.0
                y_val = float(y_item.text()) if y_item else 0.0
            except ValueError:
                continue
            points.append({"x": x_val, "y": y_val})
        return points

    def _on_roi_drawn(self, roi: Dict[str, Any]) -> None:
        self._sync_roi_table(roi)

    def _on_roi_table_changed(self) -> None:
        roi = {"unit": "px", "points": self._collect_roi_points_from_table()}
        self.preview.set_roi(roi)

    def _build_region_payload(self) -> Dict[str, Any]:
        points = self._collect_roi_points_from_table()
        image_size = self.preview.image_size()
        if image_size and image_size.width() > 0 and image_size.height() > 0:
            width = float(image_size.width())
            height = float(image_size.height())
            normalized = [
                {
                    "x": round(max(0.0, min(100.0, (point.get("x", 0.0) / width) * 100.0)), 2),
                    "y": round(max(0.0, min(100.0, (point.get("y", 0.0) / height) * 100.0)), 2),
                }
                for point in points
            ]
            return {"unit": "percent", "points": normalized}
        return {"unit": "px", "points": [{"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))} for p in points]}

    def _add_roi_point(self) -> None:
        img_size = self.preview.image_size()
        x = img_size.width() // 2 if img_size else 0
        y = img_size.height() // 2 if img_size else 0
        row = self.roi_points_table.rowCount()
        self.roi_points_table.blockSignals(True)
        self.roi_points_table.insertRow(row)
        self.roi_points_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(x)))
        self.roi_points_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(y)))
        self.roi_points_table.blockSignals(False)
        self._on_roi_table_changed()

    def _remove_roi_point(self) -> None:
        rows = sorted({index.row() for index in self.roi_points_table.selectedIndexes()}, reverse=True)
        if not rows and self.roi_points_table.rowCount():
            rows = [self.roi_points_table.rowCount() - 1]
        self.roi_points_table.blockSignals(True)
        for row in rows:
            self.roi_points_table.removeRow(row)
        self.roi_points_table.blockSignals(False)
        self._on_roi_table_changed()

    def _clear_roi_points(self) -> None:
        default_roi = self._default_roi_region()
        self._sync_roi_table(default_roi)
        self.preview.set_roi(default_roi)

    def _sync_plate_rects_from_inputs(self) -> None:
        self.preview.set_plate_sizes(
            self.min_plate_width_input.value(),
            self.min_plate_height_input.value(),
            self.max_plate_width_input.value(),
            self.max_plate_height_input.value(),
        )

    def _on_size_filter_toggled(self, enabled: bool) -> None:
        widgets = (
            self.min_plate_width_input,
            self.min_plate_height_input,
            self.max_plate_width_input,
            self.max_plate_height_input,
            self.plate_size_hint,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self.preview.set_size_overlay_enabled(enabled)
        if enabled:
            self._sync_plate_rects_from_inputs()
            self.plate_size_hint.setText(
                "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
            )
        else:
            self.plate_size_hint.setText("Фильтр выключен — рамки не ограничиваются по размеру")

    def _on_roi_usage_toggled(self, enabled: bool) -> None:
        widgets = (
            self.roi_points_table,
            self.add_point_btn,
            self.remove_point_btn,
            self.clear_roi_btn,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self.preview.set_roi_usage_enabled(enabled)
        if enabled:
            self.roi_hint_label.setText("Перетаскивайте вершины ROI на предпросмотре слева")
        else:
            self.roi_hint_label.setText("ROI отключена — поиск номеров идёт по всему кадру")

    def _on_plate_size_selected(self, target: str, width: int, height: int) -> None:
        if target == "min":
            self.min_plate_width_input.setValue(width)
            self.min_plate_height_input.setValue(height)
            label = "мин"
        else:
            self.max_plate_width_input.setValue(width)
            self.max_plate_height_input.setValue(height)
            label = "макс"
        self.plate_size_hint.setText(
            f"Прямоугольник {label}: {width}×{height} px"
        )

    def _refresh_preview_frame(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if not (0 <= index < len(channels)):
            return
        channel_name = channels[index].get("name", "Канал")
        preview_image = self._latest_frames.get(channel_name)
        if preview_image is None:
            self.preview.setPixmap(None)
            return
        self.preview.setPixmap(QtGui.QPixmap.fromImage(preview_image))

    # ------------------ Жизненный цикл ------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_workers(shutdown_executor=True)
        event.accept()
