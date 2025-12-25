# /anpr/preprocessing/plate_preprocessor.py
"""Предобработка изображений номера перед OCR."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PlatePreprocessor:
    """Выполняет коррекцию перспективы и наклона для кропа номера."""

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <= 0:
            return image
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32"
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    def _rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        (height, width) = image.shape[:2]
        if height == 0 or width == 0:
            return image
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(image, matrix, (new_width, new_height))

    def _detect_plate_quadrilateral(self, binary: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = float(binary.shape[0] * binary.shape[1])
        min_area = image_area * 0.1
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in candidates[:10]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rect = cv2.boundingRect(approx)
                width = rect[2]
                height = rect[3]
                if width == 0 or height == 0:
                    continue
                aspect_ratio = width / float(height)
                if 1.3 <= aspect_ratio <= 7.0:
                    return approx.reshape(4, 2)

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        width = rect[1][0]
        height = rect[1][1]
        if width == 0 or height == 0:
            return None
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.3 or aspect_ratio > 7.0:
            return None
        if cv2.contourArea(largest) < min_area:
            return None
        return box.astype("float32")

    def _estimate_skew_angle(self, gray: np.ndarray, binary: np.ndarray) -> tuple[float, float]:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges = cv2.dilate(edges, None, iterations=1)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=0.4 * gray.shape[1], maxLineGap=15
        )
        if lines is not None:
            angles = []
            weights = []
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < -90:
                    angle += 180
                if angle > 90:
                    angle -= 180
                if abs(angle) > 45:
                    continue
                length = np.hypot(dx, dy)
                angles.append(angle)
                weights.append(length)
            if angles:
                angles_array = np.array(angles)
                weights_array = np.array(weights)
                median_angle = float(np.average(angles_array, weights=weights_array))
                spread = float(np.std(angles_array))
                count_score = min(1.0, len(angles_array) / 6.0)
                spread_score = max(0.0, 1.0 - (spread / 15.0))
                confidence = count_score * spread_score
                return median_angle, confidence

        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        width, height = rect[1]
        if width == 0 or height == 0:
            return 0.0, 0.0
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.3 or aspect_ratio > 7.0:
            return 0.0, 0.0
        angle = rect[2]
        if width < height:
            angle = angle + 90
        return float(angle), 0.35

    def _prepare_gray_and_binary(self, plate_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return blurred, cleaned

    def _preprocess_with_mask(self, plate_image: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if plate_image.size == 0:
            return plate_image, None

        blurred, cleaned = self._prepare_gray_and_binary(plate_image)

        quadrilateral = self._detect_plate_quadrilateral(cleaned)
        if quadrilateral is not None:
            return (
                self._four_point_transform(plate_image, quadrilateral),
                self._four_point_transform(cleaned, quadrilateral),
            )

        angle, confidence = self._estimate_skew_angle(blurred, cleaned)
        if confidence < 0.35:
            return plate_image, cleaned
        if abs(angle) < 5:
            return plate_image, cleaned
        if abs(angle) > 45:
            return plate_image, cleaned
        return self._rotate_bound(plate_image, angle), self._rotate_bound(cleaned, angle)

    def _sort_boxes(
        self, boxes: List[Tuple[int, int, int, int]], row_merge_threshold: float = 0.6
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []

        heights = [h for (_, _, _, h) in boxes]
        median_height = float(np.median(heights)) if heights else 0.0
        row_threshold = max(4.0, median_height * max(0.1, row_merge_threshold))

        rows: List[List[Tuple[int, int, int, int]]] = []
        for box in sorted(boxes, key=lambda b: b[1]):
            x, y, _, h = box
            center_y = y + h / 2.0
            placed = False
            for row in rows:
                row_centers = [item[1] + item[3] / 2.0 for item in row]
                row_center = float(np.mean(row_centers)) if row_centers else center_y
                if abs(center_y - row_center) <= row_threshold:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        ordered_rows = sorted(rows, key=lambda r: min(item[1] for item in r))
        for row in ordered_rows:
            row.sort(key=lambda b: b[0])

        flattened: List[Tuple[int, int, int, int]] = []
        for row in ordered_rows:
            flattened.extend(row)
        return flattened

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        processed, _ = self._preprocess_with_mask(plate_image)
        return processed

    def extract_components(self, plate_image: np.ndarray, segmentation_config: Dict[str, Any]) -> tuple[np.ndarray, List[np.ndarray]]:
        processed, binary = self._preprocess_with_mask(plate_image)
        if processed.size == 0 or binary is None:
            return processed, []

        height, width = binary.shape[:2]
        plate_area = float(max(1, height * width))

        min_area_ratio = float(segmentation_config.get("min_symbol_area_ratio", 0.0025))
        max_area_ratio = float(segmentation_config.get("max_symbol_area_ratio", 0.18))
        min_aspect = float(segmentation_config.get("min_symbol_aspect_ratio", 0.18))
        max_aspect = float(segmentation_config.get("max_symbol_aspect_ratio", 1.35))
        padding = int(segmentation_config.get("padding_px", 2))
        max_components = int(segmentation_config.get("max_components", 18))
        row_merge_threshold = float(segmentation_config.get("row_merge_threshold", 0.55))

        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area <= 0:
                continue
            area_ratio = area / plate_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            aspect_ratio = w / float(h)
            if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
                continue
            filtered_boxes.append((x, y, w, h))

        if not filtered_boxes:
            return processed, []

        filtered_boxes = sorted(filtered_boxes, key=lambda b: b[2] * b[3], reverse=True)[: max_components * 2]
        sorted_boxes = self._sort_boxes(filtered_boxes, row_merge_threshold)[:max_components]
        if not sorted_boxes:
            return processed, []

        height = processed.shape[0]
        width = processed.shape[1]
        sorted_segments: List[np.ndarray] = []
        for x, y, w, h in sorted_boxes:
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = processed[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sorted_segments.append(crop)

        return processed, sorted_segments
