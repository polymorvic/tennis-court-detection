from typing import Literal

import cv2
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.geometry.points import Point
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.common import ArrayLike
import numpy as np
from scipy.ndimage import median_filter

from tennis_court_detection.config import get_debug_mode
from tennis_court_detection.utils.filters import filter_horizontal_lines
from tennis_court_detection.schemas.config import HorizontalTraverseDirection


def interpolate_lines_intercept(lines: list[Line | None]) -> list[Line]:
    intercepts = np.array([line.intercept if line else np.nan for line in lines])
    nan_indices = np.argwhere(np.isnan(intercepts)).flatten()
    not_nan_indices = np.argwhere(~np.isnan(intercepts)).flatten()

    interpolated = np.interp(nan_indices, not_nan_indices, intercepts[not_nan_indices])
    intercepts[nan_indices] = np.round(interpolated)
    
    return [Line(slope=0, intercept=intercept) for intercept in intercepts]


def build_points(
    lines: list[Line], 
    segment_xs: list[tuple[int, int]]
) -> list[tuple[Point, Point]]:
    
    points = []
    for line, (x1, x2) in zip(lines, segment_xs):
        p1 = Point(x1, int(line.intercept))
        p2 = Point(x2, int(line.intercept))
        points.append((p1, p2))
    return points


def adjust_lines_intercept(
    lines: list[Line], 
    window: int = 7, 
    max_dev: int | float = 3.0
) -> list[Line]:
    intercepts = np.array([l.intercept for l in lines], dtype=float)
    trend = median_filter(
        intercepts,
        size=window,
        mode="reflect"
    )

    corrected = intercepts.copy()
    mask = np.abs(intercepts - trend) > max_dev

    x = np.arange(len(intercepts))
    corrected[mask] = np.interp(
        x[mask],
        x[~mask],
        intercepts[~mask]
    )

    for line, value in zip(lines, corrected):
        line.intercept = float(value)

    return lines


def traverse_horizontal_line(
    img: np.ndarray,
    p_left: Point,
    p_right: Point,
    direction: HorizontalTraverseDirection,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.0186,
    lower_canny_thresh: int = 20,
    upper_canny_thresh: int = 100,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1
):
    p_c = Point((p_left.x + p_right.x) // 2, p_left.y)
    step = int(img.width * step_ratio)
    h_delta = int(img.height * h_delta_ratio)

    if direction == HorizontalTraverseDirection.LEFT:
        x1 = p_c.x - step
        x2 = p_c.x
    else:
        x1 = p_c.x 
        x2 = p_c.x + step
        
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    lines = []
    segment_xs = []
    while {'left': x2 > p_left.x, 'right': x1 < p_right.x}[direction]:

        crop = img[p_c.y - h_delta: p_c.y + h_delta, x1: x2]
        crop_gray = img_gray[p_c.y - h_delta: p_c.y + h_delta, x1: x2]

        edges = cv2.Canny(crop_gray, lower_canny_thresh, upper_canny_thresh)
        segments = cv2.HoughLinesP(
            edges, 
            rho = 1, 
            theta = np.pi / 180,
            threshold = int(step * hough_thresh_ratio), 
            minLineLength=int(step * min_line_len_ratio),
            maxLineGap=int(step * max_line_gap_ratio)
        )

        if get_debug_mode():
            display_img(crop_gray)
            display_img(edges)

        crop_copy = crop.copy()
        sub_lines = []
        if segments is not None:
            for line in segments:

                if get_debug_mode():
                    x1_hough, y1_hough, x2_hough, y2_hough = line[0]
                    cv2.line(crop_copy, (x1_hough, y1_hough), (x2_hough, y2_hough), (0, 255, 0), 1)

                line = Line.from_hough_segment(line[0])
                sub_lines.append(line)
        
        sub_lines = filter_horizontal_lines(sub_lines)
        if sub_lines:
            bottom_line = sorted(sub_lines, key = lambda line: line.intercept)[-1]
            bottom_line_global = transform_line(bottom_line, crop, x1, p_c.y - h_delta)
            lines.append(bottom_line_global)
        else:
            lines.append(None)


        segment_xs.append((min(x1, x2), max(x1, x2)))
    
        if direction == 'left':
            x2 = x1
            x1 -= step
            
        else:
            x1 = x2
            x2 += step

        if get_debug_mode():
            display_img(crop_copy)
            cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)

    interpolated_lines = interpolate_lines_intercept(lines)
    interpolated_lines = adjust_lines_intercept(interpolated_lines)

    if get_debug_mode():
        print(f'before adjust: {interpolated_lines=}')
        print(f'after adjust: {interpolated_lines=}')
        display_img(crop_copy)
        cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)
        display_img(img_copy)

    return build_points(interpolated_lines, segment_xs)


def adjust_horizontal_line(
    img: ArrayLike,
    left_point: Point,
    right_point: Point,
    step_ratio: float = 0.026,
    height_delta_ratio: float = 0.0186
) -> list[tuple[Point, Point]]:
    points = traverse_horizontal_line(img, left_point, right_point, HorizontalTraverseDirection.LEFT, step_ratio, height_delta_ratio)
    points.extend(
        traverse_horizontal_line(img, left_point, right_point, HorizontalTraverseDirection.RIGHT, step_ratio, height_delta_ratio)
    )
    return points

