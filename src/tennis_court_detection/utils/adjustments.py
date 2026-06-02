from typing import Literal

import cv2
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.geometry.points import Point
from cvgeomkit.utils.plotting import display_img
import numpy as np

from tennis_court_detection.config import get_debug_mode
from tennis_court_detection.utils.filters import filter_horizontal_lines


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


def traverse_horizontal_line(
    img: np.ndarray,
    p_left: Point,
    p_right: Point,
    direction: Literal['left', 'right'],
    step = 50,
    h_delta = 20,
):
    p_c = Point((p_left.x + p_right.x) // 2, p_left.y)

    if direction =='left':
        x1 = p_c.x - step
        x2 = p_c.x
    else:
        x1 = p_c.x 
        x2 = p_c.x + step
        
    img_copy = img.copy()

    lines = []
    segment_xs = []
    while {'left': x2 > p_left.x, 'right': x1 < p_right.x}[direction]:

        crop = img[p_c.y - h_delta: p_c.y + h_delta, x1: x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(crop_gray, 100, 200)
        segments = cv2.HoughLinesP(edges, 1, np.pi / 180, int(step * 0.8), int(step * 0.4), int(step * 0.1))

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
            print(sorted(sub_lines, key = lambda line: line.intercept))
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

    if get_debug_mode():
        display_img(crop_copy)
        cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)
        display_img(img_copy)

    interpolated_lines = interpolate_lines_intercept(lines)

    for line in interpolated_lines:
        print(line)

    return interpolated_lines, build_points(interpolated_lines, segment_xs)
