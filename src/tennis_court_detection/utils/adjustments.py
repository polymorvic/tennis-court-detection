from typing import Literal

import cv2
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.geometry.segments import LineSegment, transform_line_segment
from cvgeomkit.geometry.points import Point, transform_point
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.common import ArrayLike
import numpy as np
from scipy.ndimage import median_filter

from tennis_court_detection.config import get_debug_mode
from tennis_court_detection.utils.filters import filter_horizontal_lines
from tennis_court_detection.schemas.config import TraverseDirection


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


def build_segments(
    lines: list[Line],
    segment_xs: list[tuple[int, int]],
) -> list[LineSegment]:

    segments = []
    for line, (x1, x2) in zip(lines, segment_xs):
        y = int(line.intercept)
        p1 = Point(x1, y)
        p2 = Point(x2, y)
        segments.append(LineSegment(p1, p2))

    return segments


def adjust_lines_intercept(
    lines: list[Line],
    window: int = 7,
    max_dev: float = 2.0,
    iterations: int = 5,
) -> list[Line]:
    if window % 2 == 0:
        window += 1

    y = np.array([l.intercept for l in lines], dtype=float)
    edge = window // 2

    for _ in range(iterations):
        trend = median_filter(y, size=window, mode="reflect")

        mask = np.abs(y - trend) >= max_dev
        mask[:edge] = False
        mask[-edge:] = False

        if not mask.any():
            break

        y[mask] = trend[mask]

    for line, intercept in zip(lines, y):
        line.intercept = float(round(intercept))

    return lines


def traverse_horizontal_line(
    img: np.ndarray,
    p_left: Point,
    p_right: Point,
    direction: TraverseDirection,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.0186,
    lower_canny_thresh: int = 20,
    upper_canny_thresh: int = 100,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1
) -> tuple[list[Line], list[tuple[int, int]]]:
    p_c = Point((p_left.x + p_right.x) // 2, p_left.y)
    step = int(img.width * step_ratio)
    h_delta = int(img.height * h_delta_ratio)

    if direction == TraverseDirection.LEFT:
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

    if get_debug_mode():
        print(f'before adjust: {interpolated_lines=}')
        print(f'after adjust: {interpolated_lines=}')
        display_img(crop_copy)
        cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)
        display_img(img_copy)

    return interpolated_lines, segment_xs


def adjust_horizontal_line(
    img: ArrayLike,
    left_point: Point,
    right_point: Point,
    step_ratio: float = 0.026,
    height_delta_ratio: float = 0.0186
) -> list[LineSegment]:
    lines_left, xs_left = traverse_horizontal_line(
        img, left_point, right_point, TraverseDirection.LEFT,
        step_ratio, height_delta_ratio
    )
    lines_right, xs_right = traverse_horizontal_line(
        img, left_point, right_point, TraverseDirection.RIGHT,
        step_ratio, height_delta_ratio
    )

    pairs = list(zip(lines_left + lines_right, xs_left + xs_right))
    pairs.sort(key=lambda p: (p[1][0] + p[1][1]) / 2)
    lines, segment_xs = map(list, zip(*pairs))
    lines = adjust_lines_intercept(lines)
    return build_segments(lines, segment_xs)


def traverse_sideline(
    start_intersection: Intersection,
    original_img: ArrayLike,
    lower_canny_thresh: int = 20,
    upper_canny_thresh: int = 100,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1,
    window_size_ratio: float = 0.016,
    kernel_size_ratio: float = 0.3,
    move_up_first_window_ratio: float = 0.3,
    adapt_patience: int = 10
) -> list[LineSegment]:
    
    current_patience = 0
    
    def adapt_params(
        initial_lower_canny_thresh: int,
        initial_hough_thresh_ratio: float,
        canny_adapt_ratio: float = 0.05,
        hough_adapt_ratio: float = 0.95
    ) -> tuple[int, float]:
        nonlocal current_patience
        current_patience += 1
        return initial_lower_canny_thresh - int(initial_lower_canny_thresh * canny_adapt_ratio), initial_hough_thresh_ratio * hough_adapt_ratio
    
    orig_lower_canny_thresh = lower_canny_thresh
    orig_hough_thresh_ratio = hough_thresh_ratio

    def reset_params():
        nonlocal current_patience
        current_patience = 0
        return orig_lower_canny_thresh, orig_hough_thresh_ratio
    
    original_img_copy = original_img.copy()
    img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    window_size = int(original_img.height * window_size_ratio)
    kernel_size = int(window_size * kernel_size_ratio)
    move_up_first_window = int(move_up_first_window_ratio * 2 * window_size)
    counter = 0
    line_segments = []
    start_point = start_intersection.point
    original_line = filter_horizontal_lines([start_intersection.line1, start_intersection.line2], horizontal=False)[0]
    while True:

        if counter == 0:
            top_left = Point(start_point.x - window_size, start_point.y - window_size - move_up_first_window)
            bottom_right = Point(start_point.x + window_size, start_point.y + window_size - move_up_first_window)

        else:
            top_left = Point(start_point.x - window_size, start_point.y - window_size)
            bottom_right = Point(start_point.x + window_size, start_point.y + window_size)

        cv2.rectangle(original_img_copy, top_left, bottom_right, (0, 255, 0), 2)

        crop_side_gray = img_gray[top_left.y:bottom_right.y, top_left.x:bottom_right.x]
        if crop_side_gray.size == 0:
            break

        crop_side_gray = cv2.medianBlur(crop_side_gray, kernel_size)
        original_line_local = transform_line(original_line, original_img, top_left.x, top_left.y, to_global=False)

        try:
            original_line_local.limit_to_img(crop_side_gray)
        except ValueError as e:
            start_point = transform_point(Point(x = window_size, y = 0), top_left.x, top_left.y)
            continue

        crop_side_rgb = original_img[top_left.y:bottom_right.y, top_left.x:bottom_right.x]

        crop_side_edges = cv2.Canny(crop_side_gray, lower_canny_thresh, upper_canny_thresh)
        segments = cv2.HoughLinesP(
            crop_side_edges,
            rho=1,
            theta=np.pi/180,
            threshold=int(window_size * hough_thresh_ratio),
            minLineLength=int(window_size * min_line_len_ratio),
            maxLineGap=int(window_size * max_line_gap_ratio)
        )

        if current_patience == adapt_patience:
            break

        if segments is None:
            lower_canny_thresh, hough_thresh_ratio = adapt_params(lower_canny_thresh, hough_thresh_ratio)
            continue

        if get_debug_mode():
            crop_copy = crop_side_rgb.copy()
            for segment in segments:
                x1, y1, x2, y2 = segment[0]
                cv2.line(crop_copy, (x1, y1), (x2, y2), (0, 255, 0))
            display_img(crop_copy)

        lines = [Line.from_hough_segment(*segment) for segment in segments]

        not_horizontal_lines = filter_horizontal_lines(lines, horizontal=False)
        if not not_horizontal_lines:
            lower_canny_thresh, hough_thresh_ratio = adapt_params(lower_canny_thresh, hough_thresh_ratio)
            continue

        upper_points = []
        iter_segments = []
        for line in not_horizontal_lines:
            if line.slope is None or abs(line.slope - original_line.slope) > 0.5:
                continue
            p1, p2 = line.limit_to_img(crop_side_gray)
            upper_points.append(p1)

            ls = LineSegment.from_points(p1, p2)
            iter_segments.append(ls)
        
        iter_segments = sorted(iter_segments, key = lambda segment: (segment.start.x, segment.end.x))
        if sum(np.sign(line.slope) for line in not_horizontal_lines) > 0:
            idx = -1
        else:
            idx = 0

        if upper_points:
            next_point = sorted(upper_points, key = lambda point: point.x)[idx]
            next_point_global = transform_point(next_point, top_left.x, top_left.y)
            line_segments.append(
                transform_line_segment(
                    iter_segments[idx], top_left.x, top_left.y
                ))
            
            start_point = next_point_global

        else:
            start_point = transform_point(Point(x = window_size, y = 0), top_left.x, top_left.y)

            local_line = transform_line(original_line, original_img, top_left.x, top_left.y, to_global=False)
            p1, p2 = local_line.limit_to_img(crop_side_gray)
            last_segment_local = LineSegment.from_points(p1, p2)
            line_segments.append(transform_line_segment(last_segment_local, top_left.x, top_left.y))
            

        reset_params()
        counter += 1

    if get_debug_mode():
        display_img(original_img_copy)

    return line_segments