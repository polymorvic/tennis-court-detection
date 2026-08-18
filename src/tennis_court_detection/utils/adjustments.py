import cv2
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.geometry.segments import LineSegment, transform_line_segment
from cvgeomkit.geometry.points import Point, transform_point
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.common import ArrayLike
import numpy as np
from scipy.ndimage import median_filter

from tennis_court_detection.config import get_debug_mode, set_debug_mode
from tennis_court_detection.schemas.court import HalfLine
from tennis_court_detection.utils.filters import filter_horizontal_lines
from tennis_court_detection.utils.helpers import make_odd_kernel_size, lines_from_gray_img
from tennis_court_detection.utils.validators import check_if_numpy_image, exceeds_empty_threshold
from tennis_court_detection.schemas.config import Direction, LinePosition
from tennis_court_detection.utils.errors import NotEnoughLineSegmentsFound


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
    direction: Direction,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.0186,
    lower_canny_thresh: int = 20,
    upper_canny_thresh: int = 100,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1,
    line_position: LinePosition = LinePosition.BOTTOM,
    horizontal_static: bool = True,
    to_center: bool = False
) -> tuple[list[Line], list[tuple[int, int]]]:
    img = check_if_numpy_image(img)
    p_c = Point((p_left.x + p_right.x) // 2, p_left.y)
    step = int(img.width * step_ratio)
    h_delta = int(img.height * h_delta_ratio)

    position_index = {
        LinePosition.TOP: 0, 
        LinePosition.BOTTOM: -1
    }[line_position]

    if not to_center:
        if direction == Direction.LEFT:
            x1 = p_c.x - step
            x2 = p_c.x
        else:
            x1 = p_c.x 
            x2 = p_c.x + step
    else:
        if direction == Direction.LEFT:
            x1 = p_left.x
            x2 = p_left.x + step
        else:
            x1 = p_right.x - step
            x2 = p_right.x
        
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    
    lines = []
    segment_xs = []
    y1 = p_c.y - h_delta
    y2 = p_c.y + h_delta
    while (
        {Direction.LEFT: x2 > p_left.x, Direction.RIGHT: x1 < p_right.x}[direction]
        if not to_center
        else {Direction.LEFT: x1 < p_c.x, Direction.RIGHT: x2 > p_c.x}[direction]
    ):

        y_c =  (y2 + y1) // 2

        crop = img[y1: y2, x1: x2]
        crop_gray = img_gray[y1: y2, x1: x2]

        if crop.size == 0:
            break

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
            searched_line = sorted(sub_lines, key = lambda line: line.intercept)[position_index]
            searched_line_global = transform_line(searched_line, crop, x1, y1)
            lines.append(searched_line_global)
        else:
            lines.append(None)


        segment_xs.append((min(x1, x2), max(x1, x2)))

        if get_debug_mode():
            display_img(crop_copy)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
        if not to_center:
            if direction == Direction.LEFT:
                x2 = x1
                x1 -= step
            elif direction == Direction.RIGHT:
                x1 = x2
                x2 += step
        else:
            if direction == Direction.LEFT:
                x1 = x2
                x2 += step
            elif direction == Direction.RIGHT:
                x2 = x1
                x1 -= step

        if sub_lines and not horizontal_static:
            y_diff = int(searched_line_global.intercept) - y_c
            y1 += y_diff
            y2 += y_diff

    if get_debug_mode():
        display_img(crop_copy)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        display_img(img_copy)

    return lines, segment_xs


def traverse_with_validation_and_interpolation(
    img: np.ndarray,
    p_left: Point,
    p_right: Point,
    direction: Direction,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.0186,
    lower_canny_thresh: int = 20,
    upper_canny_thresh: int = 100,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1,
    line_position: LinePosition = LinePosition.BOTTOM,
    horizontal_static: bool = True,  
    to_center: bool = False 
):
    lines, segment_xs = traverse_horizontal_line(
        img, p_left, p_right, direction,
        step_ratio, h_delta_ratio,
        lower_canny_thresh, upper_canny_thresh,
        hough_thresh_ratio, min_line_len_ratio, max_line_gap_ratio,
        line_position, horizontal_static, to_center=to_center
    )

    if exceeds_empty_threshold(lines):
        raise NotEnoughLineSegmentsFound()

    interpolated_lines = interpolate_lines_intercept(lines)

    return interpolated_lines, segment_xs


def adjust_horizontal_line(
    img: ArrayLike,
    left_point: Point,
    right_point: Point,
    step_ratio: float = 0.026,
    height_delta_ratio: float = 0.0186,
    line_position: LinePosition = LinePosition.BOTTOM,
    horizontal_static: bool = True,
    to_center: bool = False
) -> list[LineSegment]:
    lines_left, xs_left = traverse_with_validation_and_interpolation(
        img, left_point, right_point, Direction.LEFT,
        step_ratio, height_delta_ratio, line_position=line_position, 
        horizontal_static=horizontal_static, to_center=to_center
    )
    lines_right, xs_right = traverse_with_validation_and_interpolation(
        img, left_point, right_point, Direction.RIGHT,
        step_ratio, height_delta_ratio, line_position=line_position,
        horizontal_static=horizontal_static, to_center=to_center
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
    adapt_patience: int = 10,
    original_line_slope_similarity_max_thresh: float = 0.5
) -> list[LineSegment]:
    original_img = check_if_numpy_image(original_img)
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
    kernel_size = make_odd_kernel_size(window_size, kernel_size_ratio)
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

        # if get_debug_mode():
        #     crop_copy = crop_side_rgb.copy()
        #     for segment in segments:
        #         x1, y1, x2, y2 = segment[0]
        #         cv2.line(crop_copy, (x1, y1), (x2, y2), (0, 255, 0))
        #     display_img(crop_copy)

        lines = [Line.from_hough_segment(*segment) for segment in segments]

        not_horizontal_lines = filter_horizontal_lines(lines, horizontal=False)
        if not not_horizontal_lines:
            lower_canny_thresh, hough_thresh_ratio = adapt_params(lower_canny_thresh, hough_thresh_ratio)
            continue

        line_candidates = [line for line in not_horizontal_lines if line.slope is not None and abs(line.slope - original_line.slope) < original_line_slope_similarity_max_thresh]

        if get_debug_mode():
            crop_copy = crop_side_rgb.copy()
            for line in line_candidates:
                p1, p2 = line.limit_to_img(crop_side_gray)
                cv2.line(crop_copy, p1, p2, (0, 255, 0))
            display_img(crop_copy)

        points = []
        iter_segments = []
        for line in line_candidates:
            if line.slope is None or abs(line.slope - original_line.slope) > original_line_slope_similarity_max_thresh:
                continue
            p1, p2 = line.limit_to_img(crop_side_gray)

            upper_point = p1 if p1.y < p2.y else p2
            lower_point = p1 if p1.y >= p2.y else p2

            points.append((lower_point, upper_point))

            ls = LineSegment.from_points(lower_point, upper_point)
            iter_segments.append(ls)
        
        iter_segments = sorted(iter_segments, key = lambda segment: (segment.start.x, segment.end.x))
        if sum(np.sign(line.slope) for line in line_candidates) > 0:
            idx = -1
        else:
            idx = 0

        if points:
            next_point = sorted(points, key = lambda point_pair: (point_pair[0].x, point_pair[1].x))[idx]
            next_point_global = transform_point(next_point[1], top_left.x, top_left.y)

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

            upper_point = p1 if p1.y < p2.y else p2
            lower_point = p1 if p1.y >= p2.y else p2

            next_point_global = transform_point(upper_point, top_left.x, top_left.y)
            start_point = next_point_global
            

        reset_params()
        counter += 1

    if get_debug_mode():
        display_img(original_img_copy)

    return line_segments


def scan_line_segments_for_horizontal_lines(
    img: ArrayLike,
    line_segments: list[LineSegment],
    side: Direction,
    roi_window_width_ratio: float = 0.1,
    canny_lower_thresh: int = 20,
    canny_upper_thresh: int = 100,
    hough_thresh: int = 50,
    min_len_ratio: float = 0.5,
    max_gap_ratio: float = 0.1
) -> list[HalfLine]:
    img_copy = img.copy()
    half_lines = set()

    for segment in line_segments[2:]:
        min_x, max_x = sorted((segment.start.x, segment.end.x))
        min_y, max_y = sorted((segment.start.y, segment.end.y))

        roi = {
            'left': img[min_y:max_y, min_x:int(max_x + roi_window_width_ratio*img_copy.width)],
            'right': img[min_y:max_y, int(min_x - roi_window_width_ratio*img_copy.width):max_x]
        }[side]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        min_len_px = int(min_len_ratio * roi.width)
        max_gap_px = int(max_gap_ratio * roi.width)
        lines = lines_from_gray_img(roi_gray, canny_lower_thresh, canny_upper_thresh, hough_thresh, min_len_px, max_gap_px)

        h_lines = filter_horizontal_lines(lines)
        if not h_lines:
            continue

        for line in h_lines:
            global_line = transform_line(line, roi, min_x, min_y)

            x = {
                Direction.LEFT: min_x,
                Direction.RIGHT: max_x
            }[side]
            point = Point(x, int(global_line.intercept))

            half_line = HalfLine(point = point, line = global_line)
            half_lines.add(half_line)

        if get_debug_mode():
            roi_gray_copy = roi_gray.copy()
            for line in h_lines:
                p1, p2 = line.limit_to_img(roi_gray)
                cv2.line(roi_gray_copy, p1, p2, (255, 0, 0), 3)
            display_img(roi_gray_copy)

        if get_debug_mode():
            cv2.rectangle(img_copy, (min_x, min_y), (int(max_x + roi_window_width_ratio*img_copy.width), max_y), (255, 0, 0), 3)

    return sorted(half_lines, key=lambda half_line: half_line.line.intercept)


def traverse_for_bottom_netline(
    img: np.ndarray,
    p_left: Point,
    p_right: Point,
    direction: Direction,
    step_ratio: float = 0.015,
    h_delta_ratio: float = 0.0186,
    canny_lower_thresh: int = 50,
    canny_upper_thresh: int = 150,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.2,
    max_line_gap_ratio: float = 0.1
) -> tuple[list[Line], list[tuple[int, int]]]:
    img = check_if_numpy_image(img)
    p_c = Point((p_left.x + p_right.x) // 2, p_left.y)
    step = int(img.width * step_ratio)
    h_delta = int(img.height * h_delta_ratio)

    if direction == Direction.LEFT:
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
        crop_gray = cv2.bilateralFilter(crop_gray, 5, 75, 75)

        crop_bin = (crop_gray < crop_gray.min() + crop_gray.std()).astype(np.uint8) * 255
        edges = cv2.Canny(crop_bin, canny_lower_thresh, canny_upper_thresh)

        hough_thresh = int(crop.width * hough_thresh_ratio)
        min_line_len_px = int(crop.width * min_line_len_ratio)
        max_line_gap_px = int(crop.width * max_line_gap_ratio)
        segments = cv2.HoughLinesP(
            edges, 
            rho = 1, 
            theta = np.pi / 180,
            threshold = hough_thresh, 
            minLineLength=min_line_len_px,
            maxLineGap=max_line_gap_px
        )

        if get_debug_mode():
            display_img(crop)
            display_img(crop_gray)
            display_img(crop_bin)
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

        if get_debug_mode():
            display_img(crop_copy)
            cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)
    
        if direction == Direction.LEFT:
            x2 = x1
            x1 -= step
            
        else:
            x1 = x2
            x2 += step


    if exceeds_empty_threshold(lines):
        raise NotEnoughLineSegmentsFound()

    interpolated_lines = interpolate_lines_intercept(lines)

    if get_debug_mode():
        print(f'before adjust: {interpolated_lines=}')
        print(f'after adjust: {interpolated_lines=}')
        display_img(crop_copy)
        cv2.rectangle(img_copy, (x1, p_c.y - h_delta), (x2, p_c.y + h_delta), (0, 255, 0), 2)
        display_img(img_copy)

    return interpolated_lines, segment_xs


def adjust_net_line_segments(
    img: ArrayLike,
    left_point: Point,
    right_point: Point,
    step_ratio: float = 0.015,
    height_delta_ratio: float = 0.0186
) -> list[LineSegment]:
    lines_left, xs_left = traverse_for_bottom_netline(
        img, left_point, right_point, Direction.LEFT,
        step_ratio, height_delta_ratio
    )
    lines_right, xs_right = traverse_for_bottom_netline(
        img, left_point, right_point, Direction.RIGHT,
        step_ratio, height_delta_ratio
    )

    pairs = list(zip(lines_left + lines_right, xs_left + xs_right))
    pairs.sort(key=lambda p: (p[1][0] + p[1][1]) / 2)
    lines, segment_xs = map(list, zip(*pairs))
    lines = adjust_lines_intercept(lines)
    return build_segments(lines, segment_xs)


def traverse_along_line(
    img: np.ndarray,
    edges: np.ndarray,
    start_point: Point,
    guide_line: Line,
    direction: Direction,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.05,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1,
    angle_tolerance_deg: float = 15,
    line_position: LinePosition = LinePosition.TOP
) -> tuple[list[Line | None], list[tuple[int, int]]]:

    img = check_if_numpy_image(img)

    if edges.shape[:2] != img.shape[:2]:
        raise ValueError("img and edges must have the same width and height")

    position_index = {
        LinePosition.TOP: 0,
        LinePosition.BOTTOM: -1
    }[line_position]

    step = max(1, int(img.width * step_ratio))
    h_delta = max(1, int(img.height * h_delta_ratio))

    lines = []
    segment_xs = []

    img_copy = img.copy()

    if direction == Direction.LEFT:
        x2 = start_point.x
        x1 = x2 - step
    else:
        x1 = start_point.x
        x2 = x1 + step

    while (
        x2 > 0
        if direction == Direction.LEFT
        else x1 < img.width
    ):
        x1_crop = max(0, x1)
        x2_crop = min(img.width, x2)

        if x1_crop >= x2_crop:
            break

        x_c = (x1_crop + x2_crop) // 2

        y_c = int(
            guide_line.slope * x_c
            + guide_line.intercept
        )

        if (
            y_c + h_delta <= 0
            or y_c - h_delta >= img.height
        ):
            break

        y1 = max(0, y_c - h_delta)
        y2 = min(img.height, y_c + h_delta)

        if y1 >= y2:
            break

        crop = img[y1:y2, x1_crop:x2_crop]
        crop_edges = edges[y1:y2, x1_crop:x2_crop]

        if crop.size == 0 or crop_edges.size == 0:
            break

        guide_line_local = transform_line(
            original_line=guide_line,
            original_img=img,
            original_x_start=x1_crop,
            original_y_start=y1,
            to_global=False
        )

        segments = cv2.HoughLinesP(
            crop_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(step * hough_thresh_ratio),
            minLineLength=int(step * min_line_len_ratio),
            maxLineGap=int(step * max_line_gap_ratio)
        )

        crop_copy = crop.copy()
        candidates = []

        if segments is not None:
            for segment in segments:

                if get_debug_mode():
                    x1_hough, y1_hough, x2_hough, y2_hough = segment[0]

                    cv2.line(
                        crop_copy,
                        (x1_hough, y1_hough),
                        (x2_hough, y2_hough),
                        (0, 255, 0),
                        1
                    )

                detected_line_local = Line.from_hough_segment(
                    segment[0]
                )

                guide_angle = np.degrees(
                    np.arctan(guide_line_local.slope)
                )

                detected_angle = np.degrees(
                    np.arctan(detected_line_local.slope)
                )

                angle_diff = abs(
                    detected_angle - guide_angle
                )

                if angle_diff > angle_tolerance_deg:
                    continue

                candidates.append(detected_line_local)

        if candidates:
            searched_line_local = sorted(
                candidates,
                key=lambda line: line.intercept
            )[position_index]

            searched_line_global = transform_line(
                original_line=searched_line_local,
                original_img=crop,
                original_x_start=x1_crop,
                original_y_start=y1,
                to_global=True
            )

            lines.append(searched_line_global)

        else:
            lines.append(guide_line)

        segment_xs.append(
            (min(x1_crop, x2_crop), max(x1_crop, x2_crop))
        )

        if get_debug_mode():
            cv2.rectangle(
                img_copy,
                (x1_crop, y1),
                (x2_crop, y2),
                (0, 255, 0),
                2
            )

        if direction == Direction.LEFT:
            x2 = x1
            x1 -= step
        else:
            x1 = x2
            x2 += step

    if get_debug_mode():
        display_img(img_copy)

    return lines, segment_xs


def traverse_v_shaped_line_pairs(
    img: np.ndarray,
    edges: np.ndarray,
    intersection: Intersection,
    step_ratio: float = 0.026,
    h_delta_ratio: float = 0.05,
    hough_thresh_ratio: float = 0.8,
    min_line_len_ratio: float = 0.4,
    max_line_gap_ratio: float = 0.1,
    angle_tolerance_deg: float = 15
) -> list[LineSegment]:

    img = check_if_numpy_image(img)

    point = intersection.point
    line_1, line_2 = intersection.line1, intersection.line2

    step = max(1, int(img.width * step_ratio))

    x_test = point.x - step

    y_1 = line_1.slope * x_test + line_1.intercept
    y_2 = line_2.slope * x_test + line_2.intercept

    if y_1 < y_2:
        left_line = line_1
        right_line = line_2
    else:
        left_line = line_2
        right_line = line_1

    left_lines, left_xs = traverse_along_line(
        img=img,
        edges=edges,
        start_point=point,
        guide_line=left_line,
        direction=Direction.LEFT,
        step_ratio=step_ratio,
        h_delta_ratio=h_delta_ratio,
        hough_thresh_ratio=hough_thresh_ratio,
        min_line_len_ratio=min_line_len_ratio,
        max_line_gap_ratio=max_line_gap_ratio,
        angle_tolerance_deg=angle_tolerance_deg
    )

    right_lines, right_xs = traverse_along_line(
        img=img,
        edges=edges,
        start_point=point,
        guide_line=right_line,
        direction=Direction.RIGHT,
        step_ratio=step_ratio,
        h_delta_ratio=h_delta_ratio,
        hough_thresh_ratio=hough_thresh_ratio,
        min_line_len_ratio=min_line_len_ratio,
        max_line_gap_ratio=max_line_gap_ratio,
        angle_tolerance_deg=angle_tolerance_deg
    )

    line_and_x_pairs = list(zip(
        left_lines + right_lines,
        left_xs + right_xs
    ))

    line_and_x_pairs.sort(
        key=lambda pair: (pair[1][0] + pair[1][1]) / 2
    )

    line_segments = []
    for line, (x_start, x_end) in line_and_x_pairs:

        ls = LineSegment.from_tuples(
            start=(
                x_start,
                int(line.slope * x_start + line.intercept)
            ),
            end=(
                x_end,
                int(line.slope * x_end + line.intercept)
            )
        )

        line_segments.append(ls)

    return line_segments