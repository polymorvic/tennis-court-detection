import cv2
import numpy as np
from cvgeomkit.common import ArrayLike, Numeric, NumpyImage
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.intersections import Intersection, transform_intersection
from tennis_court_detection.schemas.config import ServiceSide
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.utils.plotting import display_img

from tennis_court_detection.utils.validators import check_if_numpy_image
from tennis_court_detection.utils.metrics import (
    calculate_white_columns_ratio, 
    calculate_white_pixels_ratio,
    get_intercept_std
)

from tennis_court_detection.utils.helpers import (
    lines_from_gray_img, 
    angle_between_lines, 
    make_odd_kernel_size,
    mask_line_neighborhood_on_edges
)
from tennis_court_detection.config import get_debug_mode


def filter_horizontal_lines(
    lines: list[Line],
    slope_thresh: float = 0.02,
    horizontal: bool = True,
    include_none_slope: bool = False
) -> list[Line] | None:

    if horizontal:
        filtered = [
            line
            for line in lines
            if line.slope is not None and abs(line.slope) < slope_thresh
        ]
    else:
        filtered = [
            line
            for line in lines
            if (line.slope is not None and abs(line.slope) > slope_thresh) or (include_none_slope and line.slope is None)
        ]

    return filtered if filtered else None


def get_vertical_lines(
    lines: list[Line],
    theta_thresh: float = 1.0
) -> list[Line] | None:
    lines = [line for line in lines if abs(line.theta - 90) < theta_thresh]
    return lines if lines else None


def get_centre_vertical_lines(
    lines: list[Line],
    img: ArrayLike,
    delta: Numeric = 100,
    max_spread: Numeric = 10,
):
    img = check_if_numpy_image(img)
    centre_x = img.width // 2

    centre_lines = [line for line in lines if line.xv is not None and abs(line.xv - centre_x) <= delta]
    if not centre_lines:
        return []

    centre_lines = sorted(centre_lines,key=lambda line: abs(line.xv - centre_x))[:3]

    xs = [line.xv for line in centre_lines]
    spread = max(xs) - min(xs)
    if spread > max_spread:
        return []

    return centre_lines


def filter_service_intersections(
    intersections: set[Intersection],
    service_lines: list[Line],
    centre_lines: list[Line],
    service_side: ServiceSide,
    angle_tol: float = 5,
) -> tuple[Line, Line, Point] | None:
    h_line = max(service_lines, key=lambda line: line.intercept)
    h_key = h_line._key_()

    if service_side == ServiceSide.LEFT:
        v_line = max(centre_lines, key=lambda line: line.xv)

    elif service_side == ServiceSide.RIGHT:
        v_line = min(centre_lines, key=lambda line: line.xv)
    
    v_key = v_line._key_()

    for intersection in intersections:
        if abs(intersection.angle % 180 - 90) >= angle_tol:
            continue
        keys = {intersection.line1._key_(), intersection.line2._key_()}
        if h_key in keys and v_key in keys:
            return h_line, v_line, intersection.point
    return None


def ensure_is_baseline(
    baseline_candidate: Line,
    img_gray: ArrayLike,
    roi_width: int,
    canny_lower_thresh: int,
    canny_upper_thresh: int,
    hough_thresh: int,
    min_line_len_ensure_width_ratio: float,
    max_line_gap_width_ratio: float,
    delta_ensure_height_ratio: float,
    candidates_count: int = 4,
    bottom_margin_height_ratio: float = 0.014
) -> tuple[Line, bool, list[Line]]:
    img_gray = check_if_numpy_image(img_gray)
    h = int(baseline_candidate.intercept)

    h_delta = int(img_gray.height * delta_ensure_height_ratio)
    y0 = max(0, h - h_delta)
    y1 = min(img_gray.height, h + int(img_gray.height * bottom_margin_height_ratio))

    roi_gray = img_gray[y0:y1].copy()
    
    if roi_gray.height < 5:
        return baseline_candidate, False, []

    min_line_len_px = int(min_line_len_ensure_width_ratio * roi_width)
    max_line_gap_px = int(max_line_gap_width_ratio * roi_width)
    lines = lines_from_gray_img(
        roi_gray, 
        canny_lower_thresh,
        canny_upper_thresh,
        hough_thresh,
        min_line_len_px,
        max_line_gap_px
    )

    if not lines:
        return baseline_candidate, False, []
    
    horizontal_lines = filter_horizontal_lines(lines)
    if horizontal_lines:
        horizontal_lines_global = [transform_line(line, roi_gray, 0, y0) for line in horizontal_lines]
        horizontal_lines_global = sorted(horizontal_lines_global, key = lambda line: line.intercept)

        baseline_candidate = max(
            (
                line
                for line in horizontal_lines_global
                if line.intercept < baseline_candidate.intercept
            ),
            key=lambda line: line.intercept,
            default=baseline_candidate,
        )
    
    sidelines = filter_horizontal_lines(lines, horizontal=False)
    if not sidelines:
        return baseline_candidate, False, []
    
    sidelines_global = [transform_line(line, roi_gray, 0, y0) for line in sidelines]
    
    count = 0
    lines = []
    for line in sidelines_global:
        x_axis_angle = line.theta
        intersection = baseline_candidate.intersection(line, img_gray)
        if (-90 < x_axis_angle -10 or 10 < x_axis_angle < 90) and intersection is not None:
            lines.append(line)
            count += 1

    return baseline_candidate, count > candidates_count, lines


def check_is_service_line(
    img: ArrayLike,
    line_segments: list[LineSegment],
    kernel_size_ratio: float = 0.3,
    window_size_ratio: float = 0.016,
    middle_ratio: float = 0.1,
    x_overlap_ratio: float = 0.3,
    h_delta_up_ratio: float = 0.028,
    canny_lower_thresh: int = 20,
    canny_upper_thresh: int = 100,
    hough_thresh: int = 20,
    min_line_len_ratio: float = 0.2,
    max_line_gap_ratio: float = 0.1
) -> list[Intersection] | None:
    '''
    sprawdza czy line segment reprezentuje service line
    bierze srodek listy line segmentów (bo tam w tej okolicy jest najpewniej pionowa protopadla linia)
    bierzemy wyciniki po tych segmentach, sprawdzamy czy jest tam prostopadla linia
    jesli tak to bierzemy po obu stronach pionowe linie i szukamy intersekcji z dolna linia horyzontalna
    jesli nic nie znaleziono to oznacza ze to nie jest service line
    '''
    
    img = check_if_numpy_image(img)
    h_delta_up_px = int(h_delta_up_ratio * img.height)

    window_size = int(img.height * window_size_ratio)
    kernel_size = make_odd_kernel_size(window_size, kernel_size_ratio)

    keep = round(len(line_segments) * middle_ratio)
    start = (len(line_segments) - keep) // 2
    line_segments_to_check = line_segments[start:start + keep]

    for ls in line_segments_to_check:
        x_start = min(ls.start.x, ls.end.x)
        x_end = max(ls.start.x, ls.end.x)

        line_width = x_end - x_start
        x_overlap_px = int(line_width * x_overlap_ratio)

        roi = img[
            ls.start.y - h_delta_up_px: ls.start.y, 
            x_start - x_overlap_px:x_end + x_overlap_px
            ]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_blur = cv2.medianBlur(roi_gray, kernel_size)

        min_line_len_px = int(min_line_len_ratio * roi.height)
        max_line_gap_px = int(max_line_gap_ratio * roi.height)

        lines = lines_from_gray_img(
            roi_blur, 
            canny_lower_thresh, 
            canny_upper_thresh, 
            hough_thresh, 
            min_line_len_px, 
            max_line_gap_px
        )

        if get_debug_mode():
            roi_copy = roi.copy()
            for line in lines:
                p1, p2 = line.limit_to_img(roi_copy)
                cv2.line(roi_copy, p1, p2, (255, 0, 0), 1)
            display_img(roi_copy)

        if not lines or lines is None:
            continue

        h_line = sorted(filter_horizontal_lines(lines), key = lambda line: line.intercept, reverse=True)[0]
        v_lines = filter_horizontal_lines(lines, horizontal=False, include_none_slope=True)

        if not v_lines:
            continue

        intersections = []
        for v_line in v_lines:
            angle = angle_between_lines(h_line, v_line)

            if 88 <= angle <= 92 or 268 <= angle <= 272:
                inter = v_line.intersection(h_line, roi)
                global_inter = transform_intersection(inter, roi, ls.start.x, ls.start.y - h_delta_up_px)
                intersections.append(global_inter)

        return intersections
            

def filter_horizontal_lines_by_white_pixels(
    roi: ArrayLike,
    edges: ArrayLike,
    initial_horizontal_lines: list[Line],
    h_margin_img_ratio: float = 0.05,
    w_margin_img_ratio: float = 0.1,
    white_pixel_ratio_thresh: float = 0.07
) -> list[Line]:
    roi = check_if_numpy_image(roi)
    h_margin_px = int(h_margin_img_ratio * roi.height)
    w_margin_px = int(w_margin_img_ratio * roi.width)

    white_ratios = []
    h_lines = []
    for line in initial_horizontal_lines:
        narrow_roi = NumpyImage(roi[:, w_margin_px:-w_margin_px])
        edges_copy = edges.copy()
        narrow_edges = edges_copy[:, w_margin_px:-w_margin_px]
        
        if get_debug_mode():
            narrow_edges_copy = narrow_edges.copy()

        x_coords = np.arange(0, narrow_roi.width)
        y_coords = np.arange(0, narrow_roi.height)

        y_values = line.slope * x_coords + line.intercept
        mask = np.abs(y_values - y_coords[:, np.newaxis]) <= h_margin_px

        narrow_edges[~mask] = 0
        white_pixels_ratio = np.count_nonzero(narrow_edges) / (narrow_edges.width * h_margin_px * 2)
        white_ratios.append(white_pixels_ratio)

        if get_debug_mode():
            mask_img = mask_line_neighborhood_on_edges(narrow_roi, narrow_edges_copy, mask)
            display_img(mask_img)
            print(f"white_pixels_ratio: {white_pixels_ratio}")


        if white_pixels_ratio > white_pixel_ratio_thresh:
            h_lines.append(line)

    return 


def filter_horizontal_lines_by_white_pixels_segment_based(
    roi: ArrayLike,
    edges: ArrayLike,
    initial_horizontal_lines: list[Line],
    line_segments: list[LineSegment],
    h_margin_img_ratio: float = 0.05,
    w_margin_img_ratio: float = 0.1,
    line_intercept_std_ratio: float = 0.02,
    white_column_ratio_thresh: float = 0.5
) -> list[Line]:
    roi = check_if_numpy_image(roi)
    h_margin_px = int(h_margin_img_ratio * roi.height)
    w_margin_px = int(w_margin_img_ratio * roi.width)
    line_intercept_std_px = int(line_intercept_std_ratio * roi.height)

    roi = NumpyImage(roi[:, w_margin_px:-w_margin_px])
    edges = NumpyImage(edges[:, w_margin_px:-w_margin_px])

    white_ratios = []
    white_column_ratios = []
    h_lines = []
    for ls, line in zip(line_segments, initial_horizontal_lines):

        if get_intercept_std(ls) < line_intercept_std_px:
            continue

        mask = np.zeros(
            (roi.height, roi.width),
            dtype=np.uint8
        )

        roi_copy = roi.copy()

        for segment in ls:
            x_start, x_end = sorted((segment.start[0], segment.end[0]))
            y_start, y_end = sorted((segment.start[1], segment.end[1]))

            x_start = x_start - w_margin_px
            x_end = x_end - w_margin_px

            x_start = max(0, x_start)
            x_end = min(roi.width, x_end)
            y_start = max(0, y_start - h_margin_px)
            y_end = min(roi.height, y_end + h_margin_px)

            cv2.rectangle(roi_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

            narrow_edges = edges[y_start:y_end, x_start:x_end]

            if narrow_edges.size == 0:
                continue

            cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)

        if get_debug_mode():
            edges_debug = edges.copy()

        edges_copy = edges.copy()
        edges_copy[mask == 0] = 0

        white_pixels_ratio = calculate_white_pixels_ratio(edges_copy, mask)
        white_ratios.append(white_pixels_ratio)

        white_columns_ratio = calculate_white_columns_ratio(edges_copy, mask)
        white_column_ratios.append(white_columns_ratio)

        if get_debug_mode():
            mask_img = mask_line_neighborhood_on_edges(
                roi,
                edges_debug,
                mask.astype(bool)
            )

            display_img(mask_img)
            print(f"white_pixels_ratio: {white_pixels_ratio}")
            print(f"white_columns_ratio: {white_columns_ratio}")

        if white_columns_ratio > white_column_ratio_thresh:
            h_lines.append(line)

    return h_lines

