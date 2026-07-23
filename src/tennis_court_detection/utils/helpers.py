from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.points import Point, transform_point
from cvgeomkit.geometry.segments import LineSegment, transform_line_segment
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.utils.helpers import load_json, load_yaml
from tennis_court_detection.schemas.config import Params, PicsBlacklist, Direction
from tennis_court_detection.schemas.court import HalfLine, TennisCourtKeyPoints
from tennis_court_detection.utils.constants import COURT_DIMENSIONS
from tennis_court_detection.utils.validators import check_if_numpy_image, validate_number
from tennis_court_detection.config import get_debug_mode


def make_odd_kernel_size(
    window_size: int, 
    kernel_size_ratio: float, 
    min_size: int = 3
) -> int:
    min_size |= 1
    return max(min_size, int(window_size * kernel_size_ratio) | 1)


def crop_center_img(
    img: ArrayLike, 
    crop_ratio: float = 0.4
) -> tuple[ArrayLike, int, int, int]:
    validate_number(crop_ratio, float, 0, 1, min_inclusive=False)
    img = check_if_numpy_image(img)
    w = img.width
    margin = int((1 - crop_ratio) * w / 2)
    crop = img[:, margin:w - margin]
    ch, cw = crop.height, crop.width
    return crop, ch, cw, margin


def lines_from_gray_img(
    img: ArrayLike,
    canny_lower_thresh: int,
    canny_upper_thresh: int,
    hough_thresh: int,
    min_line_len_px: int,
    max_line_gap_px: float
) -> list[Line] | None:
    img = check_if_numpy_image(img)
    edges = cv2.Canny(img, canny_lower_thresh, canny_upper_thresh)
    edges = straighten_rows(edges)
    segments = cv2.HoughLinesP(
        edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = hough_thresh,
        minLineLength = min_line_len_px,
        maxLineGap = max_line_gap_px
    )

    if get_debug_mode():
        display_img(edges)
        img_copy = cv2.merge([img, img, img])
        if segments is not None:
            for segment in segments:
                x1, y1, x2, y2 = segment[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        display_img(img_copy)

    if segments is None:
        return []
    
    return [Line.from_hough_segment(*segment) for segment in segments]


def straighten_rows(
    bin_img : ArrayLike,
    white_ratio_threshold: float = 0.45,
    clear_non_matching: bool = False
) -> ArrayLike:
    validate_number(white_ratio_threshold, float, 0, 1)
    bin_img = check_if_numpy_image(bin_img)
    bin_img_out = bin_img.copy()

    w = bin_img.width

    white_counts = np.count_nonzero(bin_img_out > 0, axis=1)
    threshold = white_ratio_threshold * w

    rows_to_fill = white_counts > threshold
    bin_img_out[rows_to_fill, :] = 255

    if clear_non_matching:
        bin_img_out[~rows_to_fill, :] = 0

    return bin_img_out


def straighten_columns(
    bin_img: ArrayLike,
    white_ratio_threshold: float = 0.45,
    clear_non_matching: bool = False
) -> ArrayLike:
    validate_number(white_ratio_threshold, float, 0, 1)
    bin_img = check_if_numpy_image(bin_img)
    bin_img_out = bin_img.copy()

    h = bin_img.shape[0]

    white_counts = np.count_nonzero(bin_img_out > 0, axis=0)
    threshold = white_ratio_threshold * h

    cols_to_fill = white_counts > threshold
    bin_img_out[:, cols_to_fill] = 255

    if clear_non_matching:
        bin_img_out[:, ~cols_to_fill] = 0

    return bin_img_out


def straighten(
    bin_img: ArrayLike,
    row_white_ratio_threshold: float = 0.5,
    col_white_ratio_threshold: float = 0.3,
    clear_non_matching: bool = False
) -> ArrayLike:
    validate_number(row_white_ratio_threshold, float, 0, 1)
    validate_number(col_white_ratio_threshold, float, 0, 1)

    bin_img = check_if_numpy_image(bin_img)
    h, w = bin_img.height, bin_img.width

    row_white_counts = np.count_nonzero(bin_img > 0, axis=1)
    rows_to_fill = row_white_counts > (row_white_ratio_threshold * w)

    rows_img = np.zeros_like(bin_img)

    if clear_non_matching:
        rows_img[rows_to_fill, :] = 255
    else:
        rows_img = bin_img.copy()
        rows_img[rows_to_fill, :] = 255

    col_white_counts = np.count_nonzero(bin_img > 0, axis=0)
    cols_to_fill = col_white_counts > (col_white_ratio_threshold * h)

    cols_img = np.zeros_like(bin_img)

    if clear_non_matching:
        cols_img[:, cols_to_fill] = 255
    else:
        cols_img = bin_img.copy()
        cols_img[:, cols_to_fill] = 255

    return cv2.bitwise_or(rows_img, cols_img)


def load_process_params(path: Path | str) -> Params:
    data = load_json(path)
    return Params.model_validate(data)


def load_pics_blacklist(path: Path | str) -> PicsBlacklist:
    data = load_yaml(path)
    return PicsBlacklist.model_validate(data)


def angle_between_lines(
    line1: Line, 
    line2: Line
) -> float:
    def line_angle(line):
        if line.xv is not None:
            return 90.0

        return np.degrees(np.arctan2(line.slope, 1)) % 360

    return (line_angle(line2) - line_angle(line1)) % 360


def pipette_color(image: np.ndarray, k: int = 4) -> tuple[int, int, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HSV image with 3 channels")

    X = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(X, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    labels = labels.ravel()
    dom = np.bincount(labels).argmax()
    c = X[labels == dom]

    return tuple(map(int, np.median(c, axis=0)))


def get_next_intersection_by_margin(
    img: ArrayLike,
    intersections: list[Intersection],
    start_intersection: Intersection,
    direction: Direction,
    margin_ratio: float = 0.02
) -> Intersection | None:
    img = check_if_numpy_image(img)
    margin = margin_ratio * img.width
    sorted_intersections = sorted(intersections, key = lambda inter: inter.point.x)
    start_idx = sorted_intersections.index(start_intersection)
    
    if direction == Direction.RIGHT:
        iter_intersections = sorted_intersections[start_idx:]
    else:
        iter_intersections = sorted_intersections[:start_idx][::-1]

    for inter in iter_intersections:
        if start_intersection.point.distance(inter.point) > margin:
            return inter


def get_boundary_horizontal_intercection(
    intersections: list[Intersection], 
    direction: Direction
) -> Intersection:
    sorted_intersections = sorted(intersections, key = lambda inter: inter.point.x)
    idx = 0 if direction == Direction.LEFT else -1
    return sorted_intersections[idx]
    

def compute_intersections_for_line(
    ref_line: Line,
    other_lines: list[Line],
    img: ArrayLike,
    exclude_similar_slope: bool = False
) -> list[Intersection]:
    intersections = []
    for line in other_lines:

        if exclude_similar_slope:
            angle = angle_between_lines(ref_line, line)
            if angle < 10 or 190 > angle > 170 or angle > 350:
                continue

        inter = ref_line.intersection(line, img)
        intersections.append(inter)
    return intersections


def get_opposite_baseline_bounds(
    left_outer_segments: list[LineSegment],
    right_outer_segments: list[LineSegment]
) -> tuple[Point, Point]:
    left_bound = min(left_outer_segments[-1].to_tuple(), key=lambda p: p.x)
    right_bound = max(right_outer_segments[-1].to_tuple(), key=lambda p: p.x)

    baseline_y = min(left_bound.y, right_bound.y)
    
    left_bound = Point(x=left_bound.x, y=baseline_y)
    right_bound = Point(x=right_bound.x, y=baseline_y)

    return left_bound, right_bound


def get_point_from_segments_by_point_y(
    line_segments: list[LineSegment], 
    point: Point, 
    side: str
) -> Point:
    ls_points = []
    for ls in line_segments:
        if (ls.end.y <= point.y <= ls.start.y) or (ls.start.y <= point.y <= ls.end.y):
            ls_points.append((ls.start, ls.end))

    xs = sum([(p1.x, p2.x) for p1, p2 in ls_points], ())

    if not xs:
        raise ValueError('nie znaleziono xs')

    return Point(max(xs) if side == 'left' else min(xs), point.y)


def pair_horizontal_lines(
    img: ArrayLike,
    tolerance_ratio: float,
    left_h_lines: list[HalfLine],
    right_h_lines: list[HalfLine],
) -> list[tuple[HalfLine, HalfLine]]:
    img = check_if_numpy_image(img)
    tol_px = img.height * tolerance_ratio

    left_intercept_arr = np.array([half_line.line.intercept for half_line in left_h_lines])
    right_intercept_arr = np.array([half_line.line.intercept for half_line in right_h_lines])

    dist_matrix = np.abs(left_intercept_arr[:, np.newaxis] - right_intercept_arr)

    is_below_threshold = dist_matrix < tol_px
    is_min_h = dist_matrix == dist_matrix.min(axis=1, keepdims=True)
    is_min_v = dist_matrix == dist_matrix.min(axis=0)

    loc = np.argwhere(is_below_threshold & is_min_h & is_min_v)

    return [(left_h_lines[i], right_h_lines[j]) for i, j in loc]


def get_center_point_from_2_half_lines(
    half_line1: HalfLine,
    half_line2: HalfLine
) -> Point:
    p_x = int(abs(half_line1.point.x - half_line2.point.x))
    p_y = int(abs(half_line1.point.y - half_line2.point.y))
    return Point(p_x, p_y)


def order_ls_points(
    ls: LineSegment
) -> Point:
    p1, p2 = ls.start, ls.end
    if (p1.y, p1.x) <= (p2.y, p2.x):
        return p1, p2
    else:
        return p2, p1


def count_vertical_line_segment_pairs_distances(
    img: ArrayLike,
    line_segments: list[LineSegment],
) -> tuple[LineSegment, LineSegment, int | float]:
    ls_pairs_distances = []
    for ls1, ls2 in combinations(line_segments, 2):
        top1, bottom1 = order_ls_points(ls1)
        top2, bottom2 = order_ls_points(ls2)

        if abs(top1.x - top2.x) < 2 or abs(bottom1.x - bottom2.x) < 2:
            continue

        line1 = Line.from_points(top1, bottom1)
        line2 = Line.from_points(top2, bottom2)

        if line1.intersection(line2, img):
            continue

        if abs(top1.x - top2.x) == abs(bottom1.x - bottom2.x):
            dist = abs(top1.x - top2.x)
        else:
            dist_top = abs(top1.x - top2.x)
            dist_bottom = abs(bottom1.x - bottom2.x)
            dist = (dist_top + dist_bottom) / 2

        ls_pairs_distances.append((ls1, ls2, dist))

    return sorted(ls_pairs_distances, key = lambda item: item[-1])[0] if ls_pairs_distances else None


def traverse_vertical_line(
    img: ArrayLike,
    start_point: Point,
    max_tol_iter: int = 5,
    canny_lower_thresh: int = 20,
    canny_upper_thresh: int = 100,
    hough_thresh: int = 10,
    step_ratio: float = 0.1,
    kernel_size_ratio: float = 0.025,
    roi_width_ratio: float = 0.035,
    roi_height_ratio: float = 0.075,
    min_line_len_ratio: float = 0.2,
    max_line_gap_ratio: float = 0.1
) -> list[list[LineSegment, LineSegment]]:
    
    from tennis_court_detection.utils.filters import filter_horizontal_lines
    
    roi_width_px = int(roi_width_ratio * img.width)
    roi_height_px = int(roi_height_ratio * img.height)
    step_px = int(step_ratio * roi_height_px)

    half_w = roi_width_px // 2
    half_h = roi_height_px // 10

    current_point = start_point
    centre_service_line_segments = []
    tol = 0
    prev_roi_h = -1
    while True:
        x1 = max(0, current_point.x - half_w)
        x2 = min(img.width, current_point.x + half_w)

        y1 = max(0, current_point.y - roi_height_px)
        y2 = min(img.height, current_point.y + half_h)

        if prev_roi_h > 0 and abs(prev_roi_h - (y2 - y1)) > 10:
            return centre_service_line_segments

        roi = img[y1:y2, x1:x2]
        prev_roi_h = roi.height

        if roi.size == 0 or tol >= max_tol_iter:
            return centre_service_line_segments

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        kernel_size_px = int(kernel_size_ratio * roi.height) | 1
        roi_blur = cv2.medianBlur(roi_gray, kernel_size_px)

        min_line_len_px = int(roi.height * min_line_len_ratio)
        max_line_gap_px = int(roi.height * max_line_gap_ratio)

        lines = lines_from_gray_img(
            roi_blur,
            canny_lower_thresh,
            canny_upper_thresh,
            hough_thresh,
            min_line_len_px,
            max_line_gap_px
        )

        if not lines:
            current_point = Point(
                current_point.x,
                current_point.y - step_px
            )
            tol +=1
            continue

        v_lines = filter_horizontal_lines(
            lines,
            horizontal=False,
            include_none_slope=True
        )

        if not v_lines:
            current_point = Point(
                current_point.x,
                current_point.y - step_px
            )
            tol +=1
            continue

        tol = 0
        limit_points = []
        segments = []
        for line in v_lines:
            p1, p2 = line.limit_to_img(roi)
            limit_points.extend([p1, p2])

            ls = LineSegment.from_points(p1, p2)
            segments.append(ls)

        limit_points = sorted(limit_points, key = lambda point: point.y)
        current_point = transform_point(limit_points[0], x1, y1)

        result = count_vertical_line_segment_pairs_distances(roi, segments)

        if result is None:
            tol += 1
            continue

        ls1, ls2, _ = result

        ls1_global = transform_line_segment(ls1, x1, y1)
        ls2_global = transform_line_segment(ls2, x1, y1)

        sorted_segments = sorted([ls1_global, ls2_global], key = lambda ls: ls.start.x) # sortowanie po x zeby odpowiedni ls byl z odpowiedniej strony

        centre_service_line_segments.append(sorted_segments)

        if get_debug_mode():
            roi_copy = roi.copy()
            for line in v_lines:
                p1, p2 = line.limit_to_img(roi)
                cv2.line(roi_copy, p1, p2, (255, 0, 0), 1)

            display_img(roi_copy)

            roi_copy = roi.copy()
            for line in v_lines:
                cv2.line(roi_copy, ls1.start, ls1.end, (255, 0, 0), 1)
                cv2.line(roi_copy, ls2.start, ls2.end, (255, 0, 0), 1)

            display_img(roi_copy)


def point_on_segment(
    point: Point, 
    segment: LineSegment, 
    tol: float = 1.0
) -> bool:
    x_min = min(segment.start.x, segment.end.x) - tol
    x_max = max(segment.start.x, segment.end.x) + tol
    y_min = min(segment.start.y, segment.end.y) - tol
    y_max = max(segment.start.y, segment.end.y) + tol
    return x_min <= point.x <= x_max and y_min <= point.y <= y_max
    

def line_segments_intersections(
    segments1: list[LineSegment],
    segments2: list[LineSegment],
    img: ArrayLike
) -> Intersection | None:
    for i, ls1 in enumerate(segments1):
        for j, ls2 in enumerate(segments2):

            if segments1 is segments2 and j <= i:
                continue

            line1 = Line.from_points(ls1.start, ls1.end)
            line2 = Line.from_points(ls2.start, ls2.end)

            intersection = line1.intersection(line2, img)
            if intersection is None:
                continue

            if point_on_segment(intersection.point, ls1) and point_on_segment(intersection.point, ls2):
                return intersection

            ext_ls1 = LineSegment.from_points(*line1.limit_to_img(img))
            ext_ls2 = LineSegment.from_points(*line2.limit_to_img(img))

            if point_on_segment(intersection.point, ext_ls1) and point_on_segment(intersection.point, ext_ls2):
                return intersection

    return None


def create_reference_court(
    ref_img_height: int = 25_000, 
    ref_img_width: int = 11_000, 
    line_thickness: int = 50
) -> tuple[TennisCourtKeyPoints, np.ndarray]:
    dimensions = COURT_DIMENSIONS
    green = (0, 255, 0)
    red = (255, 0, 0)

    left_x = 0
    right_x = dimensions.width
    inner_left_x = dimensions.dist_outer_sideline
    inner_right_x = dimensions.width - dimensions.dist_outer_sideline
    half_x = dimensions.court_width_half

    top_y = 0
    mid_y = dimensions.court_length_half
    bottom_y = dimensions.length
    service_y = dimensions.dist_from_baseline
    opposite_service_y = dimensions.length - dimensions.dist_from_baseline

    ref_key_points = TennisCourtKeyPoints(
        left_outer_baseline_point=Point.from_iterable((left_x, top_y)),
        left_inner_baseline_point=Point.from_iterable((inner_left_x, top_y)),
        left_outer_netline_point=Point.from_iterable((left_x, mid_y)),
        left_inner_netline_point=Point.from_iterable((inner_left_x, mid_y)),
        right_outer_baseline_point=Point.from_iterable((right_x, top_y)),
        right_inner_baseline_point=Point.from_iterable((inner_right_x, top_y)),
        right_outer_netline_point=Point.from_iterable((right_x, mid_y)),
        right_inner_netline_point=Point.from_iterable((inner_right_x, mid_y)),
        left_service_point=Point.from_iterable((inner_left_x, opposite_service_y)),
        right_service_point=Point.from_iterable((inner_right_x, opposite_service_y)),
        left_service_netline_point=Point.from_iterable((inner_left_x, service_y)),
        right_service_netline_point=Point.from_iterable((inner_right_x, service_y)),
        left_center_service_point=Point.from_iterable((half_x, opposite_service_y)),
        right_center_service_point=Point.from_iterable((half_x, service_y)),
    )

    ref_img = np.zeros((ref_img_height, ref_img_width, 3), np.uint8)

    cv2.line(ref_img, (left_x, top_y), (left_x, mid_y), green, line_thickness)
    cv2.line(ref_img, (left_x, mid_y), (left_x, bottom_y), red, line_thickness)

    cv2.line(ref_img, (right_x, top_y), (right_x, mid_y), green, line_thickness)
    cv2.line(ref_img, (right_x, mid_y), (right_x, bottom_y), red, line_thickness)

    cv2.line(ref_img, (left_x, top_y), (right_x, top_y), green, line_thickness)
    cv2.line(ref_img, (left_x, bottom_y), (right_x, bottom_y), red, line_thickness)

    cv2.line(ref_img, (inner_left_x, top_y), (inner_left_x, mid_y), green, line_thickness)
    cv2.line(ref_img, (inner_left_x, mid_y), (inner_left_x, bottom_y), red, line_thickness)

    cv2.line(ref_img, (inner_right_x, top_y), (inner_right_x, mid_y), green, line_thickness)
    cv2.line(ref_img, (inner_right_x, mid_y), (inner_right_x, bottom_y), red, line_thickness)

    cv2.line(ref_img, (inner_left_x, opposite_service_y), (inner_right_x, opposite_service_y), red, line_thickness)
    cv2.line(ref_img, (inner_left_x, service_y), (inner_right_x, service_y), green, line_thickness)

    cv2.line(ref_img, (half_x, mid_y), (half_x, opposite_service_y), red, line_thickness)
    cv2.line(ref_img, (half_x, mid_y), (half_x, service_y), green, line_thickness)

    cv2.line(ref_img, (left_x, mid_y), (right_x, mid_y), green, line_thickness)

    return ref_key_points, ref_img


def build_input_for_homography_matrix_from_tennis_court_key_points_models(
    ref_points_model: TennisCourtKeyPoints,
    dst_points_model: TennisCourtKeyPoints,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    ref_points_dump = ref_points_model.model_dump()
    dst_points_dump = dst_points_model.model_dump()

    point_names = tuple(
        name
        for name in ref_points_dump.keys()
        if ref_points_dump[name] is not None and dst_points_dump[name] is not None
    )

    if len(point_names) < 4:
        raise ValueError("At least 4 matching points are required to compute homography")

    ref_points_arr = np.array([ref_points_dump[name] for name in point_names], dtype=np.float32)
    dst_points_arr = np.array([dst_points_dump[name] for name in point_names], dtype=np.float32)

    return ref_points_arr, dst_points_arr, point_names