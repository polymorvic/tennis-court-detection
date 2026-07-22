import cv2
from cvgeomkit.common import ArrayLike, Numeric
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.intersections import Intersection, transform_intersection
from tennis_court_detection.schemas.config import ServiceSide
from cvgeomkit.geometry.lines import Line, transform_line
from cvgeomkit.utils.plotting import display_img

from tennis_court_detection.utils.validators import check_if_numpy_image, validate_number
from tennis_court_detection.utils.helpers import lines_from_gray_img, angle_between_lines
from tennis_court_detection.config import get_debug_mode


def filter_horizontal_lines(
    lines: list[Line],
    slope_thresh: float = 0.02,
    horizontal: bool = True,
    include_none_slope: bool = False
) -> list[Line] | None:
    validate_number(slope_thresh, float, 0, 0.2)

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
    validate_number(theta_thresh, float, 0, 20)

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
    middle_ratio: float = 0.3,
    h_delta_up_ratio: float = 0.028,
    h_delta_bottom_ratio: float = 0.005,
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
    if not 0.0 <= middle_ratio <= 1.0:
        raise ValueError("middle_ratio must be between 0.0 and 1.0")
    
    img = check_if_numpy_image(img)

    keep = round(len(line_segments) * middle_ratio)
    start = (len(line_segments) - keep) // 2

    h_delta_up_px = int(h_delta_up_ratio * img.height)
    h_delta_bottom_px = int(h_delta_bottom_ratio * img.height)
    ls_to_check = line_segments[start:start + keep]

    for ls in ls_to_check:
        roi = img[ls.start.y - h_delta_up_px: ls.start.y + h_delta_bottom_px, ls.start.x:ls.end.x]
        min_line_len_px = int(min_line_len_ratio * roi.height)
        max_line_gap_px = int(max_line_gap_ratio * roi.height)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        lines = lines_from_gray_img(
            roi_gray, 
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

        if not lines:
            continue

        h_line = sorted(filter_horizontal_lines(lines), key = lambda line: line.intercept, reverse=True)[0]
        v_lines = filter_horizontal_lines(lines, horizontal=False, include_none_slope=True)

        if not v_lines:
            continue

        intersections = []
        for v_line in v_lines:
            angle = angle_between_lines(h_line, v_line)

            if 85 <= angle <= 95 or 265 <= angle <= 275:
                inter = v_line.intersection(h_line, roi)
                global_inter = transform_intersection(inter, roi, ls.start.x, ls.start.y - h_delta_up_px)
                intersections.append(global_inter)

        return intersections
            
