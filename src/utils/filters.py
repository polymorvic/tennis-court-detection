from cvgeomkit.common import ArrayLike, Numeric
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.intersections import Intersection
from src.schemas.config import ServiceSide
from cvgeomkit.geometry.lines import Line

from src.utils.validators import check_if_numpy_image, validate_number


def filter_horizontal_lines(
    lines: list[Line],
    slope_thresh: float = 0.01,
    horizontal: bool = True,
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
            if line.slope is not None and abs(line.slope) > slope_thresh
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