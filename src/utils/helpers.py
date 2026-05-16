import cv2
import numpy as np
from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.utils.plotting import display_img

from src.schemas.config import ServiceSide
from src.utils.validators import check_if_numpy_image, validate_number
from src.config import get_debug_mode


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


def service_line_scan_params(height: int) -> tuple[int, int, int]:
    roi_h = height // 5
    step = max(1, height // 20)
    warmup = max(1, height // 40)
    return roi_h, step, warmup


def lines_from_bin_img(
    bin_img: ArrayLike,
    crop_img_width: int,
    canny_lower_thresh: int,
    canny_upper_thresh: int,
    hough_thresh: int,
    min_line_len_ratio: float,
    min_line_gap: int
) -> list[Line] | None:
    edges = cv2.Canny(bin_img, canny_lower_thresh, canny_upper_thresh)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=int(min_line_len_ratio * crop_img_width),
        maxLineGap=min_line_gap
    )

    if get_debug_mode():
        display_img(edges)

        img_copy = cv2.merge([bin_img, bin_img, bin_img])
        if segments is not None:
            for segment in segments:
                x1, y1, x2, y2 = segment[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        display_img(img_copy)

    if segments is None:
         print('nic')
         return
    
    return [Line.from_hough_line(*segment) for segment in segments]


def get_horizontal_lines(
    lines: list[Line],
    slope_thresh: float = 0.01
) -> list[Line] | None:
    validate_number(slope_thresh, float, 0, 0.2)
    lines = [line for line in lines if line.slope is not None and abs(line.slope) < slope_thresh]
    return lines if lines else None


def get_vertical_lines(
    lines: list[Line],
    theta_thresh: float = 1.0
) -> list[Line] | None:
    validate_number(theta_thresh, float, 0, 20)

    lines = [
        line for line in lines
        if abs(line.theta - 90) < theta_thresh
    ]

    return lines if lines else None


def filter_service_lines(
    intersections: set[Intersection],
    service_line_candidates: list[Line],
    vertical_candidates: list[Line],
    roi: ArrayLike,
    service_side: ServiceSide,
    vert_cluster_dist: int = 25,
) -> tuple[Line, Line, Point] | None:
    
    w, h = roi.width, roi.height
    cx, cy = w / 2, h / 2

    service_line = max(service_line_candidates, key=lambda line: line.intercept)

    ref = min(vertical_candidates, key=lambda line: abs(line.xv - cx))
    cluster = [line for line in vertical_candidates if abs(line.xv - ref.xv) <= vert_cluster_dist]
    if service_side == ServiceSide.LEFT:
        centre_service_line = max(cluster, key=lambda line: line.xv)
    else:
        centre_service_line = min(cluster, key=lambda line: line.xv)

    perp_near_center = []
    for ix in intersections:
        lines = (ix.line1, ix.line2)
        has_horiz = any(line.slope is not None and abs(line.slope) < 0.05 for line in lines)
        has_vert = any(line.xv is not None for line in lines)
        if not (has_horiz and has_vert):
            continue
        dist = (ix.point.x - cx) ** 2 + (ix.point.y - cy) ** 2
        perp_near_center.append((dist, ix))

    if not perp_near_center:
        return None

    point = min(perp_near_center, key=lambda item: item[0])[1].point
    return service_line, centre_service_line, point


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