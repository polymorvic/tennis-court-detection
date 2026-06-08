from pathlib import Path
import cv2
import numpy as np
from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.utils.helpers import load_json, load_yaml
from tennis_court_detection.schemas.config import Params, PicsBlacklist, TraverseDirection
from tennis_court_detection.utils.validators import check_if_numpy_image, validate_number
from tennis_court_detection.config import get_debug_mode


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
        return
    
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
    direction: TraverseDirection,
    margin_ratio: float = 0.02
) -> Intersection | None:
    img = check_if_numpy_image(img)
    margin = margin_ratio * img.width
    sorted_intersections = sorted(intersections, key = lambda inter: inter.point.x)
    start_idx = sorted_intersections.index(start_intersection)
    
    if direction == TraverseDirection.RIGHT:
        iter_intersections = sorted_intersections[start_idx:]
    else:
        iter_intersections = sorted_intersections[:start_idx][::-1]

    for inter in iter_intersections:
        if start_intersection.point.distance(inter.point) > margin:
            return inter


def get_boundary_horizontal_intercection(
    intersections: list[Intersection], 
    direction: TraverseDirection
) -> Intersection:
    sorted_intersections = sorted(intersections, key = lambda inter: inter.point.x)
    idx = 0 if direction == TraverseDirection.LEFT else -1
    return sorted_intersections[idx]
    

def compute_intersections_for_line(
    ref_line: Line,
    other_lines: list[Line],
    img: ArrayLike
) -> list[Intersection]:
    intersections = []
    for line in other_lines:
        inter = ref_line.intersection(line, img)
        intersections.append(inter)
    return intersections