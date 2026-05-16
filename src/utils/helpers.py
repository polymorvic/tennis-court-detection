import cv2
import numpy as np
from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.lines import Line


from src.utils.validators import check_if_numpy_image, validate_number


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


def net_line_scan_params(height: int) -> tuple[int, int, int]:
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

    if segments is None:
         return
    
    return [Line.from_hough_line(*segment) for segment in segments]


def get_horizontal_lines(
    lines: list[Line],
    slope_thresh: float = 0.1
) -> list[Line] | None:
    validate_number(slope_thresh, float, 0, 0.2)
    lines = [line for line in lines if line.slope is not None and abs(line.slope) < slope_thresh]
    return lines if lines else None


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


