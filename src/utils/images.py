import cv2
import numpy as np
from skimage.morphology import skeletonize

from cvgeomkit.common import ArrayLike

from src.utils.helpers import straighten_rows


def process_img_for_netline_detection_threshold(
    gray_img: ArrayLike,
    lower_bin_thresh: int,
    upper_bin_thresh: int,
    kernel_height: int = 3,
    kernel_width: int = 25
):
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    bin_img = cv2.inRange(gray_img, lower_bin_thresh, upper_bin_thresh)
    roi_bin_closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    bin_straighten_img = straighten_rows(roi_bin_closed_img)
    skeleleton = skeletonize(bin_straighten_img)
    skel_img = (skeleleton * 255).astype(np.uint8)
    return bin_img, roi_bin_closed_img, bin_straighten_img, skel_img


def process_img_for_netline_detection_clahe(
    gray_img: ArrayLike,
    gaussian_sigma: float = 15,
    clahe_clip_limit: float = 4.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    adaptive_block_size: int = 31,
    adaptive_c: int = -5,
):
    background = cv2.GaussianBlur(gray_img, (0,0), gaussian_sigma)
    enhanced = cv2.subtract(background, gray_img)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size
    )
    enhanced = clahe.apply(enhanced)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    bin_img = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c
    )

    bin_straighten_img = straighten_rows(bin_img, clear_non_matching=True)
    skeleleton = skeletonize(bin_straighten_img)
    return (skeleleton * 255).astype(np.uint8)
