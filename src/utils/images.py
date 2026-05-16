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


def process_img_for_netline_detection_scoring(
    gray_img: ArrayLike,
    min_width_ratio: float = 0.55,
    min_dark_delta: float = 3.0,
    band_smooth: int = 9,    
):
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    backgroung = cv2.GaussianBlur(blur_img, (1, 31), 0)

    dark = cv2.subtract(backgroung, blur_img).astype(np.float32)
    
    dark = cv2.GaussianBlur(dark, (31, 1), 0)

    mask = dark > min_dark_delta

    coverage = mask.mean(axis=1)

    strength = dark.mean(axis=1)

    score = strength * coverage

    score = cv2.GaussianBlur(
        score.reshape(-1, 1),
        (1, band_smooth),
        0
    ).ravel()

    best_y = int(np.argmax(score))
    best_score = float(score[best_y])
    best_coverage = float(coverage[best_y])

    if best_coverage < min_width_ratio:
        return None, best_score, best_coverage, score

    return best_y, best_score, best_coverage, score