import cv2
import numpy as np

from cvgeomkit.common import ArrayLike
from cvgeomkit.utils.plotting import display_img

from src.utils.helpers import straighten
from src.config import get_debug_mode


def process_img_for_service_line_detection(
    gray_img: ArrayLike,
    lower_bin_thresh: int,
    upper_bin_thresh: int,
    kernel_height: int = 1,
    kernel_width: int = 25
) -> ArrayLike:
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    bin_img = cv2.inRange(gray_img, lower_bin_thresh, upper_bin_thresh)
    roi_bin_closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    bin_straighten_img = straighten(roi_bin_closed_img, clear_non_matching=True)

    if get_debug_mode():
        print('gray')
        display_img(gray_img)
        print('bin')
        display_img(bin_img)
        print('closed')
        display_img(roi_bin_closed_img)
        print('straight')
        display_img(bin_straighten_img)

    return roi_bin_closed_img


# def process_img_for_netline_detection_clahe(
#     gray_img: ArrayLike,
#     gaussian_sigma: float = 15,
#     clahe_clip_limit: float = 4.0,
#     clahe_tile_grid_size: tuple[int, int] = (8, 8),
#     kernel_height: int = 1,
#     kernel_width: int = 30
# ) -> ArrayLike:
#     kernel = np.ones((kernel_height, kernel_width), np.uint8)

#     background = cv2.GaussianBlur(gray_img, (0,0), gaussian_sigma)
#     enhanced = cv2.subtract(background, gray_img)

#     clahe = cv2.createCLAHE(
#         clipLimit=clahe_clip_limit,
#         tileGridSize=clahe_tile_grid_size
#     )
#     enhanced = clahe.apply(enhanced)
#     enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

#     bin_img = cv2.inRange(enhanced, 200, 255)

#     bin_closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
#     bin_straighten_img = straighten(bin_closed_img, clear_non_matching=True)
    
#     if get_debug_mode():
#         display_img(gray_img)
#         display_img(enhanced)
#         display_img(bin_img)
#         display_img(bin_closed_img)
#         display_img(bin_straighten_img)

#     return bin_straighten_img
