import cv2
import numpy as np

from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line

from src.utils.helpers import (crop_center_img, net_line_scan_params, lines_from_bin_img, 
                               get_horizontal_lines)
from src.utils.images import process_img_for_netline_detection_threshold, process_img_for_netline_detection_clahe
from src.utils.validators import validate_number

from src.config import get_debug_mode




class CourtDetector:

    def __init__(self, img: ArrayLike):
        self.img = NumpyImage(img)
        self.center_crop_img, self.center_crop_h, self.center_crop_w, self.center_crop_margin = crop_center_img(self.img)
        self.center_crop_img_gray = cv2.cvtColor(self.center_crop_img, cv2.COLOR_RGB2GRAY)


    def scan_for_net_line(
        self,
        roi_h: int | None = None, 
        step: int | None = None, 
        warmup: int = None
    ):
        img_h = self.img.height
        default_roi_h, default_step, default_warmup = net_line_scan_params(img_h)

        roi_h = default_roi_h if roi_h is None else roi_h
        step = default_step if step is None else step
        warmup = default_warmup if warmup is None else warmup

        validate_number(roi_h, int, 0, img_h// 2)
        validate_number(step, int, 0, img_h // 20)
        validate_number(warmup, int, 0, img_h // 40)

        ch = self.center_crop_h
        cw = self.center_crop_w
        crop = self.center_crop_img
        crop_gray = self.center_crop_img_gray
        y = ch - roi_h
        i = 0
        netline_local = None
        while y > 0:
            i += 1
            y -= step

            if i < warmup:
                continue

            roi = crop[y:y + roi_h]
            roi_gray = crop_gray[y:y + roi_h]

            # roi_bin = process_img_for_netline_detection_threshold(roi_gray, 0, 70)
            roi_bin = process_img_for_netline_detection_clahe(roi_gray)

            net_line_candidates = lines_from_bin_img(roi_bin, cw, 25, 100, 100, 0.05, 30)
            if net_line_candidates is None:
                continue

            net_line_candidates = get_horizontal_lines(net_line_candidates)
            if net_line_candidates is None:
                continue

            netline_local = sorted(net_line_candidates, key = lambda line: line.intercept, reverse=True)[0]

            if netline_local:
                break

        if get_debug_mode():
            print(netline_local)

            roi_copy = roi.copy()
            p1, p2 = netline_local.limit_to_img(roi_copy)
            cv2.line(roi_copy, p1, p2, (255, 0, 0), 2)

            display_img(roi_copy)

            netline_global = transform_line(
                original_line=netline_local,
                original_img=roi,
                original_x_start=self.center_crop_margin,
                original_y_start=y,
                to_global=True
            )

            print(netline_global)








