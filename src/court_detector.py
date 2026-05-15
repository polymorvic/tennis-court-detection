import cv2
import numpy as np

from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img

from src.utils.helpers import (crop_center_img, net_line_scan_params, lines_from_bin_img, 
                               get_horizontal_lines, straighten_rows)
from src.utils.validators import validate_number




class CourtDetector:

    def __init__(self, img: ArrayLike):
        self.img = NumpyImage(img)
        self.center_crop_img, self.center_crop_h, self.center_crop_w = crop_center_img(self.img)
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
        kernel = np.ones((3, 25), np.uint8)
        i = 0
        while y > 0:

            print(i)

            i += 1
            y -= step

            if i < warmup:
                continue

            print('i: ', i)

            roi = crop[y:y + roi_h]
            roi_gray = crop_gray[y:y + roi_h]
            roi_bin = cv2.inRange(roi_gray, 0, 70)
            roi_bin_closed = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel)
            roi_bin_straighten = straighten_rows(roi_bin_closed)

            display_img(roi)
            display_img(roi_bin)
            display_img(roi_bin_closed)
            display_img(roi_bin_straighten)

            



            # get_lines_from_bin_img(roi, black_mask_close, cw)


            # lines_group = group_lines(lines, thresh_intercept=10)
            # netline_local = sorted(lines_group, key = lambda line: line.intercept, reverse=True)[0]

            # print(netline_local)


            # netline_global = transform_line(
            #     original_line=netline_local,
            #     original_img=roi,
            #     original_x_start=margin,
            #     original_y_start=y,
            #     to_global=True
            # )

            # print(netline_global)



            # break

            # print(i)
            # i += 1




