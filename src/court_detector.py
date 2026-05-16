import cv2
import numpy as np

from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line
from cvgeomkit.geometry.intersections import compute_intersections

from src.schemas.config import ServiceSide
from src.utils.helpers import (crop_center_img, service_line_scan_params, lines_from_bin_img, 
                               get_horizontal_lines, get_vertical_lines)
from src.utils.images import process_img_for_service_line_detection
from src.utils.validators import validate_number

from src.config import get_debug_mode




class CourtDetector:

    def __init__(self, img: ArrayLike):
        self.img = NumpyImage(img)
        self.center_crop_img, self.center_crop_h, self.center_crop_w, self.center_crop_margin = crop_center_img(self.img)
        self.center_crop_img_gray = cv2.cvtColor(self.center_crop_img, cv2.COLOR_RGB2GRAY)


    def scan_for_service_line(
        self,
        service_side: ServiceSide,
        roi_h: int | None = None, 
        step: int | None = None, 
        warmup: int = None
    ):
        img_h = self.img.height
        default_roi_h, default_step, default_warmup = service_line_scan_params(img_h)

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
        service_line_local = None
        while y > 0:
            i += 1
            y -= step

            if i < warmup:
                continue

            roi = crop[y:y + roi_h]
            roi_gray = crop_gray[y:y + roi_h]
            roi_bin = process_img_for_service_line_detection(roi_gray, 150, 255)

            line_candidates = lines_from_bin_img(roi_bin, cw, 25, 100, 50, 0.05, 5)
            if line_candidates is None:
                continue

            service_line_candidates = get_horizontal_lines(line_candidates)
            if service_line_candidates is None:
                continue

            centre_service_line_candidates = get_vertical_lines(line_candidates)
            if centre_service_line_candidates is None:
                continue

            service_line_local = sorted(service_line_candidates, key = lambda line: line.intercept, reverse=True)[0]

            intersections = set(compute_intersections(line_candidates, roi))
            if intersections is None:
                continue

            for intersection in intersections:
                line1 = intersection.line1
                line2 = intersection.line2

                angle = intersection._compute_angle(line1, line2)



                print(line1, line2, angle)


            

            

            if service_side == ServiceSide.LEFT:
                pass

            else:
                pass

            # if service_line_local:
            #     break

            if get_debug_mode():
                print('intersections ---')
                print(intersections)

                print('line candidates ---')
                print(line_candidates)


                print('service_line_candidates ---')
                print(service_line_candidates)

                print('centre_service_line_candidates ---')
                print(centre_service_line_candidates)

                roi_copy = roi.copy()
                for line in line_candidates:
                    p1, p2 = line.limit_to_img(roi_copy)
                    cv2.line(roi_copy, p1, p2, (255, 0, 0), 2)

                display_img(roi_copy)

            netline_global = transform_line(
                original_line=service_line_local,
                original_img=roi,
                original_x_start=self.center_crop_margin,
                original_y_start=y,
                to_global=True
            )

            print(netline_global)








