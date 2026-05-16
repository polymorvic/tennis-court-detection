import cv2
import numpy as np

from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line
from cvgeomkit.geometry.points import transform_point
from cvgeomkit.geometry.intersections import compute_intersections

from src.schemas.config import ServiceSide
from src.utils.helpers import (crop_center_img, service_line_scan_params, lines_from_bin_img,
                               get_horizontal_lines, get_vertical_lines, get_centre_vertical_lines,
                               filter_service_intersections)
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
        centre_service_line_local = None
        service_point_local = None
        while y > 0:
            i += 1
            y -= step

            if i < warmup:
                continue

            roi = crop[y:y + roi_h]
            roi_gray = crop_gray[y:y + roi_h]
            roi_bin = process_img_for_service_line_detection(roi_gray, 150, 255)

            line_candidates = lines_from_bin_img(roi_bin, cw, 25, 100, 50, 0.05, 5)
            if not line_candidates:
                continue

            service_line_candidates = get_horizontal_lines(line_candidates)
            if not service_line_candidates:
                continue

            centre_service_line_candidates = get_vertical_lines(line_candidates)
            if not centre_service_line_candidates:
                continue

            centre_service_line_candidates = get_centre_vertical_lines(centre_service_line_candidates, roi)
            if not centre_service_line_candidates:
                continue

            intersections = set(compute_intersections(line_candidates, roi))

            if not intersections:
                continue

            filtered = filter_service_intersections(
                intersections,
                service_line_candidates,
                centre_service_line_candidates,
                service_side
            )

            if filtered is None:
                continue

            service_line_local, centre_service_line_local, service_point_local = filtered

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

            if any(x is not None for x in (
                service_line_local,
                centre_service_line_local,
                service_point_local
            )):
                
                break


        service_line_global = transform_line(
            service_line_local,
            roi,
            self.center_crop_margin,
            y
        )

        centre_service_line_global = transform_line(
            centre_service_line_local,
            roi,
            self.center_crop_margin,
            y
        )

        service_point_global = transform_point(
            service_point_local,
            self.center_crop_margin,
            y,
        )


        return service_line_global, centre_service_line_global, service_point_global









