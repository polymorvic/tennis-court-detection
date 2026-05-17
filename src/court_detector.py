import cv2

from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line
from cvgeomkit.geometry.points import transform_point
from cvgeomkit.geometry.intersections import compute_intersections

from src.schemas.config import ServiceSide
from src.utils.helpers import crop_center_img, lines_from_bin_img
                              
from src.utils.filters import get_horizontal_lines, get_vertical_lines, get_centre_vertical_lines, filter_service_intersections
from src.utils.images import process_img_for_service_line_detection

from src.config import get_debug_mode


class CourtDetector:

    def __init__(self, img: ArrayLike):
        self.img = NumpyImage(img)
        self.center_crop_img, self.center_crop_h, self.center_crop_w, self.center_crop_margin = crop_center_img(self.img)
        self.center_crop_img_gray = cv2.cvtColor(self.center_crop_img, cv2.COLOR_RGB2GRAY)


    def scan_for_service_lines(
        self,
        service_side: ServiceSide,
        roi_h: int = 80, 
        step: int = 20, 
        warmup: int = 15,
        canny_lower_thresh: int = 25,
        canny_upper_thresh: int = 100,
        hough_thresh: int = 50,
        min_line_len_ratio: float = 0.05,
        min_line_gap_px: int = 5,
        vertical_center_delta_px: int = 100,
        white_line_bin_lower_thresh: int = 150,
        white_line_bin_upper_thresh: int = 255,
    ):
        ch = self.center_crop_h
        cw = self.center_crop_w
        crop = self.center_crop_img.copy()
        crop_gray = self.center_crop_img_gray.copy()
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

            roi = crop[y:y + roi_h].copy()

            if roi.size == 0:
                return None

            roi_gray = crop_gray[y:y + roi_h].copy()
            roi_bin = process_img_for_service_line_detection(roi_gray, white_line_bin_lower_thresh, white_line_bin_upper_thresh)

            line_candidates = lines_from_bin_img(
                roi_bin, 
                cw, 
                canny_lower_thresh, 
                canny_upper_thresh, 
                hough_thresh, 
                min_line_len_ratio, 
                min_line_gap_px
            )
            if not line_candidates:
                continue

            service_line_candidates = get_horizontal_lines(line_candidates)
            
            if get_debug_mode():
                print('service_line_candidates ---')
                print(service_line_candidates)

            if not service_line_candidates:
                continue

            centre_service_line_candidates = get_vertical_lines(line_candidates)

            if get_debug_mode():
                print('centre_service_line_candidates before ---')
                print(centre_service_line_candidates)

            if not centre_service_line_candidates:
                continue

            centre_service_line_candidates = get_centre_vertical_lines(centre_service_line_candidates, roi, vertical_center_delta_px)
            
            if get_debug_mode():
                print('centre_service_line_candidates after ---')
                print(centre_service_line_candidates)

            if not centre_service_line_candidates:
                continue

            intersections = set(compute_intersections(line_candidates, roi))
            if not intersections:
                continue

            if get_debug_mode():
                print('intersections')
                print(intersections)

            filtered = filter_service_intersections(
                intersections,
                service_line_candidates,
                centre_service_line_candidates,
                service_side
            )

            if filtered is None:
                continue

            if get_debug_mode():
                print('filtered intersections')
                print(filtered)

            _, _, service_point_candidate = filtered
            detection_y = y + service_point_candidate.y
            if not 450 <= detection_y <= 720:
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

            break

        if (
            service_line_local is None
            and centre_service_line_local is None
            and service_point_local is None
        ):
            return None

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









