import cv2
import numpy as np
from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line
from cvgeomkit.geometry.points import transform_point
from cvgeomkit.geometry.intersections import compute_intersections

from tennis_court_detection.schemas.config import ServiceSide, Surface
from tennis_court_detection.utils.helpers import crop_center_img, lines_from_gray_img
                              
from tennis_court_detection.utils.filters import (filter_horizontal_lines, get_vertical_lines, get_centre_vertical_lines, 
                               filter_service_intersections, ensure_is_baseline)
from tennis_court_detection.utils.images import process_img_for_service_line_detection

from tennis_court_detection.config import get_debug_mode


class CourtDetector:

    def __init__(
        self, 
        img: ArrayLike,
        crop_center_width_ratio: float,
        roi_height_ratio: float,
        step_height_ratio: float,
        surface: Surface
    ):
        self.img = NumpyImage(img)
        self.img_gray = NumpyImage(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY))
        self.roi_h_px = int(roi_height_ratio * self.img.height)
        self.step_px = int(step_height_ratio * self.img.height)
        self.center_crop_img, self.center_crop_h, self.center_crop_w, self.center_crop_margin = crop_center_img(self.img, crop_center_width_ratio)
        self.center_crop_img_gray = crop_center_img(self.img_gray, crop_center_width_ratio)[0]

        if surface == Surface.CLAY or surface == Surface.GRASS:
            self.center_crop_img_gray = cv2.bilateralFilter(self.center_crop_img_gray, d=9, sigmaColor=30, sigmaSpace=30)


    def scan_for_baseline(
        self,
        warmup_height_ratio: float,
        canny_lower_thresh: int,
        canny_upper_thresh: int,
        canny_lower_thresh_offset: int,
        canny_upper_thresh_offset: int,
        hough_thresh: int,
        hough_thresh_offset: int,
        min_line_len_width_ratio: float,
        min_line_len_ensure_width_ratio: float,
        max_line_gap_width_ratio: float,
        horizontal_line_slope_tolerance: float,
        delta_ensure_height_ratio: float

    ):
        warmup = int(self.img.height / self.step_px * warmup_height_ratio)
        ch = self.center_crop_h
        crop = self.center_crop_img.copy()
        crop_gray = self.center_crop_img_gray.copy()
        y = ch - self.roi_h_px
        i = 0
        baseline = None
        lines_blacklist = set()
        while y > 0:
            i += 1
            y -= self.step_px

            if i < warmup:
                continue

            roi = crop[y:y + self.roi_h_px].copy()

            if roi.size == 0:
                return None

            roi_gray = crop_gray[y:y + self.roi_h_px].copy()

            min_line_len_px = int(min_line_len_width_ratio * roi.width)
            max_line_gap_px = int(max_line_gap_width_ratio * roi.width)
            lines = lines_from_gray_img(
                roi_gray, 
                canny_lower_thresh, 
                canny_upper_thresh,
                hough_thresh, 
                min_line_len_px,
                max_line_gap_px
            )
            if not lines:
                continue

            if get_debug_mode():
                print(lines)

            baseline_candidates = filter_horizontal_lines(lines, horizontal_line_slope_tolerance)

            if get_debug_mode():
                print(baseline_candidates)

            if not baseline_candidates:
                continue

            baseline_candidate = sorted(baseline_candidates, key=lambda line: line.intercept, reverse=True)[0]
            baseline = transform_line(baseline_candidate, roi, self.center_crop_margin, y)

            if get_debug_mode():
                print('baseline global')
                print(baseline)

            if baseline in lines_blacklist:
                continue

            min_line_len_px = int(min_line_len_ensure_width_ratio * roi.width)
            max_line_gap_px = 0 
            scoreboard_lines = lines_from_gray_img(
                roi_gray,
                canny_lower_thresh + canny_lower_thresh_offset,
                canny_upper_thresh + canny_upper_thresh_offset,
                hough_thresh + hough_thresh_offset,
                min_line_len_px,
                max_line_gap_px,
            )

            is_scoreboard = False
            if scoreboard_lines:
                intersections = set(compute_intersections(scoreboard_lines, roi))
                
                if intersections:
                    for inters in intersections:
                        
                        if abs(inters.angle % 180 - 90) == 0:
                            is_scoreboard = True
                            lines = [inters.line1, inters.line2]
                            h_line_local = [line for line in lines if line.slope is not None and abs(line.slope) < horizontal_line_slope_tolerance]
                            if not h_line_local:
                                continue
                            h_line_global = transform_line(h_line_local[0], roi, self.center_crop_margin, y)
                            lines_blacklist.add(h_line_global)

            if is_scoreboard:
                baseline = None
                continue

            is_baseline, sidelines = ensure_is_baseline(
                baseline, 
                self.img_gray,
                roi.width,
                canny_lower_thresh + canny_lower_thresh_offset, 
                canny_upper_thresh + canny_upper_thresh_offset,
                hough_thresh, 
                min_line_len_width_ratio,
                max_line_gap_width_ratio,
                delta_ensure_height_ratio,
            )
            
            if not is_baseline:
                lines_blacklist.add(baseline)
                baseline = None
                continue

            break

        return baseline, sidelines

