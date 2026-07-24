import cv2
import numpy as np
from cvgeomkit.common import ArrayLike, NumpyImage
from cvgeomkit.utils.plotting import display_img
from cvgeomkit.geometry.lines import transform_line, Line
from cvgeomkit.geometry.points import transform_point, Point
from cvgeomkit.geometry.intersections import compute_intersections, Intersection, transform_intersection
from cvgeomkit.geometry.segments import LineSegment

from tennis_court_detection.schemas.config import ServiceSide, Surface, Direction
from tennis_court_detection.utils.helpers import (
    crop_center_img,
    line_segments_intersections, 
    lines_from_gray_img,
    get_next_intersection_by_margin, 
    get_boundary_horizontal_intercection,
    get_opposite_baseline_bounds,
    angle_between_lines,
    get_point_from_segments_by_point_y,
    pair_horizontal_lines,
    get_center_point_from_2_half_lines,
    traverse_vertical_line,
    create_reference_court,
    build_input_for_homography_matrix_from_tennis_court_key_points_models,
    pair_2_vertical_lines_by_distance,
    line_and_line_segments_intersections
)                        
from tennis_court_detection.utils.filters import (
    filter_horizontal_lines, 
    ensure_is_baseline,
    check_is_service_line
)
from tennis_court_detection.utils.errors import NotEnoughLineSegmentsFound
from tennis_court_detection.utils.adjustments import (
    adjust_horizontal_line,
    adjust_net_line_segments,
    traverse_sideline,
    scan_line_segments_for_horizontal_lines,
    traverse_horizontal_line
)
from tennis_court_detection.config import get_debug_mode
from tennis_court_detection.schemas.court import HalfLine, TennisCourtKeyPoints


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
        self.center_crop_img_gray, *_, self.center_crop_origin_x = crop_center_img(self.img_gray, crop_center_width_ratio)
        self.surface = surface

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
        delta_ensure_height_ratio: float,
        **kwargs
    ) -> tuple[Line, list[Line]] | None:
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

            baseline, is_baseline, sidelines = ensure_is_baseline(
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


    def find_sidelines_segments(
        self,
        baseline_intersections: list[Intersection],
        sidelines_intersections_distance_max_ratio: float = 0.05
    ) -> tuple[LineSegment, LineSegment, LineSegment, LineSegment]:
        temp_img = self.img.copy()
        if self.surface == Surface.CLAY or self.surface == Surface.GRASS:
            temp_img = cv2.bilateralFilter(temp_img, d=9, sigmaColor=30, sigmaSpace=30)


        left_outer_intersection = get_boundary_horizontal_intercection(
            baseline_intersections, 
            Direction.LEFT
        )
        right_outer_intersection = get_boundary_horizontal_intercection(
            baseline_intersections, 
            Direction.RIGHT
        )
        left_inner_intersection = get_next_intersection_by_margin(
            self.img, 
            baseline_intersections, 
            left_outer_intersection, 
            Direction.RIGHT
        )
        right_inner_intersection = get_next_intersection_by_margin(
            self.img, 
            baseline_intersections, 
            right_outer_intersection, 
            Direction.LEFT
        )

        left_points_dist = left_outer_intersection.point.distance(
            left_inner_intersection.point
        )

        right_points_dist = right_outer_intersection.point.distance(
            right_inner_intersection.point
        )

        if abs(left_points_dist - right_points_dist) > sidelines_intersections_distance_max_ratio * self.img.width:
            raise ValueError('edge case cos nie tak z sideline')


        baseline_segments = adjust_horizontal_line(
            temp_img, 
            left_outer_intersection.point, 
            right_outer_intersection.point
        )

        if get_debug_mode():
            img_copy = self.img.copy()
            for p1, p2 in baseline_segments:
                cv2.line(img_copy, p1, p2, (255, 0, 0), 2)
            display_img(img_copy)

        left_outer_segments = traverse_sideline(
            left_outer_intersection,
            temp_img
        )
        left_inner_segments = traverse_sideline(
            left_inner_intersection,
            temp_img
        )
        right_inner_segments = traverse_sideline(
            right_inner_intersection,
            temp_img
        )
        right_outer_segments = traverse_sideline(
            right_outer_intersection,
            temp_img
        )

        return (baseline_segments, left_outer_segments, 
                left_inner_segments, right_inner_segments, 
                right_outer_segments)
    

    def scan_for_horizontal_lines(
        self,
        left_segments: list[LineSegment],
        right_segments: list[LineSegment],
        near_line_tol_ratio: float = 0.01,
        canny_lower_thresh: int = 20,
        canny_upper_thresh: int = 100,
        hough_thresh: int = 50,
        min_line_len_width_ratio: float = 0.5,
        max_line_gap_width_ratio: float = 0.1,
        roi_window_width_ratio: float = 0.1,
        **kwargs,
    ) -> list[LineSegment] | None:
        
        left_h_lines = scan_line_segments_for_horizontal_lines(
            self.img, 
            left_segments, 
            Direction.LEFT,
            roi_window_width_ratio,
            canny_lower_thresh,
            canny_upper_thresh,
            hough_thresh,
            min_line_len_width_ratio,
            max_line_gap_width_ratio,
        )
        
        right_h_lines = scan_line_segments_for_horizontal_lines(
            self.img, 
            right_segments, 
            Direction.RIGHT,
            roi_window_width_ratio,
            canny_lower_thresh,
            canny_upper_thresh,
            hough_thresh,
            min_line_len_width_ratio,
            max_line_gap_width_ratio,
        )

        return pair_horizontal_lines(self.img, near_line_tol_ratio, left_h_lines, right_h_lines)[::-1]


    def find_service_line(
        self,
        service_line_candidate: tuple[HalfLine, HalfLine],
        step_ratio: float = 0.026,
        height_delta_ratio: float = 0.0186,
        kernel_size_ratio: float = 0.3,
        window_size_ratio: float = 0.016,
        middle_ratio: float = 0.1,
        x_overlap_ratio: float = 0.3,
        h_delta_up_ratio: float = 0.028,
        canny_lower_thresh: int = 20,
        canny_upper_thresh: int = 100,
        hough_thresh: int = 20,
        min_line_len_ratio: float = 0.2,
        max_line_gap_ratio: float = 0.1
    ) -> tuple[LineSegment, Intersection] | None:
        hl1, hl2 = service_line_candidate
        ls = adjust_horizontal_line(self.img, hl1.point, hl2.point, step_ratio, height_delta_ratio)
        intersections = check_is_service_line(
            self.img, 
            ls,
            kernel_size_ratio,
            window_size_ratio,
            middle_ratio,
            x_overlap_ratio,
            h_delta_up_ratio,
            canny_lower_thresh,
            canny_upper_thresh,
            hough_thresh,
            min_line_len_ratio,
            max_line_gap_ratio
        )

        if intersections:
            return ls, intersections

        
    def find_centre_service_half_lines(
        self,
        intersection_point: Point,
        max_tol_iter: int = 5,
        canny_lower_thresh: int = 20,
        canny_upper_thresh: int = 100,
        hough_thresh: int = 20,
        kernel_size_ratio: float = 0.1,
        roi_width_ratio: float = 0.025,
        roi_height_up_ratio: float = 0.05,
        roi_height_bottom_ratio: float = 0.01,
        min_line_len_ratio: float = 0.2,
        max_line_gap_ratio: float = 0.1,
        min_v_lines_spread_ratio: float = 0.05,
        adapt_canny_thresh_step: int = 5,
        adapt_hough_thresh_step: int = 2,
        adapt_min_line_len_ratio_step: float = 0.02,
        adapt_max_line_gap_ratio_step: float = 0.02,
    ) -> tuple[HalfLine, HalfLine] | None:
        y_start = intersection_point.y - int(roi_height_up_ratio * self.img.height)
        y_end = intersection_point.y + int(roi_height_bottom_ratio * self.img.height)

        x_start = intersection_point.x - int(roi_width_ratio * self.img.width)
        x_end = intersection_point.x + int(roi_width_ratio * self.img.width)

        roi = self.img[y_start:y_end, x_start:x_end]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        kernel_size_px = int(kernel_size_ratio * roi.height) | 1
        roi_blur = cv2.bilateralFilter(roi_gray, kernel_size_px, 75, 75)

        centre_service_line_intersections = None

        current_canny_lower_thresh = canny_lower_thresh
        current_canny_upper_thresh = canny_upper_thresh
        current_hough_thresh = hough_thresh
        current_min_line_len_ratio = min_line_len_ratio
        current_max_line_gap_ratio = max_line_gap_ratio
        tol_iter = 0
        while not centre_service_line_intersections:

            if tol_iter >= max_tol_iter:
                break

            min_line_len_px = max(1, int(roi.height * current_min_line_len_ratio))
            max_line_gap_px = max(0, int(roi.height * current_max_line_gap_ratio))

            lines = lines_from_gray_img(
                roi_blur,
                current_canny_lower_thresh,
                current_canny_upper_thresh,
                current_hough_thresh,
                min_line_len_px,
                max_line_gap_px
            )

            if lines:
                if get_debug_mode():
                    roi_copy = roi.copy()
                    for line in lines:
                        p1, p2 = line.limit_to_img(roi)
                        cv2.line(roi_copy, p1, p2, (0, 255, 0), 1)

                    display_img(roi_copy)

                v_lines = filter_horizontal_lines(lines, horizontal=False, include_none_slope=True)
                v_lines = [line for line in v_lines if line.slope is None]
                h_lines = filter_horizontal_lines(lines)

                if len(v_lines) >= 2 and h_lines:

                    h_line = sorted(h_lines, key=lambda line: line.intercept)[-1]
                    h_line_global = transform_line(h_line, roi, x_start, y_start)

                    pair = pair_2_vertical_lines_by_distance(roi, v_lines, min_v_lines_spread_ratio)
                    if pair is not None:
                        left_v_line, right_v_line, _ = pair
                        left_v_line_global = transform_line(left_v_line, roi, x_start, y_start)
                        right_v_line_global = transform_line(right_v_line, roi, x_start, y_start)
                        break

            tol_iter += 1

            current_canny_lower_thresh = max(0, current_canny_lower_thresh - adapt_canny_thresh_step)
            current_canny_upper_thresh = min(255, current_canny_upper_thresh + adapt_canny_thresh_step)
            current_hough_thresh = max(1, current_hough_thresh - adapt_hough_thresh_step)
            current_min_line_len_ratio = max(0.0, current_min_line_len_ratio - adapt_min_line_len_ratio_step)
            current_max_line_gap_ratio = min(1.0, current_max_line_gap_ratio + adapt_max_line_gap_ratio_step)

        left_centre_service_point = left_v_line_global.intersection(h_line_global, self.img).point
        right_centre_service_point = right_v_line_global.intersection(h_line_global, self.img).point

        left_hl = HalfLine(point=left_centre_service_point, line=left_v_line_global)
        right_hl = HalfLine(point=right_centre_service_point, line=right_v_line_global)

        if get_debug_mode():
            img_copy = self.img.copy()
            img_copy = left_hl.draw_on_image(img_copy)
            img_copy = right_hl.draw_on_image(img_copy)
            display_img(img_copy)

        return left_hl, right_hl


    def find_netline(
        self,
        baseline_segments: list[LineSegment],
        left_outer_segments: list[LineSegment],
        left_inner_segments: list[LineSegment],
        right_inner_segments: list[LineSegment],
        right_outer_segments: list[LineSegment],
        service_line_segments: list[LineSegment]
    ) -> TennisCourtKeyPoints | None:
        intersections = {
            "left_outer_baseline_point": line_segments_intersections(baseline_segments, left_outer_segments, self.img),
            "left_inner_baseline_point": line_segments_intersections(baseline_segments, left_inner_segments, self.img),
            "right_inner_baseline_point": line_segments_intersections(baseline_segments, right_inner_segments, self.img),
            "right_outer_baseline_point": line_segments_intersections(baseline_segments, right_outer_segments, self.img),
            "left_service_point": line_segments_intersections(left_inner_segments, service_line_segments, self.img),
            "right_service_point": line_segments_intersections(right_inner_segments, service_line_segments, self.img),
        }

        if any(intersection is None for intersection in intersections.values()):
            return None

        court = TennisCourtKeyPoints(**{name: intersection.point for name, intersection in intersections.items()})
        ref_court, _ = create_reference_court()


        ref_points_arr, dst_points_arr, _ = build_input_for_homography_matrix_from_tennis_court_key_points_models(ref_court, court)

        H, _ = cv2.findHomography(ref_points_arr, dst_points_arr)
        if H is None:
            return None

        all_ref_points_arr = np.array([point for point in ref_court.model_dump().values()], dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(all_ref_points_arr.reshape(-1, 1, 2), H)

        transformed_points = transformed_points.squeeze().astype(int)

        transformed_court = TennisCourtKeyPoints.from_matrix(transformed_points)

        raw_netline_points = [transformed_court.left_outer_netline_point, 
                              transformed_court.left_inner_netline_point, 
                              transformed_court.right_inner_netline_point, 
                              transformed_court.right_outer_netline_point]

        if get_debug_mode():
            img_copy = self.img.copy()
            for point in raw_netline_points:
                cv2.circle(img_copy, point, 2, (0, 255, 0), -1)
            display_img(img_copy)

        return adjust_net_line_segments(
            self.img, 
            transformed_court.left_outer_netline_point, 
            transformed_court.right_outer_netline_point,
        )


    def centre_service_half_lines_to_segments(
        self,
        centre_service_half_lines: tuple[HalfLine, HalfLine],
        net_line_segmnets: list[LineSegment]
    ) -> tuple[list[LineSegment], list[LineSegment]]:

        left_service_netline_point = line_and_line_segments_intersections(
            centre_service_half_lines[0].line, 
            net_line_segmnets, 
            self.img
        ).point

        right_service_netline_point = line_and_line_segments_intersections(
            centre_service_half_lines[1].line, 
            net_line_segmnets, 
            self.img
        ).point

        return [LineSegment.from_points(centre_service_half_lines[0].point, left_service_netline_point)], \
                [LineSegment.from_points(centre_service_half_lines[1].point, right_service_netline_point)]
        

      