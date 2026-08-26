import cv2
import numpy as np
from pydantic import BaseModel

from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.intersections import Intersection
from cvgeomkit.utils.plotting import display_img


class TennisModel(BaseModel, arbitrary_types_allowed=True):
    pass


class TennisCourtLineSegments(TennisModel):
    baseline_segments: list[LineSegment]
    left_outer_segments: list[LineSegment]
    left_inner_segments: list[LineSegment]
    right_outer_segments: list[LineSegment]
    right_inner_segments: list[LineSegment]
    service_line_segments: list[LineSegment]
    centre_service_line_segments: list[LineSegment]
    netline_segments: list[LineSegment]


class ReferenceCourtTennisCourtKeyPoints(TennisModel):
    left_outer_baseline_point: Point | None = None
    left_inner_baseline_point: Point | None = None
    left_outer_netline_point: Point | None = None
    left_inner_netline_point: Point | None = None
    right_outer_baseline_point: Point | None = None
    right_inner_baseline_point: Point | None = None
    right_outer_netline_point: Point | None = None
    right_inner_netline_point: Point | None = None
    left_service_point: Point | None = None
    right_service_point: Point | None = None
    avg_centre_service_point: Point | None = None
    avg_service_netline_point: Point | None = None
    left_outer_baseline_point_opposite: Point | None = None
    left_inner_baseline_point_opposite: Point | None = None
    right_outer_baseline_point_opposite: Point | None = None
    right_inner_baseline_point_opposite: Point | None = None
    left_service_point_opposite: Point | None = None
    right_service_point_opposite: Point | None = None
    avg_centre_service_point_opposite: Point | None = None

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "ReferenceCourtTennisCourtKeyPoints":
        return cls(
            left_outer_baseline_point=Point.from_iterable(matrix[0]),
            left_inner_baseline_point=Point.from_iterable(matrix[1]),
            left_outer_netline_point=Point.from_iterable(matrix[2]),
            left_inner_netline_point=Point.from_iterable(matrix[3]),
            right_outer_baseline_point=Point.from_iterable(matrix[4]),
            right_inner_baseline_point=Point.from_iterable(matrix[5]),
            right_outer_netline_point=Point.from_iterable(matrix[6]),
            right_inner_netline_point=Point.from_iterable(matrix[7]),
            left_service_point=Point.from_iterable(matrix[8]),
            right_service_point=Point.from_iterable(matrix[9]),
            avg_centre_service_point=Point.from_iterable(matrix[10]),
            avg_service_netline_point=Point.from_iterable(matrix[11]),
            left_outer_baseline_point_opposite = Point.from_iterable(matrix[12]),
            left_inner_baseline_point_opposite = Point.from_iterable(matrix[13]),
            right_outer_baseline_point_opposite = Point.from_iterable(matrix[14]),
            right_inner_baseline_point_opposite = Point.from_iterable(matrix[15]),
            left_service_point_opposite = Point.from_iterable(matrix[16]),
            right_service_point_opposite = Point.from_iterable(matrix[17]),
            avg_centre_service_point_opposite = Point.from_iterable(matrix[18]),
        )


    def draw_on_image(
        self,
        img: ArrayLike,
        dst_points: list[str],
        color: tuple[int, int, int] = (0, 255, 255),
        radius: int = 8,
        thickness: int = -1,
        with_lines: bool = False,
        lines_color: tuple[int, int, int] = (0, 255, 0),
        lines_thickness: int = 2
    ) -> None:
        img_copy = img.copy()

        if with_lines:

            cv2.line(img_copy, self.left_outer_baseline_point, self.left_inner_baseline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_outer_baseline_point, self.right_inner_baseline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_inner_baseline_point_opposite, self.left_inner_baseline_point_opposite, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_outer_baseline_point, self.left_outer_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_inner_baseline_point, self.left_service_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_inner_baseline_point, self.left_inner_baseline_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.left_inner_netline_point, self.left_service_point_opposite, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_outer_baseline_point_opposite, self.left_inner_baseline_point_opposite, lines_color, lines_thickness)

            cv2.line(img_copy, self.left_service_point_opposite, self.avg_centre_service_point_opposite, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_service_point_opposite, self.avg_centre_service_point_opposite, lines_color, lines_thickness)

            cv2.line(img_copy, self.avg_centre_service_point_opposite, self.avg_service_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.avg_service_netline_point, self.avg_centre_service_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.right_inner_baseline_point, self.right_service_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_outer_baseline_point, self.right_outer_netline_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.right_inner_baseline_point_opposite, self.right_outer_baseline_point_opposite, lines_color, lines_thickness)

            cv2.line(img_copy, self.right_outer_baseline_point_opposite, self.right_outer_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_outer_baseline_point_opposite, self.left_outer_netline_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.right_inner_baseline_point_opposite, self.right_service_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_inner_baseline_point_opposite, self.left_service_point_opposite, lines_color, lines_thickness)

            cv2.line(img_copy, self.left_service_point, self.left_inner_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_service_point, self.right_inner_netline_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.left_service_point, self.avg_centre_service_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_service_point, self.avg_centre_service_point, lines_color, lines_thickness)

            cv2.line(img_copy, self.left_outer_netline_point, self.left_inner_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_outer_netline_point, self.right_inner_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.left_inner_netline_point, self.avg_service_netline_point, lines_color, lines_thickness)
            cv2.line(img_copy, self.right_inner_netline_point, self.avg_service_netline_point, lines_color, lines_thickness)

        for point in self.model_dump().values():
            if point is None:
                continue
            cv2.circle(img_copy, (int(point.x), int(point.y)), radius, color, thickness)

        if dst_points is not None:
            for point_name in dst_points:
                cv2.circle(img_copy, getattr(self, point_name), radius, (255, 0, 0), thickness)

        display_img(img_copy)


class TennisCourt(TennisModel):
    segments: TennisCourtLineSegments
    key_points: ReferenceCourtTennisCourtKeyPoints


class HalfLine(TennisModel):
    point: Point
    line: Line

    def __hash__(self):
        return self.line.__hash__()
    
    def __eq__(self, other):
        return self.line == other.line

    def draw_on_image(
        self, 
        img: ArrayLike, 
        radius: int = 2,
        color: tuple[int, int, int] = (0, 255, 255), 
        thickness: int = 1
    ) -> np.ndarray:
        img_copy = img.copy()
        p1, p2 = self.line.limit_to_img(img_copy)
        cv2.circle(img_copy, self.point, radius, color, thickness)
        cv2.line(img_copy, p1, p2, color, thickness)
        return img_copy