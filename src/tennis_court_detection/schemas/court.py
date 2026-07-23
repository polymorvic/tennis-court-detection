import cv2
import numpy as np
from pydantic import BaseModel

from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.intersections import Intersection


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


class TennisCourtKeyPoints(TennisModel):
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
    left_service_netline_point: Point | None = None
    right_service_netline_point: Point | None = None
    left_center_service_point: Point | None = None
    right_center_service_point: Point | None = None


    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "TennisCourtKeyPoints":
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
            left_service_netline_point=Point.from_iterable(matrix[10]),
            right_service_netline_point=Point.from_iterable(matrix[11]),
            left_center_service_point=Point.from_iterable(matrix[12]),
            right_center_service_point=Point.from_iterable(matrix[13]),
        )


    def draw_on_image(
        self,
        img: ArrayLike,
        color: tuple[int, int, int] = (0, 255, 255),
        radius: int = 8,
        thickness: int = -1,
    ) -> np.ndarray:
        img_copy = img.copy()
        
        for point in self.model_dump().values():
            if point is None:
                continue
            cv2.circle(img_copy, (int(point.x), int(point.y)), radius, color, thickness)

        return img_copy


class TennisCourt(TennisModel):
    segments: TennisCourtLineSegments
    key_points: TennisCourtKeyPoints


class HalfLine(TennisModel):
    point: Point
    line: Line

    def __hash__(self):
        return self.line.__hash__()
    
    def __eq__(self, other):
        return self.line == other.line