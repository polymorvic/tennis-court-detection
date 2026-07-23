from pydantic import BaseModel

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
    left_outer_baseline_point: Point
    left_inner_baseline_point: Point
    left_outer_netline_point: Point
    left_inner_netline_point: Point
    right_outer_baseline_point: Point
    right_inner_baseline_point: Point
    right_outer_netline_point: Point
    right_inner_netline_point: Point
    left_service_point: Point
    right_service_point: Point
    left_service_netline_point: Point
    right_service_netline_point: Point
    left_center_service_point: Point
    right_center_service_point: Point


class TennisCourt(TennisModel):
    segments: TennisCourtLineSegments
    key_points: TennisCourtKeyPoints


class BaselineNeighbourhood(TennisModel):
    adj_baseline: list[LineSegment]
    left_outer_intersection: Intersection
    left_inner_intersection: Intersection
    right_outer_intersection: Intersection
    right_inner_intersection: Intersection


class HalfLine(TennisModel):
    point: Point
    line: Line

    def __hash__(self):
        return self.line.__hash__()
    
    def __eq__(self, other):
        return self.line == other.line