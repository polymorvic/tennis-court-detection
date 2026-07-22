from pydantic import BaseModel

from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.points import Point
from cvgeomkit.geometry.lines import Line
from cvgeomkit.geometry.intersections import Intersection


class TennisModel(BaseModel, arbitrary_types_allowed=True):
    pass


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