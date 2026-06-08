from pydantic import BaseModel

from cvgeomkit.geometry.segments import LineSegment
from cvgeomkit.geometry.intersections import Intersection


class TennisModel(BaseModel, arbitrary_types_allowed=True):
    pass


class BaselineNeighbourhood(TennisModel):
    adj_baseline: list[LineSegment]
    left_outer_intersection: Intersection
    left_inner_intersection: Intersection
    right_outer_intersection: Intersection
    right_inner_intersection: Intersection