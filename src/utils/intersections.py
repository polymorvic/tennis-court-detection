from typing import TYPE_CHECKING, Self

import numpy as np

from .common import Hashable, ArrayLike
from .points import transform_point

if TYPE_CHECKING:
    from .lines import Line, transform_line
    from .points import Point


class Intersection(Hashable):

    def __init__(self, line1: "Line", line2: "Line", intersection_point: "Point") -> None:
        self.line1 = line1
        self.line2 = line2
        self.point = intersection_point
        self.angle = self._compute_angle(self.line1, self.line2)

    def __repr__(self) -> str:

        def format_line(line: "Line") -> str:
            """Helper function to format a line equation."""
            if line.xv is not None:
                return f"x = {line.xv:.2f}"
            else:
                return f"y = {line.slope:.2f} * x + {line.intercept:.2f}"

        lines = [self.line1, self.line2]
        lines.sort(key=lambda line: line.slope if line.slope is not None else np.inf)

        line1_eq = format_line(lines[0])
        line2_eq = format_line(lines[1])

        return f"Point {self.point} line1: [{line1_eq}] line2: [{line2_eq}]"

    def _key_(self) -> tuple["Point", tuple[float, float]]:

        def sort_key(line: "Line") -> tuple[float, float]:
            primary = line.slope if line.slope is not None else np.inf
            secondary = line.xv if line.xv is not None else -np.inf
            return (primary, secondary)

        lines = [self.line1, self.line2]
        lines.sort(key=sort_key)

        line_keys = [line._key_() for line in lines]
        return (self.point, tuple(line_keys))

    def distance(self, another_intersection: Self) -> float:
        return self.point.distance(another_intersection.point)

    def other_line(self, used: "Line") -> "Line":
        if self.line1 is used or self.line1._key_() == used._key_():
            return self.line2
        if self.line2 is used or self.line2._key_() == used._key_():
            return self.line1
        raise ValueError("The provided line does not belong to this intersection.")

    def _compute_angle(self, line1: "Line", line2: "Line") -> float:
        if line1.xv is None and line2.xv is not None:
            angle = 90 - line1.theta
        elif line1.xv is not None and line2.xv is None:
            angle = 90 - line2.theta
        elif line1.slope * line2.slope == -1:
            angle = 90
        else:
            angle = np.rad2deg(np.arctan((line2.slope - line1.slope) / (1 + line1.slope * line2.slope)))

        return angle + 180


def compute_intersections(lines: list['Line'], image: ArrayLike) -> list[Intersection]:
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = lines[i].intersection(lines[j], image)
            if intersection is not None:
                intersections.append(intersection)
    return intersections


def transform_intersection(
    intersection: Intersection,
    source_img: np.ndarray,
    original_x_start: int,
    original_y_start: int,
    to_global: bool = True,
    ) -> Intersection:
    from .lines import transform_line
    
    transformed_point = transform_point(intersection.point, original_x_start, original_y_start, to_global=to_global)
    line1_t = transform_line(intersection.line1, source_img, original_x_start, original_y_start, to_global)
    line2_t = transform_line(intersection.line2, source_img, original_x_start, original_y_start, to_global)
    return Intersection(line1_t, line2_t, transformed_point)