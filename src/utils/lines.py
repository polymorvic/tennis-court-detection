import copy
from typing import Iterable, Literal, Self, TYPE_CHECKING, Union

import numpy as np

from .common import Hashable
from src.utils.points import Point, transform_point

if TYPE_CHECKING:
    from .intersections import Intersection

type numeric = int | float


class Line(Hashable):

    def __init__(self, slope: float | None = None, intercept: float | None = None, xv: float | None = None) -> None:
        self.slope = slope
        self.intercept = intercept
        self.xv = xv

    def _key_(self) -> tuple[numeric, numeric, numeric | None]:
        return (self.slope, self.intercept, self.xv)

    def __repr__(self) -> str:
        return f"y = {self.slope} * x + {self.intercept}"

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def intersection(self, another_line: Self, image: np.ndarray) -> Union['Intersection', None]:
        from .intersections import Intersection
        
        if (self.slope is not None and another_line.slope is not None and self.slope == another_line.slope) or (
            self.xv is not None and another_line.xv is not None
        ):
            return None

        elif self.xv is not None and another_line.xv is None:
            x = self.xv
            y = another_line.slope * x + another_line.intercept

        elif self.xv is None and another_line.xv is not None:
            x = another_line.xv
            y = self.slope * x + self.intercept

        else:
            x = (another_line.intercept - self.intercept) / (self.slope - another_line.slope)
            y = self.slope * x + self.intercept

        height, width = image.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            return Intersection(self, another_line, Point(int(x), int(y)))
        else:
            return None

    def y_for_x(self, x: int) -> int | None:
        if self.slope is None or self.intercept is None:
            return None
        return int(self.slope * x + self.intercept)

    def x_for_y(self, y: int) -> int | None:
        if self.xv is not None:
            return int(self.xv)
        if self.slope == 0 or self.slope is None:
            return None
        return int((y - self.intercept) / self.slope)

    def get_points_by_distance(self, main_point: Point, distance: float) -> tuple[Point, Point]:
        main_x, main_y = main_point

        if self.xv is not None:
            return Point(int(main_x), int(main_y - distance)), Point(int(main_x), int(main_y + distance))

        if self.slope is None or self.intercept is None:
            raise ValueError("Cannot compute points: line is not properly defined.")

        m = self.slope
        b = self.intercept

        A = 1 + m**2
        B = -2 * main_x + 2 * m * (b - main_y)
        C = main_x**2 + (b - main_y) ** 2 - distance**2

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            raise ValueError("No real solution: check if the distance is too large or the point is far from the line.")

        sqrt_delta = np.sqrt(discriminant)

        x1 = int((-B + sqrt_delta) / (2 * A))
        x2 = int((-B - sqrt_delta) / (2 * A))

        y1 = int(self.y_for_x(x1))
        y2 = int(self.y_for_x(x2))

        return Point(x1, y1), Point(x2, y2)

    def limit_to_img(self, img: np.ndarray) -> tuple[Point, Point]:
        img_width, img_height = img.shape[1] - 1, img.shape[0] - 1

        if self.xv is not None:
            x = int(self.xv)
            return Point(x, 0), Point(x, img_height)

        if self.slope == 0:
            y = int(self.intercept)
            return Point(0, y), Point(img_width, y)

        points = []

        x_top = self.x_for_y(0)
        if x_top is not None and 0 <= x_top <= img_width:
            points.append(Point(int(x_top), 0))

        x_bottom = self.x_for_y(img_height)
        if x_bottom is not None and 0 <= x_bottom <= img_width:
            points.append(Point(int(x_bottom), img_height))

        y_left = self.y_for_x(0)
        if y_left is not None and 0 <= y_left <= img_height:
            points.append(Point(0, int(y_left)))

        y_right = self.y_for_x(img_width)
        if y_right is not None and 0 <= y_right <= img_height:
            points.append(Point(img_width, int(y_right)))

        unique_points = list(dict.fromkeys(points))

        if len(unique_points) >= 2:
            return unique_points[0], unique_points[1]

        raise ValueError("Line does not intersect the image in at least two places.")

    def check_point_on_line(self, point: Point, tolerance: int = None) -> bool:
        y = self.y_for_x(point.x)
        x = self.x_for_y(point.y)

        if y is None or x is None:
            return False

        line_point = Point(x, y).as_int()

        if tolerance is None:
            return point.x == line_point.x and point.y == line_point.y

        return abs(line_point.y - point.y) < tolerance and abs(line_point.x - point.x) < tolerance

    @property
    def theta(self) -> float:
        if self.slope is None:
            return 90.0
        return np.degrees(np.arctan(self.slope))

    @classmethod
    def from_hough_line(cls, hough_line: tuple[int, int, int, int]) -> Self:
        x1, y1, x2, y2 = hough_line
        return cls.from_points((x1, y1), (x2, y2))

    @classmethod
    def from_points(cls, p1: tuple[int, int], p2: tuple[int, int]) -> Self:
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2:
            slope, intercept = None, None
            xv = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            xv = None

        return cls(slope, intercept, xv)


class LineGroup(Line):
    
    def __init__(self, lines: list[Line] = None) -> None:
        self.lines = lines or []

        if not self.lines:
            self.slope = self.intercept = self.xv = None
        else:
            self._calculate_line_approximation()

    def __repr__(self) -> str:
        if not self.lines:
            return "LineGroup(empty)"

        if self.xv is not None:
            return f"LineGroup: x = {self.xv:.2f} (from {len(self.lines)} lines)"
        else:
            return f"LineGroup: y = {self.slope:.2f} * x + {self.intercept:.2f} (from {len(self.lines)} lines)"

    def process_line(self, line: Line, thresh_theta: float | int, thresh_intercept: float | int) -> bool:
        ref = self.lines[0]
        found = False

        if abs(ref.theta - line.theta) < thresh_theta:
            if ref.xv is None and line.xv is None:
                if abs(ref.intercept - line.intercept) < thresh_intercept:
                    found = True

            if ref.xv is not None or line.xv is not None:
                found = True

            if found:
                self.lines.append(line)
                self._calculate_line_approximation()

        self.lines = sorted(self.lines, key=lambda line: -line.intercept)
        return found

    def get_line(self, line_type: Literal["min", "max"]) -> Line:
        return {"min": self.lines[0], "max": self.lines[-1]}[line_type]

    def _calculate_line_approximation(self) -> None:
        vertical_lines = [line.xv for line in self.lines if line.xv is not None]

        if vertical_lines:
            self.xv = np.median(vertical_lines)
            self.slope, self.intercept = None, None

        else:
            self.xv = None
            self.slope = np.median([line.slope for line in self.lines])
            self.intercept = np.median([line.intercept for line in self.lines])


def transform_line(
    original_line: Line, 
    original_img: np.ndarray, 
    original_x_start: int, 
    original_y_start: int, 
    to_global: bool = True
    ) -> Line:
    pts_source: Iterable[Point] = original_line.limit_to_img(original_img)
    pts_transformed = [transform_point(p, original_x_start, original_y_start, to_global=to_global) for p in pts_source]
    return Line.from_points(*pts_transformed)