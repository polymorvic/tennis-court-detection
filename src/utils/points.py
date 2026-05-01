from typing import Iterator, TYPE_CHECKING, Self, Union

import numpy as np

from .common import Hashable

if TYPE_CHECKING:
    from .intersections import Intersection


class Point[T: (int, float)](Hashable):

    __slots__ = ("_x", "_y")

    def __init__(self, x: T, y: T) -> None:
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_y", y)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"'Point' object attribute '{name}' is read-only")

    def _key_(self) -> tuple[T, T]:
        return (self._x, self._y)

    @classmethod
    def from_xy(cls, x: T, y: T) -> Self:
        return cls(x, y)

    @classmethod
    def from_iterable(cls, iterable: tuple[T, T] | list[T, T]) -> Self:
        values = tuple(iterable)
        if len(values) != 2:
            raise ValueError(f"Expected iterable of length 2, got {len(values)}")
        return cls(values[0], values[1])

    @property
    def x(self) -> T:
        return self._x

    @property
    def y(self) -> T:
        return self._y

    def distance(self, another_point: Self) -> float:
        return np.linalg.norm(np.array([self.x, self.y]) - np.array([another_point.x, another_point.y]))

    def is_in_area(self, p1: Self, p2: Self) -> bool:
        return p1.x < self.x < p2.x and p1.y < self.y < p2.y

    def __getitem__(self, index: int) -> T:
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        else:
            raise IndexError("Point index out of range")

    def __iter__(self) -> Iterator[float]:
        yield self._x
        yield self._y


    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"Point({self._x}, {self._y})"

    def __str__(self) -> str:
        return f"Point({self._x}, {self._y})"

    @property
    def as_int(self) -> Self:
        return Point(int(self.x), int(self.y))

    @property
    def as_tuple(self) -> tuple[T, T]:
        return (self._x, self._y)
    
    
def transform_point(
    point: Union['Intersection', Point], 
    original_x_start: int, 
    original_y_start: int, 
    to_global: bool = True
    ) -> Point:
    from .intersections import Intersection
    
    if isinstance(point, Intersection):
        point = point.point

    if to_global:
        return Point(point.x + original_x_start, point.y + original_y_start)
    else:
        return Point(point.x - original_x_start, point.y - original_y_start)