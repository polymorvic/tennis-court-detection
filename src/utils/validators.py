from typing import Type

import numpy as np
from cvgeomkit.common import ArrayLike, NumpyImage, Numeric

from src.utils.errors import NotArrayError


def check_if_numpy_image(
    img: ArrayLike
) -> NumpyImage:
    
    if isinstance(img, NumpyImage):
        return img

    if isinstance(img, np.ndarray):
        return NumpyImage(img)

    raise NotArrayError()


def validate_number(
    value: Numeric,
    expected_type: Type[Numeric],
    min_value: Numeric,
    max_value: Numeric,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}")

    if min_value is not None:
        valid_min = value >= min_value if min_inclusive else value > min_value
        if not valid_min:
            op = ">=" if min_inclusive else ">"
            raise ValueError(f"Value must be {op} {min_value}")

    if max_value is not None:
        valid_max = value <= max_value if max_inclusive else value < max_value
        if not valid_max:
            op = "<=" if max_inclusive else "<"
            raise ValueError(f"Value must be {op} {max_value}")