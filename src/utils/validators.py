import numpy as np
from cvgeomkit.common import ArrayLike, NumpyImage

from src.utils.errors import NotArrayError


def check_if_numpy_image(img: ArrayLike) -> NumpyImage:

    if isinstance(img, NumpyImage):
        return img

    if isinstance(img, np.ndarray):
        return NumpyImage(img)

    raise NotArrayError()