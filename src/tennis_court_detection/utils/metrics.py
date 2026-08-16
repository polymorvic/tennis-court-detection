import numpy as np
from cvgeomkit.common import ArrayLike


def calculate_white_pixels_ratio(
    edges: ArrayLike,
    mask: ArrayLike
) -> float:
    """Calculate what percentage of all pixels within the mask are white.

    Args:
        edges (ArrayLike): Binary image of edges.
        mask (ArrayLike): Binary mask defining the region of interest.

    Returns:
        float: Ratio of white pixels within the mask.
    """
    mask_pixels = np.count_nonzero(mask)

    if mask_pixels == 0:
        return 0.0

    white_pixels = np.count_nonzero((edges > 0) & (mask > 0))

    return white_pixels / mask_pixels


def calculate_white_columns_ratio(
    edges: ArrayLike,
    mask: ArrayLike
) -> float:
    """Calculate what percentage of all columns within the mask include at least one white pixel.

    Args:
        edges (ArrayLike): Binary image of edges.
        mask (ArrayLike): Binary mask defining the region of interest.

    Returns:
        float: Ratio of columns containing at least one white pixel within the mask.
    """
    mask_bool = mask > 0

    mask_columns = np.any(mask_bool, axis=0)
    white_columns = np.any((edges > 0) & mask_bool, axis=0)

    mask_columns_count = np.count_nonzero(mask_columns)

    if mask_columns_count == 0:
        return 0.0

    return np.count_nonzero(white_columns) / mask_columns_count