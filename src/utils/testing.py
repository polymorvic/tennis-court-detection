from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.schemas.testing import TestType

from cvgeomkit.geometry.lines import Line 
from cvgeomkit.geometry.points import Point


def build_output_dir(parent_dir: str | Path, test_type: TestType) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(parent_dir) / test_type.value / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def gt_hline_from_annotation(p1: Point, p2: Point,) -> Line:
    approx_intercept = np.median(np.array([p1.y, p2.y])).astype(np.int64)
    return Line(slope=0, intercept=approx_intercept)


def save_test_histogram(
    output_path: Path | str,
    data: np.ndarray,
    filename: str,
    title: str
) -> None:
    is_int_like = np.allclose(data, np.round(data))
    bins = np.arange(int(data.min()), int(data.max()) + 2, 10) if is_int_like else "auto"
    output_path = Path(output_path) / f"{filename}.png"

    plt.figure()
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xlabel('Error')
    plt.ylabel("Frequency")
    plt.title(f'Histogram of {title}')
    plt.savefig(output_path)
    plt.close()
