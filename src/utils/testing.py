from datetime import datetime
from pathlib import Path

import numpy as np

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