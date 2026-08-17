from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tennis_court_detection.schemas.testing import TestType
from tennis_court_detection.utils.validators import check_if_numpy_image
from tennis_court_detection.schemas.config import Surface

from cvgeomkit.common import ArrayLike
from cvgeomkit.geometry.lines import Line 
from cvgeomkit.geometry.points import Point


def get_surface_from_filename(filename: str) -> Surface:
    filename = filename.lower()

    if filename.startswith('03') or filename.startswith('09') or filename.startswith('10'):
        return Surface.CLAY
    
    elif filename.startswith('07') or filename.startswith('08'):
        return Surface.GRASS 
    
    return Surface.HARD



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
    plt.xlabel('Error px')
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title}'s error")
    plt.savefig(output_path)
    plt.close()


def prepare_test_results_report(
    output_path: str | Path,
    results_data: list[dict],
    not_found_list: list[str],
    no_annotation_list: list[str],
    filename: str,
    results_sheet_name: str = 'results', 
    stats_sheet_name: str = 'stats',
    not_found_sheet_name: str = 'not_found',
    no_annotation_sheet_name: str = 'no_annotation',
) -> pd.Series | None:
    
    if not results_data:
        raise ValueError("results_data is empty!")

    with pd.ExcelWriter(Path(output_path) / f'{filename}.xlsx', engine='openpyxl') as writer:

        results_df = pd.DataFrame(results_data)
        error = results_df['gt_intercept'] - results_df['pred_intercept']
        results_df['error'] = error
        results_df['abs_error'] = error.abs()
        results_df.to_excel(writer, sheet_name=results_sheet_name, index=False)

        stats_df = (
            results_df['error']
            .agg(['median', 'mean', 'std', 'min', 'max'])
            .to_frame()
            .T
        )
        stats_df.to_excel(writer, sheet_name=stats_sheet_name, index=False)

        not_found_df = pd.DataFrame(not_found_list, columns = ['pic_name'])
        not_found_df.to_excel(writer, sheet_name=not_found_sheet_name, index=False)

        no_annotation_df = pd.DataFrame(no_annotation_list, columns = ['pic_name'])
        no_annotation_df.to_excel(writer, sheet_name=no_annotation_sheet_name, index=False)

    return results_df['error']


def draw_results_on_img(
    output_path: str | Path,
    img: ArrayLike,
    pred_line: Line,
    gt_line: Line,
    filename: str
) -> None:
    img = check_if_numpy_image(img)
    img_copy = img.copy()
    p1_pred, p2_pred = pred_line.limit_to_img(img_copy)
    p1_gt, p2_gt = gt_line.limit_to_img(img_copy)

    pred_color = (255, 0, 0)   
    gt_color = (0, 255, 0) 

    cv2.line(img_copy, p1_pred, p2_pred, pred_color, 2)
    cv2.line(img_copy, p1_gt, p2_gt, gt_color, 2)

    w = img_copy.width

    legend_x = w - 220
    legend_y = 20

    cv2.rectangle(
        img_copy,
        (legend_x - 10, legend_y - 10),
        (w - 10, legend_y + 70),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        img_copy,
        (legend_x - 10, legend_y - 10),
        (w - 10, legend_y + 70),
        (0, 0, 0),
        1
    )

    cv2.line(
        img_copy,
        (legend_x, legend_y + 15),
        (legend_x + 40, legend_y + 15),
        pred_color,
        3
    )

    cv2.putText(
        img_copy,
        "Prediction",
        (legend_x + 50, legend_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        img_copy,
        (legend_x, legend_y + 45),
        (legend_x + 40, legend_y + 45),
        gt_color,
        3
    )

    cv2.putText(
        img_copy,
        "Ground Truth",
        (legend_x + 50, legend_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(str(Path(output_path) / filename), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))


Box = tuple[int, int, int, int]

def put_text_boxes_overlap(
    box1: Box, 
    box2: Box, 
    margin: int = 3
) -> bool:
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    return not (
        x2 + margin < a1
        or a2 + margin < x1
        or y2 + margin < b1
        or b2 + margin < y1
    )


StringCollection: TypeAlias = list[str] | tuple[str] | set[str]

def group_pics_by_match(
    pic_names: list[str],
    match_numbers: StringCollection | None = None
) -> dict[str, list[str]]:
    groups = {
        pic_name.split('_')[0]
        for pic_name in pic_names
    }

    if match_numbers is not None:
        groups &= match_numbers

    return {
        match_number: sorted(
            pic_name
            for pic_name in pic_names
            if pic_name.split('_')[0] == match_number
        )
        for match_number in sorted(groups)
    }

