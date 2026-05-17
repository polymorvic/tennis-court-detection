import tyro
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.annotations import TennisCourtAnnotationCollection
from src.utils.testing import (build_output_dir, gt_hline_from_annotation, 
                               save_test_histogram, prepare_test_results_report, draw_results_on_img)
from src.utils.helpers import load_process_params, load_pics_blacklist
from src.schemas.testing import TestType
from src.court_detector import CourtDetector
from src.utils.annotations import transform_keypoint_annotation

from cvgeomkit.utils.helpers import read_image_as_numpyimage
from cvgeomkit.geometry.points import Point

def run(
    test_type: TestType,
    pics_path: Path | str = 'data/pics',
    params_path: Path | str = 'config/process_params.config.json',
    blacklist_path: Path | str = 'config/pics_blacklist.config.yaml',
    annotation_path: Path | str = 'data/annotations.json',
    output_dir: Path | str = 'results/lines'
):
    '''
    uv run python -m tests.lines --test-type SERVICE_LINE
    '''
    proj_cwd = Path.cwd()
    pics_path = proj_cwd / pics_path
    test_out_dir = build_output_dir(proj_cwd / output_dir, test_type)

    tcac = TennisCourtAnnotationCollection.from_clean_file(annotation_path)
    params = load_process_params(params_path)
    blacklist = load_pics_blacklist(blacklist_path).blacklist

    service_side = params.match_params.service_side.value
    roi_h = params.detection_params.traversing.roi_height_px
    step = params.detection_params.traversing.step_px
    warmup = params.detection_params.traversing.warmup
    canny_lower_thresh = params.detection_params.lines_detection.canny_lower_thresh
    canny_upper_thresh = params.detection_params.lines_detection.canny_upper_thresh
    hough_thresh = params.detection_params.lines_detection.hough_thresh
    min_line_len_ratio = params.detection_params.lines_detection.min_line_len_ratio
    min_line_gap_px = params.detection_params.lines_detection.min_line_gap_px
    vertical_center_delta_px = params.detection_params.lines_detection.vertical_center_delta_px
    white_line_bin_lower_thresh = params.detection_params.lines_detection.white_line_bin_lower_thresh
    white_line_bin_upper_thresh = params.detection_params.lines_detection.white_line_bin_upper_thresh



    results = []
    not_found = []
    no_annotation = []
    for file in tqdm(sorted(pics_path.glob("*png"))):

        if file.name in blacklist:
            continue

        ann = tcac.filter_by_image(file.name)
        if ann is None:
            print(f'Brak annotacji dla zdjęcia: {file.stem}')
            no_annotation.append(file.name)
            continue

        img = read_image_as_numpyimage(file)

        detector = CourtDetector(img)
        centre_intresection = detector.scan_for_service_lines(
            service_side,
            roi_h,
            step,
            warmup,
            canny_lower_thresh,
            canny_upper_thresh,
            hough_thresh,
            min_line_len_ratio,
            min_line_gap_px,
            vertical_center_delta_px,
            white_line_bin_lower_thresh,
            white_line_bin_upper_thresh
        )

        if centre_intresection is None:
            not_found.append(file.name)
            continue

        pred_line, *_ = centre_intresection

        left_service_point = Point.from_iterable(transform_keypoint_annotation(img, tcac.get_key_point_for_image(file.name, "left_service_point").coordinates.as_array))
        right_service_point = Point.from_iterable(transform_keypoint_annotation(img, tcac.get_key_point_for_image(file.name, "right_service_point").coordinates.as_array))

        gt_line = gt_hline_from_annotation(left_service_point, right_service_point)

        draw_results_on_img(test_out_dir, img, pred_line, gt_line, f'{file.stem}-result.png')

        results.append({
            'pic_name': file.name,
            'pred_line': pred_line,
            'gt_line': gt_line,
            'pred_intercept': pred_line.intercept,
            'gt_intercept': gt_line.intercept,
        })

    intercept_errors = prepare_test_results_report(
        test_out_dir,
        results,
        not_found,
        no_annotation,
        f'_{test_type.value}-report'

    )

    save_test_histogram(
        test_out_dir, 
        intercept_errors, 
        f'_{test_type.value}-hist', 
        test_type.value
    )

    



if __name__ == '__main__':
    tyro.cli(run)
