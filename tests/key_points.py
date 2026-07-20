import tyro
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tennis_court_detection.utils.annotations import TennisCourtAnnotationCollection
from tennis_court_detection.utils.testing import (build_output_dir,)
from tennis_court_detection.utils.helpers import (
    load_process_params, 
    load_pics_blacklist, 
    compute_intersections_for_line
)
from tennis_court_detection.schemas.testing import TestType
from tennis_court_detection.court_detector import CourtDetector
from tennis_court_detection.utils.annotations import transform_keypoint_annotation
from tennis_court_detection.utils.testing import get_surface_from_filename
from cvgeomkit.utils.helpers import read_image_as_numpyimage


def run(
    test_type: TestType,
    pics_path: Path | str = 'data/pics',
    params_path: Path | str = 'config/process_params.config.json',
    blacklist_path: Path | str = 'config/pics_blacklist.config.yaml',
    annotation_path: Path | str = 'data/annotations.json',
    output_dir: Path | str = 'results'
):
    '''
    uv run python -m tests.key_points --test-type KEY_POINTS
    '''
    proj_cwd = Path.cwd()
    pics_path = proj_cwd / pics_path
    test_out_dir = build_output_dir(proj_cwd / output_dir, test_type)

    tcac = TennisCourtAnnotationCollection.from_clean_file(annotation_path)
    params = load_process_params(params_path)
    blacklist = load_pics_blacklist(blacklist_path).blacklist

    basic_params = params.detection_params.basic
    baseline_params = params.detection_params.baseline

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

        surface = get_surface_from_filename(file.name)
        detector = CourtDetector(img, **basic_params.model_dump(), surface=surface)
        result = detector.scan_for_baseline(**baseline_params.model_dump())

        if result is None:
            not_found.append(file.name)
            continue
        else:
            baseline, sidelines = result

        intersections = compute_intersections_for_line(baseline, sidelines, img, exclude_similar_slope=True)
        
        try:
            segments = detector.find_sidelines_segments(intersections)

        except Exception:
            continue
        
        baseline_segments, left_outer_segments, left_inner_segments,right_inner_segments, right_outer_segments = segments


        court_image = np.zeros_like(img, dtype=np.uint8)

        for segments in [baseline_segments, left_outer_segments, 
                        left_inner_segments, right_inner_segments, right_outer_segments]:
            for segment in segments:
                cv2.line(court_image, segment.start, segment.end, (255, 0, 0), 5)

        img_copy = img.copy()
        mask_rgb = np.where(court_image, court_image, img_copy)

        mask_save = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(test_out_dir / file.name), mask_save)

        




    #     results.append({
    #         'pic_name': file.name,
    #         'pred_line': pred_line,
    #         'gt_line': gt_line,
    #         'pred_intercept': pred_line.intercept,
    #         'gt_intercept': gt_line.intercept,
    #     })

    # intercept_errors = prepare_test_results_report(
    #     test_out_dir,
    #     results,
    #     not_found,
    #     no_annotation,
    #     f'_{test_type.value}-report'

    # )

    # save_test_histogram(
    #     test_out_dir, 
    #     intercept_errors, 
    #     f'_{test_type.value}-hist', 
    #     test_type.value
    # )

    



if __name__ == '__main__':
    tyro.cli(run)
