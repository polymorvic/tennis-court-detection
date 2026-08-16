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
    not_found_dir = test_out_dir / 'not_found'

    not_found_dir.mkdir(exist_ok=True)

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
        # if ann is None:
        #     print(f'Brak annotacji dla zdjęcia: {file.stem}')
        #     no_annotation.append(file.name)
        #     continue

        img = read_image_as_numpyimage(file)
        img_copy = img.copy()

        surface = get_surface_from_filename(file.name)
        detector = CourtDetector(img, **basic_params.model_dump(), surface=surface)
        result = detector.scan_for_baseline(**baseline_params.model_dump())

        if result is None:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue
        else:
            baseline, sidelines = result

        intersections = compute_intersections_for_line(baseline, sidelines, img, exclude_similar_slope=True)
        
        try:
            segments = detector.find_sidelines_segments(intersections)

        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue
        
        baseline_segments, left_outer_segments, left_inner_segments,right_inner_segments, right_outer_segments = segments

        try:
            paired_horizontal_half_lines = detector.scan_for_horizontal_lines(
                **baseline_params.model_dump(), 
                left_segments=left_inner_segments, 
                right_segments=right_inner_segments)
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue
        
        if not paired_horizontal_half_lines:
            continue

        try:
            result = detector.find_service_line(paired_horizontal_half_lines[0])
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if result is None:
            continue

        service_line_segments, inters = result

        try:
            centre_service_half_lines = detector.find_centre_service_half_lines(inters[0].point)
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not centre_service_half_lines:
            continue

        try:
            netline_bottom_segments = detector.find_bottom_netline(
                baseline_segments, 
                left_outer_segments,
                left_inner_segments,
                right_inner_segments,
                right_outer_segments,
                service_line_segments
            )
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not netline_bottom_segments:
            continue

        try:
            left_centre_service_line_segments, right_centre_service_line_segments = detector.centre_service_half_lines_to_segments(
                centre_service_half_lines,
                netline_bottom_segments
            )
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        try:
            netline_top_segments = detector.find_top_netline(
                netline_bottom_segments,
                left_outer_segments,
                right_outer_segments,
                paired_horizontal_half_lines,
                centre_service_half_lines
            )
        except Exception:
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not netline_top_segments:
            continue

        img_copy = img.copy()
        for segments in [baseline_segments, left_outer_segments, 
                        left_inner_segments, right_inner_segments, right_outer_segments,
                        service_line_segments, left_centre_service_line_segments, 
                        right_centre_service_line_segments, netline_bottom_segments, netline_top_segments]:
            for segment in segments:
                cv2.line(img_copy, segment.start, segment.end, (255, 0, 0), 1)

        cv2.imwrite(str(test_out_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    tyro.cli(run)
