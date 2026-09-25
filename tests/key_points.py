import tyro
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tennis_court_detection.schemas.court import CourtSegmentsCollection
from tennis_court_detection.utils.annotations import TennisCourtAnnotationCollection
from tennis_court_detection.utils.testing import (
    build_output_dir, 
    calculate_error, 
    compose_reports,
    save_reports,
    put_points_on_image,
    put_legend_on_image
)
from tennis_court_detection.utils.helpers import (
    load_process_params, 
    compute_intersections_for_line
)
from tennis_court_detection.schemas.testing import TestType
from tennis_court_detection.court_detector import CourtDetector
from tennis_court_detection.utils.testing import get_surface_from_filename
from cvgeomkit.utils.helpers import read_image_as_numpyimage, load_yaml


def run(
    test_type: TestType = TestType.KEY_POINTS,
    pics_path: Path | str = 'data/pics',
    params_path: Path | str = 'config/process_params.config.json',
    edge_cases_path: Path | str = 'config/edge_case_images.yaml',
    annotation_path: Path | str = 'data/annotations.json',
    output_dir: Path | str = 'results'
):
    '''
    uv run python -m tests.key_points --test-type KEY_POINTS
    '''
    proj_cwd = Path.cwd()
    pics_path = proj_cwd / pics_path
    test_out_dir = build_output_dir(proj_cwd / output_dir, test_type)
    test_out_dir_pic = test_out_dir / 'pics'

    test_out_dir_pic.mkdir(exist_ok=True)
    not_found_dir = test_out_dir_pic / 'not_found'

    not_found_dir.mkdir(exist_ok=True)

    tcac = TennisCourtAnnotationCollection.from_clean_file(annotation_path)
    params = load_process_params(params_path)
    edge_cases = load_yaml(edge_cases_path)['edge_cases']

    basic_params = params.detection_params.basic
    baseline_params = params.detection_params.baseline

    results = []
    not_found = []
    no_annotation = []
    found_count = 0
    detail_report_rows = []
    summary_report_rows = []
    for file in tqdm(sorted(pics_path.glob("*png"))):

        if file.stem in edge_cases:
            continue

        ann = tcac.filter_by_image(file.name)

        if not ann:
            no_annotation.append(file.name)
            continue

        ground_truth_points = ann.prepare_for_execution()

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
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue
        
        baseline_segments, left_outer_segments, left_inner_segments,right_inner_segments, right_outer_segments = segments

        try:
            paired_horizontal_half_lines = detector.scan_for_horizontal_lines(
                **baseline_params.model_dump(), 
                left_segments=left_inner_segments, 
                right_segments=right_inner_segments)
        except Exception:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue
        
        if not paired_horizontal_half_lines:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        try:
            result = detector.find_service_line(paired_horizontal_half_lines[0])
        except Exception:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if result is None:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        service_line_segments, inters = result

        try:
            centre_service_half_lines = detector.find_centre_service_half_lines(inters[0].point)
        except Exception:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not centre_service_half_lines:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
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
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not netline_bottom_segments:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        try:
            left_centre_service_line_segments, right_centre_service_line_segments = detector.centre_service_half_lines_to_segments(
                centre_service_half_lines,
                netline_bottom_segments
            )
        except Exception:
            not_found.append(file.name)
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
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        if not netline_top_segments:
            not_found.append(file.name)
            cv2.imwrite(str(not_found_dir / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            continue

        court_segments = CourtSegmentsCollection(
            baseline=baseline_segments,
            left_outer=left_outer_segments,
            left_inner=left_inner_segments,
            right_inner=right_inner_segments,
            right_outer=right_outer_segments,
            service_line=service_line_segments,
            left_centre_service_line=left_centre_service_line_segments,
            right_centre_service_line=right_centre_service_line_segments,
            netline_bottom=netline_bottom_segments
        )

        projected_points = detector.find_opposite_side_points(court_segments)

        baseline_opposite_segments = detector.find_opposite_baseline(
            **projected_points
        )

        service_line_opposite_segments = detector.find_opposite_service_line(
            **projected_points
        )

        service_lines_opposite_results = detector.find_centre_service_lines_opposite(
            netline_bottom_segments, 
            left_centre_service_line_segments,
            right_centre_service_line_segments,
            service_line_opposite_segments,
            left_inner_segments,
            right_inner_segments,
            centre_service_half_lines
        )

        if service_lines_opposite_results is None:
            left_centre_service_line_segments_opposite, right_centre_service_line_segments_opposite = (
                detector.find_centre_service_lines_opposite_fallback(
                    left_centre_service_line_segments, 
                    right_centre_service_line_segments, 
                    service_line_opposite_segments
                )
            )
        else:
            left_centre_service_line_segments_opposite, right_centre_service_line_segments_opposite = service_lines_opposite_results

        img_copy = img.copy()
        for segments in [baseline_segments, left_outer_segments, 
                        left_inner_segments, right_inner_segments, right_outer_segments,
                        service_line_segments, left_centre_service_line_segments, 
                        right_centre_service_line_segments, netline_bottom_segments, netline_top_segments, 
                        baseline_opposite_segments, service_line_opposite_segments,
                        left_centre_service_line_segments_opposite, right_centre_service_line_segments_opposite]:
            for segment in segments:
                cv2.line(img_copy, segment.start, segment.end, (255, 0, 0), 1)

        pred_points = detector.get_all_intersections(
            baseline_segments, 
            left_outer_segments, 
            left_inner_segments, 
            right_inner_segments, 
            right_outer_segments,
            service_line_segments, 
            left_centre_service_line_segments, 
            right_centre_service_line_segments, 
            netline_bottom_segments, 
            netline_top_segments, 
            baseline_opposite_segments, 
            service_line_opposite_segments,
            left_centre_service_line_segments_opposite, 
            right_centre_service_line_segments_opposite
        )

        errors_per_point, summary_errors, stats_errors = calculate_error(ground_truth_points, pred_points)

        img_copy = put_points_on_image(img_copy, ground_truth_points, pred_points)
        img_copy = put_legend_on_image(
            img_copy,
            mean_error=summary_errors['all_mean_error'],
            min_error=stats_errors['min_error'],
            max_error=stats_errors['max_error'],
            std_error=stats_errors['std_error']
        )

        cv2.imwrite(str(test_out_dir_pic / file.name), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        found_count += 1

        errors_per_point['image_name'] = file.name
        detail_report_rows.append(errors_per_point)

        combined_errors = {**summary_errors, **stats_errors}
        combined_errors['image_name'] = file.name
        summary_report_rows.append(combined_errors)

    report_df_detailed, report_df_summary, stats_df = compose_reports(detail_report_rows, summary_report_rows)
    save_reports(report_df_detailed, report_df_summary, stats_df, str(test_out_dir / "results.xlsx"))
    
    print(f"Found count: {found_count}")
    print(f"Not found count: {len(not_found)}, {len(not_found) / (found_count + len(not_found))}")


if __name__ == '__main__':
    tyro.cli(run)
