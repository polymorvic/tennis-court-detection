from pathlib import Path

import cv2

from tennis_court_detection.court_detector import CourtDetector
import torch
from tennis_court_detection.utils.helpers import load_process_params
import torch


import tyro

from tennis_court_detection.training.preproc import transform
from tennis_court_detection.training.architectures import load_resnet50_model
from tennis_court_detection.training.device import get_device

def run(
    shot_classifier_params_path: Path,
    params_config_path: Path,
    video_path: Path,
):
    '''
    uv run python main.py --shot-classifier-params-path models/shot-classifier-best_28052026.pt --params-config-path config/process_params.config.json --video-path data/test_video.mov
    '''
    device = get_device()
    model = load_resnet50_model(shot_classifier_params_path)
    model.to(device)

    params_court_detection = load_process_params(params_config_path)

    crop_center_ratio = params_court_detection.detection_params.basic.crop_center_ratio
    roi_h_px = params_court_detection.detection_params.basic.roi_h_px
    step_px = params_court_detection.detection_params.basic.step_px

    warmup = params_court_detection.detection_params.baseline.warmup
    canny_lower_thresh = params_court_detection.detection_params.baseline.canny_lower_thresh
    canny_upper_thresh = params_court_detection.detection_params.baseline.canny_upper_thresh
    hough_thresh = params_court_detection.detection_params.baseline.hough_thresh
    min_line_len_ratio = params_court_detection.detection_params.baseline.min_line_len_ratio
    min_line_len_ensure_ratio = params_court_detection.detection_params.baseline.min_line_len_ensure_ratio
    min_line_gap_px = params_court_detection.detection_params.baseline.min_line_gap_px
    h_line_slope_tolerance = params_court_detection.detection_params.baseline.h_line_slope_tolerance
    h_delta_ensure_px = params_court_detection.detection_params.baseline.h_delta_ensure_px

    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        i += 1
        ret, frame = cap.read()


        if not ret:
            print("the end")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            img_tensor = transform(frame).to(device)
            img_tensor = img_tensor.unsqueeze(0)
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()

            if prediction > 0.5:
                tekst = "OK"

                detector = CourtDetector(frame, crop_center_ratio, roi_h_px, step_px)

                baseline, sidelines  = detector.scan_for_baseline(
                    warmup,
                    canny_lower_thresh,
                    canny_upper_thresh,
                    hough_thresh,
                    min_line_len_ratio,
                    min_line_len_ensure_ratio,
                    min_line_gap_px,
                    h_line_slope_tolerance
                )

                if not baseline:
                    continue

                p1, p2 = baseline.limit_to_img(frame)
                cv2.line(frame, p1, p2, (0, 0, 255), 4)


            else:
                tekst = "NIE OK"

                # cv2.imwrite(f"bad_frames/bad_frame_{i:04}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cv2.putText(frame, f'proba: {prediction:.2f}, {tekst}, linia {baseline}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("video frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(run)