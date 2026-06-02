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

    basic_params = params_court_detection.detection_params.basic
    surface = params_court_detection.match_params.surface
    baseline_params = params_court_detection.detection_params.baseline

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

                detector = CourtDetector(frame, **basic_params.model_dump(), surface=surface)
                result = detector.scan_for_baseline(**baseline_params.model_dump())

                if result is None:
                    continue
                else:
                    baseline, sidelines = result

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