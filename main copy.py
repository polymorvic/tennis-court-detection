import cv2
from torch import nn
from src.court_detector import CourtDetector
import torch
from torchvision import models
from torch import nn
from src.utils.helpers import load_process_params

import torch
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage, ToTensor

from torch import nn

class MinMaxTransform(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.float().div(255.0)
    

def compose_transform(*ops: nn.Module) -> Compose:

    return Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        MinMaxTransform(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        *ops
        ])
    

transform = compose_transform()


params = torch.load("models/shot-classifier-best.pt")["model"]
model = models.resnet50()

model.fc.in_features # liczba cech do glowy

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
)

model.load_state_dict(params)
model.eval()

params_court_detection = load_process_params('config/process_params.config.json')

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

cap = cv2.VideoCapture("test_video.mov")
i = 0
has_moved_to_front_camera = False
while True:
    i += 1
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not ret:
        print("the end")
        break

    with torch.no_grad():
        img_tensor = transform(frame)
        img_tensor = img_tensor.unsqueeze(0)
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            tekst = "OK"

            if not has_moved_to_front_camera:
                has_moved_to_front_camera = True


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
            has_moved_to_front_camera = False
            tekst = "NIE OK"

            # cv2.imwrite(f"bad_frames/bad_frame_{i:04}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cv2.putText(frame, f'proba: {prediction:.2f}, {tekst}, linia {baseline}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    

    cv2.imshow("video frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()