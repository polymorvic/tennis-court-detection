import cv2
from cvgeomkit.common import ArrayLike
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage, ToTensor
from torch import nn

from tennis_court_detection.utils.validators import check_if_numpy_image


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


def preprocess_image_for_surface_classification(
    image: ArrayLike,
    margin_top_ratio: float = 0.4,
    margin_bottom_ratio: float = 0.2,
    margin_left_right_ratio: float = 0.3,
    kernel_size_ratio: float = 0.03,
) -> ArrayLike:
    image = check_if_numpy_image(image)
    image_height, image_width = image.height, image.width

    margin_top_px = int(image_height * margin_top_ratio)
    margin_bottom_px = int(image_height * margin_bottom_ratio)
    margin_left_right_px = int(image_width * margin_left_right_ratio)

    roi = image[
        margin_top_px:image_height - margin_bottom_px,
        margin_left_right_px:image_width - margin_left_right_px,
    ]

    kernel_size = max(3, int(min(roi.shape[:2]) * kernel_size_ratio) | 1)

    return cv2.medianBlur(roi, kernel_size)