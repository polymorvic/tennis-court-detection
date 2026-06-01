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