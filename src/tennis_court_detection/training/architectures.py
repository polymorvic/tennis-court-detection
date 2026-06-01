from torchvision import models
import torch.nn as nn
import torch


def build_resnet50_model(
    weights: models.ResNet50_Weights | None = models.ResNet50_Weights.DEFAULT,
    outputs_num: int = 1
) -> models.ResNet:
    model = models.resnet50(weights=weights)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, outputs_num),
    )
    return model


def load_resnet50_model(params_path: str) -> models.ResNet:
    model = build_resnet50_model(weights=None)
    params = torch.load(params_path)["model"]
    model.load_state_dict(params)
    model.eval()
    return model