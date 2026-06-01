from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
from tennis_court_detection.training.preproc import transform
import pandas as pd
from typing import Literal


class ShotDataset(Dataset):
    def __init__(self, split_spec_path: str | Path, pics_root: str | Path, mode: Literal['train', 'val', 'test']):
        self.pics_root = Path(pics_root)

        data = pd.read_csv(split_spec_path)
        self.data = data[data["dataset"] == mode].sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def _img_path(self, img_name: str) -> Path:
        subdir = "skip" if "skip_" in img_name else "."
        if not img_name.endswith('.png'):
            img_name += '.png'
        return self.pics_root / subdir / img_name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = self._img_path(row["img_name"])
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img_tensor, label