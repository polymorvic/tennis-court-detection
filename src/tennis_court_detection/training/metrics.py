from pydantic import BaseModel
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Literal


class History(BaseModel):
    epoch_losses: list[float] = []
    epoch_accuracy: list[float] = []
    step_losses: list[float] = []
    best_loss: float = np.inf
    best_accuracy: float = 0
    _running_loss: float
    _correct_pred: float

    def on_epoch_start(self) -> None:
        self._running_loss = .0
        self._correct_pred = .0

    def on_epoch_end(self, dataset: Dataset, focus: Literal['loss', 'accuracy']) -> bool:
        """
        Czy dana epoka polepszyla model wzgledem treningu na podstawie metryki przekazywanej w argumencie focus
        """
        current_loss = self._running_loss / len(dataset)
        current_accuracy = self._correct_pred / len(dataset)
        self.epoch_losses.append(current_loss)
        self.epoch_accuracy.append(current_accuracy)

        if is_better_loss := current_loss < self.best_loss:
            self.best_loss = current_loss
  
        if is_better_accuracy := current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy

        return  {'loss': is_better_loss, 'accuracy': is_better_accuracy}[focus]

        
    def on_step_end(self, loss: float, y_hat: torch.Tensor, y_gt: torch.Tensor) -> None:
        y_pred = (F.sigmoid(y_hat).reshape(-1) >= 0.5).long()
        self._correct_pred += (y_pred == y_gt).sum().item()
        self._running_loss += loss
        self.step_losses.append(loss)

        

    def get_latest(self, mode: Literal['train', 'val']) -> str:
        return f"{mode} loss= {self.epoch_losses[-1]:.4f} {mode} accuracy= {self.epoch_accuracy[-1]:.4f}"