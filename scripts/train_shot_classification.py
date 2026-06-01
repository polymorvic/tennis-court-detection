import torch
import numpy as np
import random
from pathlib import Path

from torch import nn
from tennis_court_detection.training.architectures import build_resnet50_model
from tennis_court_detection.training.dataset import ShotDataset
from torch.utils.data import DataLoader
from tennis_court_detection.training.metrics import History
import tyro
from typing import Literal
import torch.nn.functional as F
from tqdm import tqdm


def run(
        dataset_split_spec_path: Path,
        imgs_root: Path,
        batch_size: int = 32,
        epochs_num: int = 10,
        trainable_layers: tuple[str] = ('fc',),
        metrics_focus: Literal['loss', 'accuracy'] = 'loss',
        models_output_dir: Path = Path('models'),
        test_threshold: float = 0.5,
        start_lr: float = 0.0001,
        lr_scheduler_patience: int = 4
):
    '''
    uv run python scripts/train_shot_classification.py --dataset-split-spec-path data/shot_classification_split_data.csv --imgs-root data/pics
    '''
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    train_dataset = ShotDataset(dataset_split_spec_path, imgs_root, 'train')
    val_dataset = ShotDataset(dataset_split_spec_path, imgs_root, 'val')
    test_dataset = ShotDataset(dataset_split_spec_path, imgs_root, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = build_resnet50_model()

    for name, param in model.named_parameters():
        if not name.startswith(trainable_layers): 
            param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_scheduler_patience)
    epochs_num = epochs_num

    model = model.to(device)

    train_hist = History()
    val_hist = History()
    learning_rates = []

    for ep in range(epochs_num):
        model.train()

        train_hist.on_epoch_start()

        for batch_x, batch_y in tqdm(train_dataloader, desc="Training Batches"):
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  
            y_hat = model(batch_x)

            loss = criterion(y_hat, batch_y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            train_hist.on_step_end(loss.item(), y_hat, batch_y)

        train_hist.on_epoch_end(train_dataset, metrics_focus)
        model.eval()
        val_hist.on_epoch_start()
        with torch.no_grad():

            for batch_x, batch_y in tqdm(val_dataloader, desc="Validation Batches"):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_hat = model(batch_x)

                loss = criterion(y_hat, batch_y.reshape(-1, 1))
                val_hist.on_step_end(loss.item(), y_hat, batch_y)


        save_best_model = val_hist.on_epoch_end(val_dataset, metrics_focus)
        scheduler.step(val_hist.epoch_losses[-1])
        learning_rates.append(optimizer.param_groups[0]['lr'])
        print(f"Epoka {ep+1}/{epochs_num}: {train_hist.get_latest(mode='train')}, {val_hist.get_latest('val')}")

        checkpoint_data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep + 1}
        torch.save(checkpoint_data, models_output_dir / 'shot-classifier-last.pt')
        if save_best_model:
            torch.save(checkpoint_data, models_output_dir / 'shot-classifier-best.pt')


    model.eval()

    with torch.no_grad():
        test_loss = .0
        correct_pred = .0

        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_hat = model(batch_x)

            loss = criterion(y_hat, batch_y.reshape(-1, 1))
            test_loss += loss.item()

            y_pred = (F.sigmoid(y_hat).reshape(-1) >= test_threshold).long()
            correct_pred += (y_pred == batch_y).sum().item()

        test_loss /= len(test_dataloader.dataset)
        test_accuracy = correct_pred / len(test_dataloader.dataset)

    print(f"Test loss= {test_loss:.4f} Test accuracy= {test_accuracy:.4f}")


if __name__ == "__main__":
    tyro.cli(run)
