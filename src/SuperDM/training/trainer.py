from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SuperDM.utility.io import ensure_dir
from SuperDM.utility.plotting import plot_loss_curve


class Trainer:
    """Training loop with validation, logging, and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config: dict,
        scaler,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.scaler = scaler

        training_cfg = config["training"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.batch_size = training_cfg["batch_size"]
        self.epochs = training_cfg["epochs"]
        self.lr = training_cfg["learning_rate"]
        self.weight_decay = training_cfg["weight_decay"]
        self.num_workers = training_cfg["num_workers"]

        self.artifacts_dir = Path(training_cfg["artifacts_dir"])
        self.checkpoint_dir = Path(training_cfg["checkpoint_dir"])
        ensure_dir(self.artifacts_dir)
        ensure_dir(self.checkpoint_dir)

        self.writer = SummaryWriter(log_dir=str(self.artifacts_dir / "tb_logs"))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.history = {"train_loss": [], "val_loss": []}

    def train(self) -> None:
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)

            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

            self._save_checkpoint(
                path=self.checkpoint_dir / "last.pt",
                epoch=epoch,
                val_loss=val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    path=self.checkpoint_dir / "best.pt",
                    epoch=epoch,
                    val_loss=val_loss,
                )

        plot_loss_curve(
            train_losses=self.history["train_loss"],
            val_losses=self.history["val_loss"],
            save_path=self.artifacts_dir / "loss_curve.png",
        )
        self.writer.close()

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_examples = 0

        progress = tqdm(loader, leave=False)
        for batch_x, batch_y in progress:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            with torch.set_grad_enabled(training):
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )