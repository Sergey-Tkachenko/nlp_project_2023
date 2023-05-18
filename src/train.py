import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import wandb
import transformers
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class TrainConfig:
    model: str
    checkpoints_folder: str
    device: torch.device

    batch_size: int
    epochs: int
    max_length: int

    lr: float


class Trainer:
    checkpoint_field_model: str = "model"
    checkpoint_field_optimizer: str = "optimizer"
    checkpoint_field_epoch: int = "epoch"

    checkpoint_last_name: str = "last.tar"

    # TODO add logging of the best
    checkpoint_best_name: str = "best.tar"

    def __init__(self, model: transformers.DebertaModel, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config: TrainConfig,
        wandb_project: str,
    ) -> None:

        wandb.init(project=wandb_project, config=asdict(config))

        self.model.to(config.device)

        if not os.path.exists(config.checkpoints_folder):
            os.makedirs(config.checkpoints_folder)

        start_epoch = self.load_checkpoint(config.checkpoints_folder) + 1

        for epoch in range(start_epoch, config.epochs):
            self.make_train_step(train_dataloader)

            self.make_evaluation_step(val_dataloader)

            self.save_checkpoint(config.checkpoints_folder, epoch)

        wandb.finish()

    def make_inference(self, dataloader: torch.utils.data.DataLoader) -> tuple:

        self.model.eval()

        with torch.no_grad():
            predicts = []
            labels = []

            for batch in tqdm(dataloader):
                labels.append(batch["labels"])

                batch = Trainer._move_dict_items_to_device(batch, self.model.device)

                outputs = self.model(**batch)

                predicts.append(outputs["logits"].cpu())

        return torch.cat(predicts), torch.cat(labels).squeeze()

    @staticmethod
    def _move_dict_items_to_device(target_dict: dict, device: str):
        return {key: target_dict[key].squeeze().to(device) for key in target_dict}

    # TODO: get rid of manual metric specification
    def make_evaluation_step(self, dataloader: torch.utils.data.DataLoader, return_labels: bool = True):

        logits, labels = self.make_inference(dataloader)

        predicted_probas = torch.softmax(logits, dim=-1).numpy()
        predicted_labels = torch.argmax(logits, dim=-1).numpy()

        wandb.log(
            {
                "accuracy": accuracy_score(labels, predicted_labels),
                "f1": f1_score(labels, predicted_labels),
                "recall": recall_score(labels, predicted_labels),
                "precision": precision_score(labels, predicted_labels),
                "auc_score": roc_auc_score(labels, predicted_probas[:, 1]),
            }
        )

    def make_train_step(self, dataloader: torch.utils.data.DataLoader):

        self.model.train()

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()

            batch = Trainer._move_dict_items_to_device(batch, self.model.device)

            outputs = self.model(**batch)

            loss = outputs["loss"]
            loss.backward()

            wandb.log({"train_loss": loss.detach().cpu().numpy()})

            self.optimizer.step()

    def load_checkpoint(self, folder: str) -> int:

        checkpoint_path = Path(folder) / Trainer.checkpoint_last_name

        if not os.path.exists(checkpoint_path):
            warnings.warn("No checkpoints found. Start epoch 0 with given model and optimizer.")

            return -1

        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)

        self.model.load_state_dict(checkpoint[Trainer.checkpoint_field_model])
        self.optimizer.load_state_dict(checkpoint[Trainer.checkpoint_field_optimizer])

        return checkpoint[Trainer.checkpoint_field_epoch]

    def save_checkpoint(self, folder: str, epoch: int) -> None:
        checkpoint_name = Path(folder) / Trainer.checkpoint_last_name

        torch.save(
            {
                Trainer.checkpoint_field_model: self.model.state_dict(),
                Trainer.checkpoint_field_optimizer: self.optimizer.state_dict(),
                Trainer.checkpoint_field_epoch: epoch,
            },
            checkpoint_name,
        )
