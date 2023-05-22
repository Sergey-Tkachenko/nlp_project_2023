import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import wandb
import transformers
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput


@dataclass
class TrainConfig:
    model: str
    checkpoints_folder: str
    device: torch.device

    batch_size: int
    epochs: int
    max_length: int

    lr: float
    test_sets: list
    val_sets: list
    train_sets: list

    others: dict


class Logger(ABC):
    @abstractmethod
    def log(self, data: dict[Any]):
        pass

    def finish(self):
        pass


class DummyLogger(Logger):
    def log(self, data: dict[Any, Any]):
        pass


class WandbLogger(Logger):
    def __init__(self, project: str, experiment_config: TrainConfig) -> None:
        super().__init__()

        wandb.init(project=project, config=experiment_config)

    def log(self, data: dict[Any, Any]):
        wandb.log(data)

    def finish(self):
        wandb.finish()


class Trainer:
    checkpoint_field_model: str = "model"
    checkpoint_field_optimizer: str = "optimizer"
    checkpoint_field_epoch: int = "epoch"

    checkpoint_last_name: str = "last.tar"
    checkpoint_best_name: str = "best.tar"

    target_metric: str = "f1"

    def __init__(self, model: transformers.DebertaModel, optimizer: torch.optim.Optimizer, logger: Logger) -> None:
        self.model = model
        self.optimizer = optimizer

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.logger = logger

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config: TrainConfig,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> None:

        self.model.to(config.device)

        if not os.path.exists(config.checkpoints_folder):
            os.makedirs(config.checkpoints_folder)

        start_epoch = (
            self.load_checkpoint(self._build_checkpoint_path(config.checkpoints_folder, self.checkpoint_last_name)) + 1
        )

        best_accuracy = 0
        best_path = self._build_checkpoint_path(config.checkpoints_folder, self.checkpoint_best_name)

        for epoch in range(start_epoch, config.epochs):
            self.make_train_step(train_dataloader)

            metrics = self.make_evaluation_step(val_dataloader)
            self.logger.log(metrics)

            test_metrics = self.make_evaluation_step(test_dataloader)
            test_metrics = {"test_" + name: test_metrics[name] for name in test_metrics}
            self.logger.log(test_metrics)

            self.save_checkpoint(
                self._build_checkpoint_path(config.checkpoints_folder, self.checkpoint_last_name), epoch
            )

            if metrics[self.target_metric] > best_accuracy:
                self.logger.log({"current_best_epoch": epoch})

                self.save_checkpoint(best_path, epoch)
                best_accuracy = metrics[self.target_metric]

        # use best for evaluation on test
        if not os.path.exists(best_path):
            warnings.warn("Best checkpoint have not been found, using the last one.")
            best_path = self._build_checkpoint_path(config.checkpoints_folder, self.checkpoint_last_name)

        self.load_checkpoint(best_path, True)

    def _convert_to_logits_if_needed(self, output=SequenceClassifierOutput | torch.Tensor):
        if isinstance(output, SequenceClassifierOutput):
            return output["logits"]
        else:
            return output

    @staticmethod
    def _build_checkpoint_path(folder: str, name: str):
        return Path(folder) / name

    def make_inference(self, dataloader: torch.utils.data.DataLoader) -> tuple:

        self.model.eval()

        with torch.no_grad():
            predicts = []
            labels = []

            for batch in tqdm(dataloader):
                labels.append(batch.pop("labels"))

                batch = Trainer._move_dict_items_to_device(batch, self.model.device)

                logits = self._convert_to_logits_if_needed(self.model(**batch))

                predicts.append(logits.cpu())

        return torch.cat(predicts), torch.cat(labels).squeeze()

    @staticmethod
    def _move_dict_items_to_device(target_dict: dict, device: str):
        return {key: target_dict[key].squeeze().to(device) for key in target_dict}

    # TODO: get rid of manual metric specification
    def make_evaluation_step(self, dataloader: torch.utils.data.DataLoader, return_labels: bool = True):

        logits, labels = self.make_inference(dataloader)

        predicted_probas = torch.softmax(logits, dim=-1).numpy()
        predicted_labels = torch.argmax(logits, dim=-1).numpy()

        metrics_dict = {
            "accuracy": accuracy_score(labels, predicted_labels),
            "f1": f1_score(labels, predicted_labels),
            "recall": recall_score(labels, predicted_labels),
            "precision": precision_score(labels, predicted_labels),
            "auc_score": roc_auc_score(labels, predicted_probas[:, 1]),
        }

        return metrics_dict

    def make_train_step(self, dataloader: torch.utils.data.DataLoader):

        self.model.train()

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()

            batch = Trainer._move_dict_items_to_device(batch, self.model.device)

            labels = batch.pop("labels")

            logits = self._convert_to_logits_if_needed(self.model(**batch))

            loss = self.loss_function(logits, labels)

            loss.backward()

            self.logger.log({"train_loss": loss.detach().cpu().numpy()})

            self.optimizer.step()

    def load_checkpoint(self, checkpoint_path: Path, model_only: bool = False) -> int:

        if not os.path.exists(checkpoint_path):
            warnings.warn(f"No checkpoints found {checkpoint_path}. Start epoch 0 with given model and optimizer.")
            return -1

        warnings.warn(f"Using checkpoint from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)

        self.model.load_state_dict(checkpoint[Trainer.checkpoint_field_model])

        if model_only:
            return -1

        self.optimizer.load_state_dict(checkpoint[Trainer.checkpoint_field_optimizer])

        return checkpoint[Trainer.checkpoint_field_epoch]

    def save_checkpoint(self, checkpoint_path: Path, epoch: int) -> None:

        torch.save(
            {
                Trainer.checkpoint_field_model: self.model.state_dict(),
                Trainer.checkpoint_field_optimizer: self.optimizer.state_dict(),
                Trainer.checkpoint_field_epoch: epoch,
            },
            checkpoint_path,
        )
