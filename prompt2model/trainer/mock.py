"""This module provides a dummy trainer for testing purposes."""

from typing import Any

import datasets
from transformers import AutoModel, PreTrainedModel

from prompt2model.trainer import Trainer


class MockTrainer(Trainer):
    """This dummy trainer does not actually train anything."""

    def __init__(self, pretrained_model_name: str):
        """Initialize a dummy model trainer.

        Args:
            pretrained_model_name: A HuggingFace model ID to use for training.
        """
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        self.wandb = None
        raise NotImplementedError

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> PreTrainedModel:
        """This dummy trainer returns the given model without any training.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A pretrained HuggingFace model.
        """
        return self.model
