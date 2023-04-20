"""This module provides a dummy trainer for testing purposes."""

from typing import Any

import datasets
import transformers

from prompt2model.trainer import Trainer


class MockTrainer(Trainer):
    """This dummy trainer does not actually train anything."""

    def __init__(self, retrieved_model: transformers.PreTrainedModel):
        """Initialize a dummy model trainer.

        Args:
            retrieved_model: A model to use for training.
        """
        self.model = retrieved_model
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        self.wandb = None
        raise NotImplementedError

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> transformers.PreTrainedModel:
        """This dummy trainer returns the given model without any training.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A HuggingFace model.
        """
        return self.model
