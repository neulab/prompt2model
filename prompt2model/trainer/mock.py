"""This module provides a dummy trainer for testing purposes."""

from typing import Any

import datasets
import transformers

from prompt2model.prompt_parser import PromptSpec
from prompt2model.trainer import Trainer


class MockTrainer(Trainer):
    """This dummy trainer does not actually train anything."""

    def __init__(self):
        """Initialize a dummy BERT-based trainer."""
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        self.wandb = None
        raise NotImplementedError

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
        prompt_spec: PromptSpec,
    ) -> transformers.PreTrainedModel:
        """This dummy trainer returns an untrained BERT-base model.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.
            prompt_spec: A prompt specification.

        Returns:
            A trained HuggingFace model.
        """
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
        return model
