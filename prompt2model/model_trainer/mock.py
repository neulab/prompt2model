"""This module provides a dummy trainer for testing purposes."""

from typing import Any

import datasets
from transformers import PreTrainedModel  # noqa
from transformers import PreTrainedTokenizer

from prompt2model.model_trainer import BaseTrainer


class MockTrainer(BaseTrainer):
    """This dummy trainer does not actually train anything."""

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """This dummy trainer returns the given model without any training.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A HuggingFace model and tokenizer.
        """
        _ = training_datasets, hyperparameter_choices
        return self.model, self.tokenizer
