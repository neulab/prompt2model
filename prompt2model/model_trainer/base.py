"""An base class for trainers."""
from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers
from transformers import AutoModel, AutoTokenizer


# pylint: disable=too-few-public-methods
class BaseTrainer(ABC):
    """Train a model with a fixed set of hyperparameters."""

    def __init__(self, pretrained_model_name: str):
        """Initialize a model trainer.

        Args:
            pretrained_model_name: A HuggingFace model name to use for training.
        """
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, padding_side="left"
        )
        self.wandb = None

    @abstractmethod
    def train_model(
        self,
        hyperparameter_choices: dict[str, Any],
        training_datasets: list[datasets.Dataset],
        validation_datasets: list[datasets.Dataset] | None = None,
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a model with the given hyperparameters and return it."""
