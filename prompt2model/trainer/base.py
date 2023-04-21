"""An interface for trainers."""

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers


# pylint: disable=too-few-public-methods
class Trainer(ABC):
    """Train a model with a fixed set of hyperparameters."""

    @abstractmethod
    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a model with the given hyperparameters and return it."""
