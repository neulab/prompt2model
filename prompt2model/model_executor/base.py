"""An interface for generating model outputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers


class ModelOutputs:
    """A class to hold model outputs."""

    def __init__(
        self, predictions: list[Any], probabilities: list[Any] | None = None
    ) -> None:
        """Initialize ModelOutputs class.

        Args:
            predictions: A list of model predictions.
            probabilities: (Optional) A list of model probabilities.
        """
        self.predictions = predictions
        self.probabilities = probabilities


class ModelExecutor(ABC):
    """An interface for automatic model evaluation."""

    @abstractmethod
    def make_predictions(
        self,
        model: transformers.PreTrainedModel,
        test_set: datasets.Dataset,
    ) -> ModelOutputs:
        """Evaluate a model on a test set.

        Args:
            model: The model to evaluate.
            test_set: The dataset to make predictions on.

        Returns:
            An object containing model outputs.
        """
