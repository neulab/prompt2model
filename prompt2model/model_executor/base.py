"""An interface for generating model outputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import datasets
import transformers


@dataclass(frozen=True)
class ModelOutput:
    """A model output for a single example.

    Attributes:
        prediction: The prediction by the model
        confidence: A confidence value in the prediction (or None)
        auxiliary_info: Any other auxiliary information provided by the model
    """

    prediction: Any
    confidence: float | None
    auxiliary_info: dict[str, Any]


class ModelExecutor(ABC):
    """An interface for automatic model evaluation."""

    @abstractmethod
    def make_predictions(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        test_set: datasets.Dataset,
        input_column: str,
    ) -> list[ModelOutput]:
        """Evaluate a model on a test set.

        Args:
            model: The model to evaluate.
            tokenizer: The model's associated tokenizer.
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.

        Returns:
            A list of model outputs, one for each element in the test set.
        """
