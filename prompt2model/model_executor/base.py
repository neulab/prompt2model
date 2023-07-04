"""An interface for generating model outputs."""

from __future__ import annotations  # noqa FI58

import logging
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

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        test_set: datasets.Dataset,
        input_column: str,
        batch_size: int = 10,
        max_new_tokens: int = 1024,
    ) -> None:
        """Initializes a new instance of ModelExecutor.

        Args:
            model: The model to evaluate.
            tokenizer: The model's associated tokenizer.
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.
            batch_size: The batch size to use when making predictions.
            max_new_tokens: The max_new_tokens to use when making predictions.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.test_set = test_set
        self.input_column = input_column
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        if self.tokenizer.pad_token is None:
            logging.warning(
                "Trying to init an ModelExecutor's tokenizer without pad_token"
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def make_prediction(self) -> list[ModelOutput]:
        """Evaluate a model on a test set.

        Returns:
            A list of model outputs, one for each element in the test set.
        """

    @abstractmethod
    def make_single_prediction(self, model_input: str) -> ModelOutput:
        """Make prediction on one example.

        Args:
            model_input: The input string to the model.

        Returns:
            A single model output, useful for exposing a model to a user interface.
        """
