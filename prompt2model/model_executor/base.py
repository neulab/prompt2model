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
        test_set: datasets.Dataset | None = None,
        input_column: str | None = None,
        batch_size: int = 10,
        tokenizer_max_length: int = 256,
        sequence_max_length: int = 512,
    ) -> None:
        """Initializes a new instance of ModelExecutor.

        Args:
            model: The model to evaluate.
            tokenizer: The model's associated tokenizer.
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.
            batch_size: The batch size to use when making predictions.
            tokenizer_max_length: The maximum number of tokens that
                tokenizer is allowed to generate.
            sequence_max_length: The maximum number of tokens to generate.
                This includes the input and output tokens.
        """
        self.model = model
        self.tokenizer = tokenizer
        assert (test_set is None) == (
            input_column is None
        ), "input_column and test_set should be provided simultaneously."
        if test_set and input_column:
            self.test_set = test_set
            self.input_column = input_column
        else:
            logging.info(
                (
                    "No test set and no input_column provided."
                    "This ModelExecutor will only be used to make"
                    " single predictions in DemoCreator."
                )
            )
        self.batch_size = batch_size
        if self.tokenizer.pad_token is None:
            logging.warning(
                "Trying to init an ModelExecutor's tokenizer without pad_token"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.eos_token_id
        self.tokenizer_max_length = tokenizer_max_length
        self.sequence_max_length = sequence_max_length
        if self.sequence_max_length is None:
            logging.warning(
                (
                    "The `max_length` in `self.model.generate` will default to "
                    f"`self.model.config.max_length` ({self.model.config.max_length})"
                    " if `sequence_max_length` is `None`."
                )
            )
        if hasattr(self.model.config, "max_position_embeddings"):
            max_embeddings = self.model.config.max_position_embeddings
            if sequence_max_length is not None and max_embeddings < sequence_max_length:
                logging.warning(
                    (
                        f"The sequence_max_length ({sequence_max_length})"
                        f" is larger than the max_position_embeddings ({max_embeddings})."  # noqa: E501
                        f" So the sequence_max_length will be set to {max_embeddings}."  # noqa: E501
                    )
                )
                self.sequence_max_length = max_embeddings

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
