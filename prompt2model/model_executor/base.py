"""An interface for generating model outputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import datasets
import transformers

from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("ModelExecutor")


@dataclass(frozen=False)
class ModelOutput:
    """A model output for a single example.

    Attributes:
        prediction: The prediction by the model.
        auxiliary_info: Any other auxiliary information provided by the model.
    """

    prediction: Any
    auxiliary_info: dict[str, Any]


class ModelExecutor(ABC):
    """An interface for automatic model evaluation."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        batch_size: int = 10,
        tokenizer_max_length: int = 256,
        sequence_max_length: int = 512,
    ) -> None:
        """Initializes a new instance of ModelExecutor.

        Args:
            model: The model to evaluate.
            tokenizer: The model's associated tokenizer.
            batch_size: The batch size to use when making predictions.
            tokenizer_max_length: The maximum number of tokens that
                tokenizer is allowed to generate.
            sequence_max_length: The maximum number of tokens in
                the input and output.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if self.tokenizer.pad_token is None:
            logger.warning(
                "Trying to init an ModelExecutor's tokenizer without pad_token."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer_max_length = tokenizer_max_length
        self.sequence_max_length = sequence_max_length
        if self.sequence_max_length is None:
            max_length = self.model.config.max_length
            logger.warning(
                (
                    "The `max_length` in `self.model.generate` will default to "
                    f"`self.model.config.max_length` ({max_length})"
                    " if `sequence_max_length` is `None`."
                )
            )
            self.sequence_max_length = max_length
        if hasattr(self.model.config, "max_position_embeddings"):
            max_embeddings = self.model.config.max_position_embeddings
            if sequence_max_length is not None and sequence_max_length > max_embeddings:
                logger.warning(
                    (
                        f"The sequence_max_length ({sequence_max_length})"
                        f" is larger than the max_position_embeddings ({max_embeddings})."  # noqa: E501
                        f" So the sequence_max_length will be set to {max_embeddings}."  # noqa: E501
                    )
                )
                self.sequence_max_length = max_embeddings

    @abstractmethod
    def make_prediction(
        self,
        test_set: datasets.Dataset,
        input_column: str,
    ) -> list[ModelOutput]:
        """Evaluate a model on a test set.

        Args:
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.

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
