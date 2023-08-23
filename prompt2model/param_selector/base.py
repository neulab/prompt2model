"""An interface for model selection."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers

from prompt2model.prompt_parser.base import PromptSpec


# pylint: disable=too-few-public-methods
class ParamSelector(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def select_from_hyperparameters(
        self,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        hyperparameters: dict[str, list[Any]],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            hyperparameters: A dictionary of hyperparameter choices.

        Return:
            A model and tokenizer (with hyperparameters from given range).
        """

    @abstractmethod
    def select_from_spec(
        self,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            prompt_spec: A prompt to infer hyperparameters from.

        Return:
            A model and tokenizer (with hyperparameters from inferred range).
        """
