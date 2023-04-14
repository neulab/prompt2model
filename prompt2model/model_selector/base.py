"""An interface for model selection."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import transformers
from prompt_parser.base import PromptSpec


# pylint: disable=too-few-public-methods
class ModelSelector(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def select_model(
        self,
        prompt_spec: PromptSpec,
        hyperparameters: dict[str, list[Any]] | None = None,
    ) -> transformers.PreTrainedModel:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            hyperparameters: (Optional) A dictionary of hyperparameter choices.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.

        Return:
            A model (with hyperparameters selected from the specified range).

        """
