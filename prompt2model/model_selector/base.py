"""An interface for model selection."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers
from prompt_parser.base import PromptSpec


# pylint: disable=too-few-public-methods
class ModelSelector(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def select_model(
        self,
        hyperparameters: dict[str, list[Any]] | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> transformers.PreTrainedModel:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            hyperparameters: (Optional) A dictionary of hyperparameter choices.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.

        Return:
            A model (with hyperparameters selected from the specified range).

        """


class DefaultParameterSelector(ModelSelector):
    """Uses a default set of parameters."""

    def __init__(
        self,
        trainer_type: type,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
    ):
        """Initialize with train/val datasets and a prompt specification."""
        self.trainer_type = trainer_type
        self.training_sets = training_sets
        self.validation = validation
        self.default_hyperparameter_choices = self._default_hyperparameter_choices()

    def _default_hyperparameter_choices(self) -> dict[str, Any]:
        """Extract or infer hyperparameter choices from the prompt specification."""
        return {
            "model": "t5-base",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        }

    def select_model(
        self,
        hyperparameters: dict[str, list[Any]] | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> transformers.PreTrainedModel:
        """Use a pre-defined default set of hyperparameters.

        Return:
            A model trained using default hyperparameters.
        """
        trainer = self.trainer_type(
            self.training_sets, self._default_hyperparameter_choices(), prompt_spec
        )
        single_model = trainer.train_model()
        return single_model
