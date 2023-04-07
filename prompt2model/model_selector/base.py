"""An interface for model selection.

Input:
   1) training dataset (datasets.Dataset)
   2) validation dataset (datasets.Dataset)
   3) Dictionary-of-lists consisting of hyperparameter
      values to consider (e.g. different base models to
      consider, different optimizers, different LRs, etc)

Output:
   A single model (transformers.PreTrainedModel)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import datasets
import transformers
from prompt_parser.base import PromptSpec


class ModelSelector(ABC):
    """
    Select a good model from among a set of hyperparameter choices.
    """

    @abstractmethod
    def select_model(
        self,
        hyperparameters: Optional[dict[str, list[Any]]],
        prompt_spec: Optional[PromptSpec],
    ) -> transformers.PreTrainedModel:
        """
        Select a model from among the hyperparameter choices, potentially
        by calling a third-party library or API.
        Hyperparameter choices may be set to a default value or inferred
        from the prompt specification.
        """


class DefaultParameterSelector(ModelSelector):
    """
    Uses a default set of parameters.
    """

    def __init__(
        self,
        trainer_type: Type,
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
        hyperparameters: Optional[dict[str, list[Any]]] = None,
        prompt_spec: Optional[PromptSpec] = None,
    ) -> transformers.PreTrainedModel:
        """
        Select a model from among the hyperparameter choices, potentially
        by calling a third-party library or API.
        """
        trainer = self.trainer_type(
            self.training_sets, self._default_hyperparameter_choices(), prompt_spec
        )
        single_model = trainer.train_model()
        return single_model
