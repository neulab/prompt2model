"""Mock model selector for testing purposes."""
from __future__ import annotations

from typing import Any

import datasets
import transformers

from prompt2model.model_trainer import BaseTrainer
from prompt2model.param_selector.base import ParamSelector
from prompt2model.prompt_parser import PromptSpec


class MockParamSelector(ParamSelector):
    """Uses a default set of parameters."""

    def __init__(self, trainer: BaseTrainer):
        """Initialize with train/val datasets and a prompt specification.

        Args:
            trainer: A trainer to use for training models during model selection.
        """
        self.trainer = trainer

    def _example_hyperparameter_choices(self) -> dict[str, Any]:
        """Example hyperparameters (for testing only)."""
        return {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        }

    def select_from_hyperparameters(
        self,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        hyperparameters: dict[str, list[Any]],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Use a pre-defined default set of hyperparameters.

        Args:
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.
            hyperparameters: (Optional) A dictionary of hyperparameter choices.

        Return:
            A model and tokenizer (trained using default hyperparameters).
        """
        single_model = self.trainer.train_model(
            self._example_hyperparameter_choices(), training_sets
        )
        return single_model

    def select_from_spec(
        self,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """The MockParamSelector cannot infer hyperparameters from the spec.

        Args:
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.
            hyperparameters: (Optional) A dictionary of hyperparameter choices.
        """
        raise NotImplementedError
