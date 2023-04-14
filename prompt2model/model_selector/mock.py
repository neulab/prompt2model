"""Mock model selector for testing purposes."""

from typing import Any

import datasets
import transformers
from model_selector.base import ModelSelector
from prompt_parser import PromptSpec
from trainer import Trainer


class MockModelSelector(ModelSelector):
    """Uses a default set of parameters."""

    def __init__(
        self,
        trainer: Trainer,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
    ):
        """Initialize with train/val datasets and a prompt specification."""
        self.trainer = trainer
        self.training_sets = training_sets
        self.validation = validation

    def _example_hyperparameter_choices(self) -> dict[str, Any]:
        """Example hyperparameters (for testing only)."""
        return {
            "model": "t5-base",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        }

    def select_model(
        self,
        prompt_spec: PromptSpec,
        hyperparameters: dict[str, list[Any]] | None = None,
    ) -> transformers.PreTrainedModel:
        """Use a pre-defined default set of hyperparameters.

        Return:
            A model trained using default hyperparameters.
        """
        single_model = self.trainer.train_model(
            self.training_sets, self._example_hyperparameter_choices(), prompt_spec
        )
        return single_model
