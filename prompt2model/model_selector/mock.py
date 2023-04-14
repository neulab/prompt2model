"""Mock model selector for testing purposes."""

from typing import Any

import datasets
import transformers

from prompt2model.model_selector.base import ModelSelector
from prompt2model.prompt_parser import PromptSpec
from prompt2model.trainer import Trainer


class MockModelSelector(ModelSelector):
    """Uses a default set of parameters."""

    def _example_hyperparameter_choices(self) -> dict[str, Any]:
        """Example hyperparameters (for testing only)."""
        return {
            "model": "t5-base",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        }

    def select_model(
        self,
        trainer: Trainer,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
        hyperparameters: dict[str, list[Any]] | None = None,
    ) -> transformers.PreTrainedModel:
        """Use a pre-defined default set of hyperparameters.

        Args:
            trainer: A trainer object.
            training_sets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            prompt_spec: (Optional) A prompt to infer hyperparameters from.
            hyperparameters: (Optional) A dictionary of hyperparameter choices.

        Return:
            A model trained using default hyperparameters.
        """
        single_model = trainer.train_model(
            training_sets, self._example_hyperparameter_choices(), prompt_spec
        )
        return single_model
