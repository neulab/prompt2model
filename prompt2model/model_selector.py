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

from typing import Any

import datasets
import transformers

from prompt_parser import PromptSpec
from trainer import Trainer


class ModelSelector:
    """
    Select a good model based on validation metrics from among a set of
    hyperparameter choices.
    """
    def __init__(
        self,
        training_sets: list[datasets.Dataset],
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
    ):
        """Initialize with train/val datasets and a prompt specification."""
        self.training_sets = training_sets
        self.validation = validation
        self.prompt_spec = prompt_spec
        self.hyperparameter_choices = self._extract_hyperparameter_choices()

    def _extract_hyperparameter_choices(self) -> dict[str, list[Any]]:
        """Extract or infer hyperparameter choices from the prompt specification."""
        return {
            "model": ["bert-base-uncased", "t5-base"],
            "optimizer": ["adam", "sgd+m"],
            "learning_rate": [0.0001, 0.001, 0.01],
        }

    def select_model(
        self,
    ) -> transformers.PreTrainedModel:
        """
        Select a model from among the hyperparameter choices, potentially
        by calling a third-party library or API.
        """
        t = Trainer(self.training_sets, self.hyperparameter_choices, self.prompt_spec)
        single_model = t.train_model()
        return single_model
