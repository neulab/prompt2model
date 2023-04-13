"""An interface for automatic model evaluation.

Input:
   1) Trained model
   2) Test set
   3) Metrics to use (currently, inferred from PromptSpec)

Output:
   Dictionary of metric values
"""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers
from prompt_parser.base import PromptSpec


class Evaluator(ABC):
    """An interface for automatic model evaluation."""

    @abstractmethod
    def evaluate_model(self, model: transformers.PreTrainedModel) -> dict[str, Any]:
        """Evaluate a model on a test set.

        Args:
            model: The model to evaluate.

        Returns:
            A dictionary of metric values to return.

        """

    @abstractmethod
    def write_metrics(self, metrics_dict: dict[str, Any], metrics_path: str) -> None:
        """Write or display metrics to a file.

        Args:
            metrics_dict: A dictionary of metrics to write.
            metrics_path: The file path to write metrics to.

        """


class BaseEvaluator(Evaluator):
    """A dummy evaluator that always returns the same metric value."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        metrics: list[datasets.Metric] | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> None:
        """Initialize the evaluation setting.

        Args:
            dataset: The dataset to evaluate metrics on.
            metrics: (Optional) The metrics to use.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.

        """
        self.test_data = dataset
        self.metrics = metrics
        self.prompt_spec = prompt_spec

    def evaluate_model(
        self,
        model: transformers.PreTrainedModel,
    ) -> dict[str, Any]:
        """Return empty metrics dictionary."""
        return {}

    def write_metrics(self, metrics_dict: dict[str, Any], metrics_path: str) -> None:
        """Do nothing."""
        _ = metrics_dict, metrics_path  # suppress unused variable warnings
