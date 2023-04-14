"""A dummy evaluator for testing purposes."""
from typing import Any

import datasets
import transformers

from prompt2model.evaluator.base import Evaluator
from prompt2model.prompt_parser import PromptSpec


class MockEvaluator(Evaluator):
    """A dummy evaluator that always returns the same metric value."""

    def __init__(self) -> None:
        """Initialize the evaluation setting."""

    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        model: transformers.PreTrainedModel,
        metrics: list[datasets.Metric] | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> dict[str, Any]:
        """Return empty metrics dictionary.

        Args:
            dataset: The dataset to evaluate metrics on.
            model: The model to evaluate.
            metrics: (Optional) The metrics to use.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.

        Returns:
            An empty dictionary (for testing purposes).
        """
        return {}

    def write_metrics(self, metrics_dict: dict[str, Any], metrics_path: str) -> None:
        """Do nothing."""
        _ = metrics_dict, metrics_path  # suppress unused variable warnings
