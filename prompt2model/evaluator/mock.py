"""A dummy evaluator for testing purposes."""
from typing import Any

import datasets
import transformers
from evaluator.base import Evaluator
from prompt_parser import PromptSpec


class MockEvaluator(Evaluator):
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
