"""An interface for automatic model evaluation.

Input:
   1) Trained model
   2) Test set
   3) Metrics to use (currently, inferred from PromptSpec)

Output:
   Dictionary of metric values
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import datasets
import transformers
from prompt_parser.base import PromptSpec


class Evaluator(ABC):
    """An interface for automatic model evaluation."""

    @abstractmethod
    def evaluate_model(self, model: transformers.PreTrainedModel) -> dict[str, Any]:
        """Evaluate a model on a test set. The specific metrics to use are
        specified or inferred from the PromptSpec."""

    @abstractmethod
    def write_metrics(self, metrics_dict: dict[str, Any], metrics_path: str) -> None:
        """Write or display metrics to a file"""


class BaseEvaluator(Evaluator):
    """A dummy evaluator that always returns the same metric value."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        metrics: list[datasets.Metric],
        prompt_spec: Optional[PromptSpec],
    ) -> None:
        """Initialize with dataset and either a list of metrics or a prompt
        specification, from which the list of metrics is inferred."""
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
