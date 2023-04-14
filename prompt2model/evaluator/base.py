"""An interface for automatic model evaluation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import transformers


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
