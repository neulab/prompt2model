"""An interface for automatic model evaluation."""

from __future__ import annotations  # noqa FI58

import json
from abc import ABC, abstractmethod
from typing import Any

import datasets
import evaluate

from prompt2model.model_executor import ModelOutput


class ModelEvaluator(ABC):
    """An interface for automatic model evaluation."""

    @abstractmethod
    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        gt_column: str,
        predictions: list[ModelOutput],
        model_input_column: str | None = None,
        metrics: list[evaluate.Metric] | None = None,
        encoder_model_name: str = "xlm-roberta-base",
    ) -> dict[str, Any]:
        """Evaluate a model on a test set..

        Args:
            dataset: The dataset to evaluate metrics on.
            gt_column: The dataset column to use as ground truth.
            predictions: Model outputs to evaluate.
            metrics: (Optional) The metrics to use, defaults to using chr_f,
                exact_match, and bert_score.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.

        Returns:
            A dictionary of metric values to return.
        """

    def write_metrics(self, metrics_dict: dict[str, Any], metrics_path: str) -> None:
        """This function writes metrics to a file.

        Args:
            metrics_dict: A dictionary of metrics to write.
            metrics_path: The file path to write metrics to.

        """
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f)
