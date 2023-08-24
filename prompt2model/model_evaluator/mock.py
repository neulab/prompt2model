"""A dummy evaluator for testing purposes."""
from __future__ import annotations  # noqa FI58

from typing import Any

import datasets
import evaluate

from prompt2model.model_evaluator.base import ModelEvaluator
from prompt2model.model_executor import ModelOutput


class MockEvaluator(ModelEvaluator):
    """A dummy evaluator that always returns the same metric value."""

    def __init__(self) -> None:
        """Initialize the evaluation setting."""

    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        gt_column: str,
        predictions: list[ModelOutput],
        model_input_column: str | None = None,
        metrics: list[evaluate.Metric] | None = None,
        encoder_model_name: str = "xlm-roberta-base",
    ) -> dict[str, Any]:
        """Return empty metrics dictionary.

        Args:
            dataset: The dataset to evaluate metrics on.
            gt_column: The dataset column to use as ground truth.
            predictions: Corresponding model outputs to evaluate.
            metrics: (Optional) The metrics to use, defaults to using
                chr_f, exact_match, and bert_score.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.

        Returns:
            An empty dictionary (for testing purposes).
        """
        return {}
