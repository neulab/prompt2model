"""An interface for automatically evaluate Seq2Seq generation model."""

from prompt2model.evaluator.base import Evaluator
from typing import Any

import datasets

from prompt2model.model_executor import ModelOutput
from prompt2model.prompt_parser import PromptSpec
from prompt2model.model_executor import GenerationModelExecutor

class Seq2SeqEvaluator(Evaluator):
    """An evaluator computing `ChrF++`, `Exact Match` and `Embedding Distance`."""

    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        gt_column: str,
        predictions: list[ModelOutput],
        metrics: list[datasets.Metric] | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> dict[str, Any]:
        """Evaluate a model on a test set..

        Args:
            dataset: The dataset to evaluate metrics on.
            gt_column: The dataset column to use as ground truth.
            predictions: Model outputs to evaluate.
            metrics: (Optional) The metrics to use.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.

        Returns:
            A dictionary of metric values to return.
        """
