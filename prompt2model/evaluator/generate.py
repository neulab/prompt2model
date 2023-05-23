"""An interface for automatically evaluate Seq2Seq generation model."""

from prompt2model.evaluator.base import Evaluator
from typing import Any

import datasets
from datasets import load_metric
from prompt2model.model_executor import ModelOutput
from prompt2model.prompt_parser import PromptSpec

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
        _ = prompt_spec
        if metrics is None:
            # Load the required metrics
            metrics = [
                load_metric("chrf"),
                load_metric("exact_match"),
                load_metric("embedding_distance"),
            ]
        # Get the ground truth from the dataset
        ground_truth = dataset[gt_column]

        # Extract the predicted strings from ModelOutput
        predicted_strings = [each.prediction for each in predictions]

        # Initialize the metric values dictionary
        metric_values = {}

        # Compute and store metric values
        for metric in metrics:
            metric_name = metric.name
            if metric_name == "chrf":
                references = ground_truth.tolist()
                metric.add_batch(predictions=predicted_strings, references=references)
                metric_values[metric_name] = metric.compute()
            elif metric_name == "exact_match":
                references = ground_truth.tolist()
                metric.add_batch(predictions=predicted_strings, references=references)
                metric_values[metric_name] = metric.compute()["exact_match"]
            elif metric_name == "embedding_distance":
                references = ground_truth.tolist()
                metric.add_batch(predictions=predicted_strings, references=references)
                metric_values[metric_name] = metric.compute()["distance"]

        return metric_values