"""An interface for automatically evaluate Seq2Seq generation model."""

from __future__ import annotations  # noqa FI58

import logging
from typing import Any

import datasets
import evaluate

from prompt2model.model_evaluator.base import Evaluator
from prompt2model.model_executor import ModelOutput
from prompt2model.prompt_parser import PromptSpec


class Seq2SeqEvaluator(Evaluator):
    """An evaluator computing `ChrF++`, `Exact Match` and `Embedding Distance`."""

    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        gt_column: str,
        predictions: list[ModelOutput],
        metrics: list[evaluate.Metric] | None = None,
        prompt_spec: PromptSpec | None = None,
        encoder_model_name: str = "xlm-roberta-base",
    ) -> dict[str, Any]:
        """Evaluate a model on a test set.

        Args:
            dataset: The dataset to evaluate metrics on.
            gt_column: The dataset column to use as ground truth.
            predictions: Model outputs to evaluate.
            metrics: (Optional) The metrics to use.
            prompt_spec: (Optional) A PromptSpec to infer the metrics from.
            encoder_model_name: Encoder model to use in metric like BertScore.

        Returns:
            A dictionary of metric values to return.
        """
        if prompt_spec is not None:
            logging.error("prompt_spec is not supported for Seq2SeqEvaluator")
            raise NotImplementedError
        if metrics is not None:
            metric_names = {metric.name for metric in metrics}
            is_subset = metric_names < {"chr_f", "exact_match", "bert_score"}
            if not is_subset:
                logging.error("metrics must be within chr_f exact_match and bert_score")
                raise NotImplementedError
            else:
                logging.basicConfig(level=logging.INFO)
                logging.info("using metrics: " + str(metric_names))
        else:
            logging.basicConfig(level=logging.INFO)
            logging.info("using default metrics of chrf, exact_match and bertscore")
            metrics = [
                evaluate.load("chrf"),
                evaluate.load("exact_match"),
                evaluate.load("bertscore"),
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
            assert metric_name in ["chr_f", "exact_match", "bert_score"]
            if metric_name == "chr_f":
                metric.add_batch(predictions=predicted_strings, references=ground_truth)
                metric_values["chr_f++"] = metric.compute(word_order=2)["score"]
            elif metric_name == "exact_match":
                metric.add_batch(predictions=predicted_strings, references=ground_truth)
                metric_values[metric_name] = metric.compute()["exact_match"]
            elif metric_name == "bert_score":
                metric.add_batch(predictions=predicted_strings, references=ground_truth)
                metric_values[metric_name] = metric.compute(
                    model_type=encoder_model_name
                )["f1"]

        return metric_values