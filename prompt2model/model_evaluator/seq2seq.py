"""An interface for automatically evaluate Seq2Seq generation model."""

from __future__ import annotations  # noqa FI58

import logging
from typing import Any

import datasets
import evaluate
import numpy as np

from prompt2model.model_evaluator.base import ModelEvaluator
from prompt2model.model_executor import ModelOutput


class Seq2SeqEvaluator(ModelEvaluator):
    """An evaluator computing `ChrF++`, `Exact Match` and `Embedding Distance`."""

    def evaluate_model(
        self,
        dataset: datasets.Dataset,
        gt_column: str,
        predictions: list[ModelOutput],
        model_input_column: str | None = None,
        metrics: list[evaluate.Metric] | None = None,
        encoder_model_name: str = "xlm-roberta-base",
    ) -> dict[str, Any]:
        """Evaluate a model on a test set.

        Args:
            dataset: The dataset to evaluate metrics on.
            gt_column: The dataset column to use as ground truth.
            predictions: Model outputs to evaluate.
            model_input_column: (Optional) For autoregistered models,
                the prediction sometimes contains the model input.
                So we need to delete the model input if it's in the predictions.
            metrics: (Optional) The metrics to use.

        Returns:
            A dictionary of metric values to return.
        """
        if metrics is not None:
            metric_names = [each.name for each in metrics]
            metric_names = sorted(metric_names, key=lambda name: name.lower())
            assert set(metric_names) < {
                "chr_f",
                "exact_match",
                "bert_score",
            }, "Metrics must be within chr_f, exact_match, and bert_score."
            logging.info(f"Using selected metrics: {', '.join(metric_names)}.")
        else:
            logging.info("Using default metrics of chrf, exact_match and bert_score.")
            metrics = [
                evaluate.load("chrf"),
                evaluate.load("exact_match"),
                evaluate.load("bertscore"),
            ]
        # Get the ground truth from the dataset
        ground_truths = dataset[gt_column]
        # Extract the predicted strings from ModelOutput
        predicted_strings = [each.prediction for each in predictions]
        assert len(ground_truths) == len(
            predicted_strings
        ), "The length of input dataset and predictions are not equal."
        # Initialize the metric values dictionary
        metric_values = {}

        if model_input_column is not None:
            logging.info(
                "The model_input_column is not None. The model input will be detached from predictions if necessary."  # noqa 501
            )
            model_inputs = dataset[model_input_column]
            for idx, model_input in enumerate(model_inputs):
                ground_truth = ground_truths[idx]
                predicted_string = predicted_strings[idx]
                if (model_input in predicted_string) and (
                    model_input not in ground_truth
                ):
                    predicted_string = predicted_string.replace(model_input, "")
                    predicted_strings[idx] = predicted_string

        # Compute and store metric values
        for metric in metrics:
            metric_name = metric.name
            assert metric_name in ["chr_f", "exact_match", "bert_score"]
            if metric_name == "chr_f":
                metric.add_batch(
                    predictions=predicted_strings, references=ground_truths
                )
                metric_values["chr_f++"] = metric.compute(word_order=2)["score"]
            elif metric_name == "exact_match":
                metric.add_batch(
                    predictions=predicted_strings, references=ground_truths
                )
                metric_values[metric_name] = metric.compute()["exact_match"]
            elif metric_name == "bert_score":
                metric.add_batch(
                    predictions=predicted_strings, references=ground_truths
                )
                metric_values["average_bert_score"] = np.average(
                    metric.compute(model_type=encoder_model_name)["f1"]
                )

        return metric_values
