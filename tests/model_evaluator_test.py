"""Testing Seq2SeqEvaluator."""

import gc
import logging
from unittest.mock import patch

import evaluate
import pytest
from datasets import Dataset

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import ModelOutput

logger = logging.getLogger("ModelEvaluator")
# These following variables are initialized following the output of
# DatasetProcessor. The outputs of T5 do not contain the inputs,
# but the outputs of GPT2 do.

GROUND_TRUTH = ["The cat is sleeping.", "The dog is playing."]
MODEL_INPUTS = [
    "Translate Chinese to English: 猫在睡觉。",
    "Translate Chinese to English: 狗在玩耍。",
]

T5_PREDICTIONS = [
    ModelOutput("The cat is sleeping.", auxiliary_info={}),
    ModelOutput("The dog is barking.", auxiliary_info={}),
]

GPT_PREDICTIONS = [
    ModelOutput(
        "Translate Chinese to English: 猫在睡觉。The cat is sleeping.", auxiliary_info={}
    ),
    ModelOutput(
        "Translate Chinese to English: 狗在玩耍。The dog is barking.", auxiliary_info={}
    ),
]

# Note that there is no eos token in the validation dataset.
VALIDATION_DATASET = Dataset.from_dict(
    {"model_ouput": GROUND_TRUTH, "model_input": MODEL_INPUTS}
)


def test_t5_evaluator_with_default_metrics():
    """Test the Seq2SeqEvaluator with the output of T5."""
    evaluator = Seq2SeqEvaluator()
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        metric_values = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            predictions=T5_PREDICTIONS,
            encoder_model_name="xlm-roberta-base",
        )
        mock_info.assert_called_once_with(
            "Using default metrics of chr_f, exact_match and bert_score."
        )
        mock_warning.assert_not_called()
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5, 'average_bert_score': 0.97}
    assert len(metric_values.keys()) == 3
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    assert round(metric_values["average_bert_score"], 2) == 0.97
    gc.collect()


def test_gpt_evaluator_with_default_metrics():
    """Test the Seq2SeqEvaluator with the output of GPT."""
    evaluator = Seq2SeqEvaluator()
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        metric_values = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            predictions=GPT_PREDICTIONS,
            model_input_column="model_input",
            encoder_model_name="xlm-roberta-base",
        )
        info_list = [each.args[0] for each in mock_info.call_args_list]
        assert info_list == [
            "Using default metrics of chr_f, exact_match and bert_score.",
            "The model_input_column is not None. The model input will be detached from predictions if necessary.",  # noqa E501
        ]
        mock_warning.assert_not_called()
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5, 'average_bert_score': 0.97}
    assert len(metric_values.keys()) == 3
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    assert round(metric_values["average_bert_score"], 2) == 0.97
    gc.collect()


def test_t5_evaluator_with_selected_metrics():
    """Test the T5 Evaluator with chr_f, exact_match metrics."""
    evaluator = Seq2SeqEvaluator()
    metrics = [
        evaluate.load("chrf"),
        evaluate.load("exact_match"),
    ]
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        metric_values = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            predictions=GPT_PREDICTIONS,
            model_input_column="model_input",
            metrics=metrics,
            encoder_model_name="xlm-roberta-base",
        )
        info_list = [each.args[0] for each in mock_info.call_args_list]
        assert info_list == [
            "Using selected metrics: chr_f, exact_match.",
            "The model_input_column is not None. The model input will be detached from predictions if necessary.",  # noqa E501
        ]
        mock_warning.assert_not_called()
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5}
    assert len(metric_values.keys()) == 2
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    gc.collect()


def test_gpt_evaluator_with_selected_metrics():
    """Test the GPT Evaluator with chr_f, exact_match metrics."""
    evaluator = Seq2SeqEvaluator()
    metrics = [
        evaluate.load("chrf"),
        evaluate.load("exact_match"),
    ]
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        metric_values = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            model_input_column="model_input",
            predictions=T5_PREDICTIONS,
            metrics=metrics,
            encoder_model_name="xlm-roberta-base",
        )
        info_list = [each.args[0] for each in mock_info.call_args_list]
        assert info_list == [
            "Using selected metrics: chr_f, exact_match.",
            "The model_input_column is not None. The model input will be detached from predictions if necessary.",  # noqa E501
        ]
        mock_warning.assert_not_called()
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5}
    assert len(metric_values.keys()) == 2
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    gc.collect()


def test_evaluator_with_unsupported_metrics():
    """Test the Seq2SeqEvaluator with unsupported metrics."""
    # Metrics must be within chr_f exact_match and bert_score.
    evaluator = Seq2SeqEvaluator()
    # Evaluate the model
    metrics = [
        evaluate.load("accuracy"),
        evaluate.load("exact_match"),
    ]
    with pytest.raises(ValueError) as exc_info:
        _ = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            model_input_column="model_input",
            predictions=T5_PREDICTIONS,
            metrics=metrics,
            encoder_model_name="xlm-roberta-base",
        )
        assert (
            str(exc_info.value)
            == "Metrics must be within chr_f exact_match and bert_score."
        )
    gc.collect()


def test_evaluator_handle_deficient_predictions():
    """Test the evaluator handle the error of deficient predictions."""
    evaluator = Seq2SeqEvaluator()
    # The length of input dataset and predictions should be equal.
    deficient_predictions = [
        ModelOutput("The cat is sleeping.", auxiliary_info={}),
    ]
    with pytest.raises(ValueError) as exc_info:
        _ = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            predictions=deficient_predictions,
            encoder_model_name="xlm-roberta-base",
        )
        assert (
            str(exc_info.value)
            == "The length of input dataset and predictions are not equal."
        )
    gc.collect()


def test_gpt_evaluator_without_model_input_column():
    """Test Evaluator with the output of GPT but does specify model_input_column."""
    evaluator = Seq2SeqEvaluator()
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        metric_values = evaluator.evaluate_model(
            dataset=VALIDATION_DATASET,
            gt_column="model_ouput",
            predictions=GPT_PREDICTIONS,
            encoder_model_name="xlm-roberta-base",
        )
        info_list = [each.args[0] for each in mock_info.call_args_list]
        assert info_list == [
            "Using default metrics of chr_f, exact_match and bert_score.",
        ]
        mock_warning.assert_not_called()
    # Since the model_input_column is not specified, the model input will not be
    # detached from predictions. Thus the metric values will be relatively lower.
    # metric_values = {'chr_f++': 53.36, 'exact_match': 0.0, 'average_bert_score': 0.85}
    assert len(metric_values.keys()) == 3
    assert round(metric_values["chr_f++"], 2) == 53.36
    assert round(metric_values["exact_match"], 2) == 0.00
    assert round(metric_values["average_bert_score"], 2) == 0.85
    gc.collect()
