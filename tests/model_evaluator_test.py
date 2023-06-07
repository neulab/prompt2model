"""Testing Seq2SeqEvaluator."""

import evaluate
import pytest
from datasets import Dataset

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import ModelOutput

# Define the ground truth dataset
GROUND_TRUTH = ["The cat is sleeping.", "The dog is playing."]

# Define the model PREDICTIONS
PREDICTIONS = [
    ModelOutput("The cat is sleeping.", confidence=0.9, auxiliary_info={}),
    ModelOutput("The dog is barking.", confidence=0.8, auxiliary_info={}),
]

# Create a dummy dataset with the ground truth
DATASET = Dataset.from_dict({"output_col": GROUND_TRUTH})


def test_seq2seq_evaluator_with_default_metrics():
    """Test the Seq2SeqEvaluator with chrf, exact_match and bertscore metrics."""
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        DATASET, "output_col", PREDICTIONS, encoder_model_name="xlm-roberta-base"
    )
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5, 'bert_score': [1.0, 0.95]}
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    assert all(score > 0.9 for score in metric_values["bert_score"])
    assert len(metric_values["bert_score"]) == 2


def test_seq2seq_evaluator_with_selected_metrics():
    """Test the Seq2SeqEvaluator with chrf, exact_match metrics."""
    evaluator = Seq2SeqEvaluator()
    metrics = [
        evaluate.load("chrf"),
        evaluate.load("exact_match"),
    ]
    metric_values = evaluator.evaluate_model(
        DATASET,
        "output_col",
        PREDICTIONS,
        metrics=metrics,
        encoder_model_name="xlm-roberta-base",
    )
    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5}
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    assert "bert_score" not in metric_values


def test_seq2seq_evaluator_with_unsupported_metrics():
    """Test the Seq2SeqEvaluator with unsupported metrics."""
    evaluator = Seq2SeqEvaluator()
    # Evaluate the model
    metrics = [
        evaluate.load("accuracy"),
        evaluate.load("exact_match"),
    ]
    with pytest.raises(NotImplementedError):
        _ = evaluator.evaluate_model(
            DATASET,
            "output_col",
            PREDICTIONS,
            metrics=metrics,
            encoder_model_name="xlm-roberta-base",
        )
