"""Testing Seq2SeqEvaluator."""

from datasets import Dataset

from prompt2model.evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import ModelOutput


def test_seq2seq_evaluator():
    """Test the Seq2SeqEvaluator with ground truth dataset and model predictions."""
    evaluator = Seq2SeqEvaluator()

    # Define the ground truth dataset
    ground_truth = ["The cat is sleeping.", "The dog is playing."]

    # Define the model predictions
    predictions = [
        ModelOutput("The cat is sleeping.", confidence=0.9, auxiliary_info={}),
        ModelOutput("The dog is barking.", confidence=0.8, auxiliary_info={}),
    ]

    # Create a dummy dataset with the ground truth
    dataset = Dataset.from_dict({"output_col": ground_truth})

    # Evaluate the model
    metric_values = evaluator.evaluate_model(dataset, "output_col", predictions)

    # Assert the expected metric values
    # metric_values = {'chr_f++': 78.30, 'exact_match': 0.5, 'bert_score': [1.0, 0.95]}
    assert round(metric_values["chr_f++"], 2) == 78.30
    assert round(metric_values["exact_match"], 2) == 0.50
    assert all(score > 0.9 for score in metric_values["bert_score"])
    assert len(metric_values["bert_score"]) == 2
