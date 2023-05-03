"""Testing TextToTextTrainer."""

import datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer

from prompt2model.model_trainer.text2text import TextToTextTrainer


def test_text_to_text_trainer():
    """Test the `train_model` function of a TextToTextTrainer.

    This function tests the TextToTextTrainer class by training it on a small T5 model
    and verifying that the trained model is a T5ForConditionalGeneration model and
    the trained tokenizer is a T5Tokenizer model.
    """
    trainer = TextToTextTrainer("t5-small")

    training_datasets = [
        datasets.Dataset.from_dict(
            {
                "input_col": ["translate apple to french"] * 2,
                "output_col": ["pomme"] * 2,
            }
        ),
    ]

    # Train the TextToTextTrainer instance on the training dataset
    trained_model, trained_tokenizer = trainer.train_model(training_datasets, {})

    # Verify that the trained model is a T5ForConditionalGeneration model
    assert isinstance(trained_model, T5ForConditionalGeneration)

    # Verify that the trained tokenizer is a T5Tokenizer model
    assert isinstance(trained_tokenizer, T5Tokenizer)
