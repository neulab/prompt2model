"""Testing TextToTextTrainer."""

import os
import shutil
import tempfile

import datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer

from prompt2model.model_trainer.text2text import TextToTextTrainer


def test_text_to_text_trainer():
    """Test the `train_model` function of a TextToTextTrainer.

    This function tests the TextToTextTrainer class by training it on a small T5 model
    and verifying that the trained model is a T5ForConditionalGeneration model and
    the trained tokenizer is a T5Tokenizer model.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a TextToTextTrainer instance with the cache directory
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
        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets, {"output_dir": cache_dir}
        )

        # Verify that the trained model is a T5ForConditionalGeneration model
        assert isinstance(trained_model, T5ForConditionalGeneration)

        # Verify that the trained tokenizer is a T5Tokenizer model
        assert isinstance(trained_tokenizer, T5Tokenizer)

    # Delete the wandb cache directory
    wandb_cache_dir = "./wandb"
    if os.path.exists(wandb_cache_dir):
        shutil.rmtree(wandb_cache_dir)
