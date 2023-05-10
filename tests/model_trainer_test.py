"""Testing T5Trainer."""

import tempfile

import datasets
import transformers

from prompt2model.model_trainer.GPT import GPTTrainer
from prompt2model.model_trainer.T5 import T5Trainer


def test_t5_trainer():
    """Test the `train_model` function of a T5Trainer.

    This function tests the T5Trainer class by training it on a small T5 model
    and verifying that the trained model is a T5ForConditionalGeneration model and
    the trained tokenizer is a T5Tokenizer model.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a T5Trainer instance with the cache directory
        trainer = T5Trainer("t5-small")
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "input_col": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        # Train the T5Trainer instance on the training dataset
        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets, {"output_dir": cache_dir}
        )

        # Verify that the trained model is a T5ForConditionalGeneration model
        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)

        # Verify that the trained tokenizer is a T5Tokenizer model
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_gpt_trainer():
    """Test the `train_model` function of a GPTTrainer.

    This function tests the GPTTrainer class by training it on a small GPT model
    and verifying that the trained model is a T5ForConditionalGeneration model and
    the trained tokenizer is a T5Tokenizer model.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a T5Trainer instance with the cache directory
        trainer = GPTTrainer("gpt2")
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "input_col": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "input_col": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                }
            ),
        ]

        # Train the T5Trainer instance on the training dataset
        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets, {"output_dir": cache_dir}
        )

        # Verify that the trained model is a GPT2LMHeadModel model
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)

        # Verify that the trained tokenizer is a PreTrainedTokenizerFast model
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
