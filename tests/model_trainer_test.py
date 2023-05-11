"""Testing T5Trainer."""

import tempfile

import datasets
import transformers

from prompt2model.model_trainer.base import ModelTrainer


def test_trainer():
    """Test the `train_model` function of a T5Trainer.

    This function tests the T5Trainer class by training it on a small T5 model
    and verifying that the trained model is a T5ForConditionalGeneration model and
    the trained tokenizer is a T5Tokenizer model.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a T5Trainer instance with the cache directory
        trainer = ModelTrainer("t5-small", has_encoder=True)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
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

    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a T5Trainer instance with the cache directory
        trainer = ModelTrainer("gpt2", has_encoder=False)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                    "output_col": ["pomme"] * 2,
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
