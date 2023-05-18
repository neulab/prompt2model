"""Testing GenerationModelTrainer with different configurations."""

import os
import tempfile

import datasets
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer():
    """Test the `GenerationModelTrainer` class to train a encoder-decoder model."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("t5-small", has_encoder=True)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets,
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
        )

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_gpt_trainer():
    """Test the `GenerationModelTrainer` to train an autoregressive model."""
    # Test decoder-only GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("gpt2", has_encoder=False)
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
                    "model_input": ["translate English to French."] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets,
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
        )

        assert isinstance(trained_model, transformers.GPT2LMHeadModel)

        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
