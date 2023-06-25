"""Testing GenerationModelTrainer with different configurations."""

import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer_with_model_max_length():
    """Train a encoder-decoder model with a specified model_max_length of 512 ."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "t5-small", has_encoder=True, model_max_length=512
        )
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            {
                "output_dir": cache_dir,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
            },
            training_datasets,
        )

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_t5_trainer_without_model_max_length():
    """Train a encoder-decoder model without a specified model_max_length ."""
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

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
            )
            # Check if logging.info was called six times
            # Eech epoch will log 2 times, in `on_epoch_end` and `evaluate_model`
            assert mock_info.call_count == 2 * 3
            # Check if logging.warning was called once
            assert mock_warning.call_count == 1

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_t5_trainer_with_unsupported_evaluation_strategy():
    """Train a T5 model with unsupported evaluation_strategy."""
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "t5-small", has_encoder=True, model_max_length=512
        )
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "step",
                },
                training_datasets,
                validation_datasets,
            )

            # Check if logging.info was called three times
            # Eech epoch will log 2 times, in `on_epoch_end` and `evaluate_model`
            assert mock_info.call_count == 6

            # Check if logging.warning was called once
            assert mock_warning.call_count == 1

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_gpt_trainer_without_validation_datasets():
    """Train an autoregressive model without validation datasets."""
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
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
            )
            # Check if logging.info wasn't called
            assert mock_info.call_count == 0
            # Check if logging.warning was called once
            assert mock_warning.call_count == 1

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_with_validation_datasets():
    """Train an autoregressive model with validation datasets."""
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
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]
        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
                validation_datasets,
            )
            # Check if logging.info was called four times
            # Eech epoch will log 2 times, in `on_epoch_end` and `evaluate_model`
            assert mock_info.call_count == 4
            # logging.warning wasn't called.
            assert mock_warning.call_count == 0

        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_trainer_with_unsupported_parameter():
    """Test the error handler with an unsupported hyperparameter."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with pytest.raises(AssertionError):
        with tempfile.TemporaryDirectory() as cache_dir:
            trainer = GenerationModelTrainer(
                "t5-small", has_encoder=True, model_max_length=512
            )
            training_datasets = [
                datasets.Dataset.from_dict(
                    {
                        "model_input": ["translate apple to french"] * 2,
                        "output_col": ["pomme"] * 2,
                    }
                ),
            ]

            trainer.train_model(
                {"output_dir": cache_dir, "train_epochs": 1, "batch_size": 1},
                training_datasets,
            )
