"""Testing GenerationModelTrainer with different configurations."""

import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer_with_tokenizer_max_length():
    """Train a encoder-decoder model with a specified tokenizer_max_length of 512 ."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "patrickvonplaten/t5-tiny-random",
            has_encoder=True,
            tokenizer_max_length=512,
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


def test_gpt_trainer_with_tokenizer_max_length():
    """Train a auto-regressive model with a specified tokenizer_max_length of 512 ."""
    # Test auto-regressive GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=512
        )
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
            mock_warning.assert_called_once_with(
                "The validation split for autoregressive model is missed, which should not contain labels as the training spilt. Thus this evaluation will be skipped."  # noqa 501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_without_tokenizer_max_length():
    """Train a auto-regressive model without a specified tokenizer_max_length."""
    # Test auto-regressive GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
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

        trained_model, trained_tokenizer = trainer.train_model(
            {
                "output_dir": cache_dir,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
            },
            training_datasets,
        )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_t5_trainer_without_tokenizer_max_length():
    """Train a encoder-decoder model without a specified tokenizer_max_length ."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
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
            trainer = GenerationModelTrainer(
                "patrickvonplaten/t5-tiny-random",
                has_encoder=True,
                tokenizer_max_length=None,
            )
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
            )
            # Check if logging.info was called six times
            # Eech epoch will log 3 times, in `on_epoch_end`, `evaluate_model`
            assert mock_info.call_count == 3 * 2
            # Check if logging.warning was called for not having a tokenizer_max_length
            # and not having an validation dataset.
            assert mock_warning.call_count == 2

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_t5_trainer_with_unsupported_evaluation_strategy():
    """Train a T5 model with unsupported evaluation_strategy."""
    # We only support `epoch` as evaluation_strategy, so `step` strategy is unsupported.
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "patrickvonplaten/t5-tiny-random",
            has_encoder=True,
            tokenizer_max_length=512,
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
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "step",
                },
                training_datasets,
                validation_datasets,
            )

            # Check if logging.info was called three times
            # Eech epoch will log 3 times, in `on_epoch_end`, `evaluate_model`
            assert mock_info.call_count == 3 * 2

            # Check if logging.warning was called once
            assert mock_warning.call_count == 1

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_gpt_trainer_without_validation_datasets():
    """Train an autoregressive model without validation datasets."""
    # Test decoder-only GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
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
            # for not having a validation dataset.
            assert mock_warning.call_count == 1

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_with_validation_datasets():
    """Train an autoregressive model with validation datasets."""
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
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
            # Eech epoch will log 3 times, in `on_epoch_end` and `evaluate_model`
            assert mock_info.call_count == 3 * 2
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
                "patrickvonplaten/t5-tiny-random",
                has_encoder=True,
                tokenizer_max_length=512,
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


def test_truncation_warning_for_gpt_trainer():
    """Test the warning for GPT2 model trainer is correct raised when tokenizing."""
    trainer = GenerationModelTrainer(
        "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=32
    )
    training_dataset = datasets.Dataset.from_dict(
        {
            "model_input": [
                "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
            ]
            * 2,
            "output_col": ["pomme"] * 2,
        }
    )
    with patch("logging.warning") as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logging.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )


def test_truncation_warning_for_t5_trainer():
    """Test the warning for T5 model trainer is correct raised when tokenizing."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True, tokenizer_max_length=32
    )
    training_dataset = datasets.Dataset.from_dict(
        {
            "model_input": [
                "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
            ]
            * 2,
            "output_col": ["pomme"] * 2,
        }
    )
    with patch("logging.warning") as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logging.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )
