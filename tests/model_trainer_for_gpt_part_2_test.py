"""Testing GPT (autoregressive) ModelTrainer with different configurations."""

import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_gpt_trainer_without_validation_datasets():
    """Test GPT Trainer without validation datsets for epoch evaluation."""
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa 501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
            num_train_epochs = 2
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 2,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
            )
            # We set hte evaluation strategy to epoch but don't pass
            # in the validation dataset. So the evaluation will be skipped.
            # Check if logging.info wasn't called.
            mock_info.assert_not_called()

            # Check if logging.warning was called once
            mock_warning.assert_called_once_with(
                "The validation split for autoregressive model is missed, which should not contain labels as the training spilt. Thus this evaluation will be skipped."  # noqa 501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_with_unsupported_evaluation_strategy():
    """Test GPT Trainer with unsupported evaluation_strategy."""
    # We only support `epoch` as evaluation_strategy, so `step` strategy is unsupported.
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa 501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nBroke me out and gave me awful texture all over my face. I typically have clear skin and after using this product my skin HATED it. Could work for you though.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["2<|endoftext|>"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trainer = GenerationModelTrainer(
                "sshleifer/tiny-gpt2",
                has_encoder=False,
            )
            num_train_epochs = 2
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 2,
                    "evaluation_strategy": "step",
                },
                training_datasets,
                validation_datasets,
            )

            # Check if logging.info was called correctly.
            # Eech epoch will log 3 times, twice in `on_epoch_end`
            # and once in `evaluate_model`.
            assert mock_info.call_count == 3 * num_train_epochs
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert (
                info_list.count("Conduct evaluation after each epoch ends.")
                == info_list.count(
                    "Using default metrics of chrf, exact_match and bert_score."
                )
                == num_train_epochs
            )
            # The other logging.info is the `metric_values` in `evaluate_model`.

            # Check if logging.warning was called once.
            # Since we don't support step evaluation_strategy,
            # so the evaluation  will be changed to epoch.
            mock_warning.assert_called_once_with(
                "Only `epoch` evaluation strategy is supported, the evaluation strategy will be set to evaluate_after_epoch."  # noqa E501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_with_unsupported_parameter():
    """Test the error handler with an unsupported hyperparameter with GPT Trainer."""
    # We actually support per_device_train_batch_size, but the testcase uses batch_size.
    with pytest.raises(AssertionError) as exc_info:
        with tempfile.TemporaryDirectory() as cache_dir:
            trainer = GenerationModelTrainer(
                "sshleifer/tiny-gpt2",
                has_encoder=False,
            )
            training_datasets = [
                datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa 501
                        ],
                        "model_output": ["4<|endoftext|>"],
                    }
                ),
            ]

            trainer.train_model(
                {"output_dir": cache_dir, "train_epochs": 1, "batch_size": 1},
                training_datasets,
            )

        supported_keys = {
            "output_dir",
            "logging_steps",
            "evaluation_strategy",
            "save_strategy",
            "num_train_epochs",
            "per_device_train_batch_size",
            "warmup_steps",
            "weight_decay",
            "logging_dir",
            "learning_rate",
            "test_size",
        }

        assert str(exc_info.value) == (
            f"Only support {supported_keys} as training parameters."
        )


def test_gpt_trainer_with_truncation_warning():
    """Test GPT Trainer correctly raised truncation warning when tokenizing."""
    trainer = GenerationModelTrainer(
        "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=32
    )
    training_dataset = datasets.Dataset.from_dict(
        {
            "model_input": [
                "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
            ]
            * 2,
            "model_output": ["pomme"] * 2,
        }
    )
    with patch("logging.info") as mock_info, patch("logging.warning") as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logging.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )
        mock_info.assert_not_called()
