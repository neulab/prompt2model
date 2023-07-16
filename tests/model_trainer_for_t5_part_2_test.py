"""Testing T5 (encoder-decoder) ModelTrainer with different configurations (part 1)."""

import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import transformers
from datasets import concatenate_datasets

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer_without_validation_datasets():
    """Test T5 Trainer without validation datsets for epoch evaluation."""
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["4"],
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["5"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trainer = GenerationModelTrainer(
                "patrickvonplaten/t5-tiny-random",
                has_encoder=True,
                tokenizer_max_length=128,
            )
            num_train_epochs = 1
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
            )
            # Check if logging.info was called correctly.
            # Eech epoch will log 3 times, twice in `on_epoch_end`
            # and once in `evaluate_model`.
            assert mock_info.call_count == 3 * num_train_epochs
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert (
                info_list.count(
                    "Using default metrics of chrf, exact_match and bert_score."
                )
                == num_train_epochs
            )
            # The other two kind of logging.info in `on_epoch_end` of
            # `ValidationCallback`are logging the epoch num wtih the
            # val_dataset_size and logging the `metric_values`.

            assert trainer.validation_callback.epoch_count == num_train_epochs

            concatenated_training_dataset = concatenate_datasets(training_datasets)
            splited_dataset = concatenated_training_dataset.train_test_split(
                test_size=0.15, seed=trainer.training_seed
            )
            val_dataset = splited_dataset["test"]
            assert trainer.validation_callback.val_dataset is not None
            assert len(trainer.validation_callback.val_dataset.features) == 2
            assert (
                trainer.validation_callback.val_dataset["model_input"]
                == val_dataset["model_input"]
            )
            assert (
                trainer.validation_callback.val_dataset["model_output"]
                == val_dataset["model_output"]
            )

            # The evaluation_strategy is set to epoch, but validation
            # datasets are not provided. So the training dataset will
            # be splited to generate validation dataset.
            mock_warning.assert_called_once_with(
                "The validation split for encoder-decoder model is missed. The training dataset will be split to create the validation dataset."  # noqa E501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)
        del trainer


def test_t5_trainer_with_unsupported_evaluation_strategy():
    """Train a T5 model with unsupported evaluation_strategy."""
    # We only support `epoch` as evaluation_strategy, so `step` strategy is unsupported.
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "patrickvonplaten/t5-tiny-random",
            has_encoder=True,
            tokenizer_max_length=128,
        )
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["4"],
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["5"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            num_train_epochs = 1
            trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "step",
                },
                training_datasets,
                validation_datasets,
            )

            # Check if logging.info was called correctly.
            # Eech epoch will log 3 times, in `on_epoch_end`, `evaluate_model`
            assert mock_info.call_count == 3 * num_train_epochs
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert (
                info_list.count(
                    "Using default metrics of chrf, exact_match and bert_score."
                )
                == num_train_epochs
            )
            # The other two kind of logging.info in `on_epoch_end` of
            # `ValidationCallback`are logging the epoch num wtih the
            # val_dataset_size and logging the `metric_values`.

            assert trainer.validation_callback.epoch_count == num_train_epochs
            assert (
                trainer.validation_callback.val_dataset_size
                == len(validation_datasets)
                != 0
            )

            # Check if logging.warning was called once
            mock_warning.assert_called_once_with(
                "Only `epoch` evaluation strategy is supported, the evaluation strategy will be set to evaluate_after_epoch."  # noqa E501
            )
        del trainer


def test_t5_trainer_with_unsupported_parameter():
    """Test the error handler with an unsupported hyperparameter with T5 Trainer."""
    # We actually support per_device_train_batch_size, but the testcase uses batch_size.
    with pytest.raises(AssertionError) as exc_info:
        with tempfile.TemporaryDirectory() as cache_dir:
            trainer = GenerationModelTrainer(
                "patrickvonplaten/t5-tiny-random",
                has_encoder=True,
                tokenizer_max_length=128,
            )
            training_datasets = [
                datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n",  # noqa 501
                        ],
                        "model_output": ["4"],
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
        del trainer


def test_t5_trainer_with_truncation_warning():
    """Test T5 Trainer correctly raised truncation warning when tokenizing."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True, tokenizer_max_length=32
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
    del trainer
