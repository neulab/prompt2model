"""Testing GPT (autoregressive) ModelTrainer with different configurations."""

import gc
import logging
import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import torch.nn as nn
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"
loss_function = nn.CrossEntropyLoss()
IGNORE_INDEX = loss_function.ignore_index
logger = logging.getLogger("ModelTrainer")


def test_gpt_trainer_with_get_left_padding_length():
    """Test the get_left_padding_length function of the GPT Trainer."""
    trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
    test_cases = [
        ([1, 1, 1, 3, 1], 1, 3),  # There is 3 `1` in the prefix.
        ([1, 1, 1, 1], 1, 4),  # There is 4 `1` in the prefix.
        ([2, 2, 2, 2, 2], 1, 0),  # There is 0 `1` in the prefix.
        ([0, 0, 1, 1, 1], 1, 0),  # There is 0 `1` in the prefix.
    ]
    for each in test_cases:
        # The GPT tokenizer uses left padding.
        assert trainer.get_left_padding_length(each[0], each[1]) == each[2]
    gc.collect()


def test_gpt_model_trainer_tokenize():
    """Test that the Trainer for GPT model correctly tokenizes a dataset."""
    trainer = GenerationModelTrainer(
        "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=64
    )
    training_dataset = datasets.Dataset.from_dict(
        # The eos_token of GPT2 is <|endoftext|>.
        # The GPT tokenizer do not automatically add eos_token
        # in the end of sequence. So we need to manually
        # add eos_token in the end.
        {
            "model_input": [
                "<task 0>convert to text2text\nExample:\nfoo\nLabel:\nbaz<|endoftext|>",  # noqa: E501
                "<task 0>convert to text2text\nExample:\nfoo foo foo foo\nLabel:\nbaz baz baz baz<|endoftext|>",  # noqa: E501
                "<task 0>convert to text2text\nExample:\nfoo foo\nLabel:\nbaz baz<|endoftext|>",  # noqa: E501
            ],
            "model_output": [
                "baz<|endoftext|>",
                "baz baz baz baz<|endoftext|>",
                "baz baz<|endoftext|>",
            ],
        }
    )
    tokenized_dataset = trainer.tokenize_dataset(training_dataset, shuffle=False)

    output_encodings = trainer.tokenizer.batch_encode_plus(
        training_dataset["model_output"],
        truncation=True,
        max_length=trainer.tokenizer_max_length,
        padding=True,
    )

    for idx, input_id in enumerate(tokenized_dataset["input_ids"]):
        attentent_mask = tokenized_dataset["attention_mask"][idx]
        label = tokenized_dataset["labels"][idx]
        output_encoding_id = output_encodings["input_ids"][idx]
        # Test the length of input_id is the same as attention_mask.
        assert len(input_id) == len(attentent_mask)
        # Test that each pad_token in input_id corresponds to
        # a 0 in attention_mask.
        assert trainer.get_left_padding_length(
            input_id, trainer.model.config.pad_token_id
        ) == trainer.get_left_padding_length(attentent_mask, 0)
        # Test that the last token of input_id is an eos_token.
        assert input_id[-1] == trainer.model.config.eos_token_id

        # The end of the `model_input` is the `model_output`, only which
        # should be taken into account by the loss function.
        # length_of_output_encoding_id_without_padding is the length
        # of raw tokenized `model_output` without padding.
        length_of_output_encoding_id_without_padding = len(
            output_encoding_id
        ) - trainer.get_left_padding_length(
            output_encoding_id, trainer.model.config.pad_token_id
        )

        # The IGNORE_INDEX is the ignored index of cross-entropy
        # loss. length_of_compute_loss_label is the length of labels
        # that are taken into account by the loss function.
        assert IGNORE_INDEX == -100
        length_of_compute_loss_label = len(label) - trainer.get_left_padding_length(
            label, IGNORE_INDEX
        )

        # So length_of_output_encoding_id_without_padding
        # should be equal to length_of_compute_loss_label.

        assert (
            length_of_compute_loss_label == length_of_output_encoding_id_without_padding
        )
        # The tail of the label should be exactly the same as
        # the raw tokenized `model_output` without padding
        # and the same as the tail of `input_id`.
        assert (
            label[-length_of_compute_loss_label:]
            == output_encoding_id[-length_of_output_encoding_id_without_padding:]
            == input_id[-length_of_output_encoding_id_without_padding:]
        )
        # The end of the `model_input` is the `model_output`. And the end of
        # `model_output` is the eos_token. So the last token of input_id
        #  and output_encoding_id should both be the eos_token.
        # Since we set the padding_token as eos_token, the
        # pad_token_id should equal to eos_token_id.
        assert (
            output_encoding_id[-1]
            == label[-1]
            == input_id[-1]
            == trainer.model.config.eos_token_id
            == trainer.model.config.pad_token_id
        )
        # For the GPT model, len(input_id) = len(atattention_mask) = len(label).
        assert len(input_id) == len(attentent_mask) == len(label)
    gc.collect()


def test_gpt_trainer_with_tokenizer_max_length():
    """Test GPT Trainer with a specified tokenizer_max_length of 512."""
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n5<|endoftext|>",  # noqa E501
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nI have been using this every night for 6 weeks now. I do not see a change in my acne or blackheads. My skin is smoother and brighter. There is a glow. But that is it.\nLabel:\n3<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["5<|endoftext|>", "3<|endoftext|>"],
                }
            ),
        ]

        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            trainer = GenerationModelTrainer(
                "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=512
            )
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 2,
                    "evaluation_strategy": "no",
                },
                training_datasets,
            )
            # Though we did not pass in validation dataset, we set
            # evaluation_strategy to `no`. Check if logger.info was
            # called once for not setting the evaluation strategy.
            mock_info.assert_called_once_with(
                "The trainer doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logger.warning wasn't called.
            mock_warning.assert_not_called()

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
    gc.collect()


def test_gpt_trainer_without_tokenizer_max_length():
    """Test GPT Trainer without a specified tokenizer_max_length."""
    # Test the autoregressive GenerationModelTrainer implementation.
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n5<|endoftext|>",  # noqa E501
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nI have been using this every night for 6 weeks now. I do not see a change in my acne or blackheads. My skin is smoother and brighter. There is a glow. But that is it.\nLabel:\n3<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["5<|endoftext|>", "3<|endoftext|>"],
                }
            ),
        ]
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            num_train_epochs = 2
            trainer = GenerationModelTrainer(
                "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=None
            )
            trained_model, trained_tokenizer = trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 2,
                    "evaluation_strategy": "no",
                },
                training_datasets,
            )

            # Though we did not pass in validation dataset, we set
            # evaluation_strategy to `no`. Check if logger.info was
            # called once for not setting the evaluation strategy.
            mock_info.assert_called_once_with(
                "The trainer doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logger.warning was called once for
            # not setting the tokenizer_max_length.
            mock_warning.assert_called_once_with(
                "Set the tokenizer_max_length is preferable for finetuning model, which saves the cost of training."  # noqa E501
            )

            trained_model.save_pretrained(cache_dir)
            trained_tokenizer.save_pretrained(cache_dir)
            assert isinstance(trained_model, transformers.GPT2LMHeadModel)
            assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
    gc.collect()


def test_gpt_trainer_with_epoch_evaluation():
    """Test GPT Trainer with validation datsets for epoch evaluation."""
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nBroke me out and gave me awful texture all over my face. I typically have clear skin and after using this product my skin HATED it. Could work for you though.\nLabel:\n",  # noqa E501
                    ],
                    "model_output": ["2<|endoftext|>"],
                }
            ),
        ]

        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning, patch.object(
            logging.getLogger("ModelEvaluator"), "info"
        ) as mock_evaluator_info:
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
                    "evaluation_strategy": "epoch",
                },
                training_datasets,
                validation_datasets,
            )
            # Check if logger.info was called correctly.
            # Eech epoch will log 3 times, twice in `on_epoch_end`
            # and once in `evaluate_model`.
            assert mock_info.call_count == 2 * num_train_epochs
            assert mock_evaluator_info.call_count == 1 * num_train_epochs
            info_list = [each.args[0] for each in mock_evaluator_info.call_args_list]
            assert (
                info_list.count(
                    "Using default metrics of chr_f, exact_match and bert_score."
                )
                == num_train_epochs
            )
            # The other two kind of logger.info in `on_epoch_end` of
            # `ValidationCallback`are logging the epoch num wtih the
            # val_dataset_size and logging the `metric_values`.

            assert trainer.validation_callback.epoch_count == num_train_epochs
            assert (
                trainer.validation_callback.val_dataset_size == len(validation_datasets)
                and len(validation_datasets) != 0
            )

            # Check if logger.warning was not called.
            mock_warning.assert_not_called()

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
    gc.collect()


def test_gpt_trainer_without_validation_datasets():
    """Test GPT Trainer without validation datsets for epoch evaluation."""
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
        ]

        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
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
            # We set the evaluation strategy to epoch but don't pass
            # in the validation dataset. So the evaluation will be skipped.
            # Check if logger.info wasn't called.
            mock_info.assert_not_called()

            # Check if logger.warning was called once
            mock_warning.assert_called_once_with(
                "The validation split for autoregressive model is missing, which should not contain labels as the training spilt. Thus this evaluation will be skipped."  # noqa E501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
    gc.collect()


def test_gpt_trainer_with_unsupported_evaluation_strategy():
    """Test GPT Trainer with unsupported evaluation_strategy."""
    # We only support `epoch` as evaluation_strategy, so `step` strategy is unsupported.
    with tempfile.TemporaryDirectory() as cache_dir:
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
                    ],
                    "model_output": ["4<|endoftext|>"],
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nBroke me out and gave me awful texture all over my face. I typically have clear skin and after using this product my skin HATED it. Could work for you though.\nLabel:\n",  # noqa E501
                    ],
                    "model_output": ["2<|endoftext|>"],
                }
            ),
        ]

        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning, patch.object(
            logging.getLogger("ModelEvaluator"), "info"
        ) as mock_evaluator_info:
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

            # Check if logger.info was called correctly.
            # Eech epoch will log 3 times, twice in `on_epoch_end`
            # and once in `evaluate_model`.
            assert mock_info.call_count == 2 * num_train_epochs
            assert mock_evaluator_info.call_count == 1 * num_train_epochs
            info_list = [each.args[0] for each in mock_evaluator_info.call_args_list]
            assert (
                info_list.count(
                    "Using default metrics of chr_f, exact_match and bert_score."
                )
                == num_train_epochs
            )
            # The other two kind of logger.info in `on_epoch_end` of
            # `ValidationCallback`are logging the epoch num wtih the
            # val_dataset_size and logging the `metric_values`.

            assert trainer.validation_callback.epoch_count == num_train_epochs
            assert (
                trainer.validation_callback.val_dataset_size == len(validation_datasets)
                and len(validation_datasets) != 0
            )

            # Check if logger.warning was called once.
            # Since we don't support step evaluation_strategy,
            # so the evaluation  will be changed to epoch.
            mock_warning.assert_called_once_with(
                "Only `epoch` evaluation strategy is supported, the evaluation strategy will be set to evaluate_after_epoch."  # noqa E501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
    gc.collect()


def test_gpt_trainer_with_unsupported_parameter():
    """Test the error handler with an unsupported hyperparameter with GPT Trainer."""
    # In this test case we provide an unsupported parameter called `batch_size` to
    # `trainer.train_model`. The supported parameter is `per_device_train_batch_size`.
    with pytest.raises(ValueError) as exc_info:
        with tempfile.TemporaryDirectory() as cache_dir:
            trainer = GenerationModelTrainer(
                "sshleifer/tiny-gpt2",
                has_encoder=False,
            )
            training_datasets = [
                datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n4<|endoftext|>",  # noqa E501
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
    gc.collect()


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
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logger.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. Consider increasing the tokenizer_max_length if possible. Otherwise, truncation may lead to unexpected results."  # noqa: E501
        )
        mock_info.assert_not_called()
    gc.collect()
