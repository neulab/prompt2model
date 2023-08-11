"""Testing GenerationModelTrainer with different configurations."""

import os
import tempfile
from unittest.mock import patch

import datasets
import pytest
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_get_left_padding_length():
    """Test the get_left_padding_length function."""
    trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
    test_cases = [
        ([1, 1, 1, 3, 1], 1, 3),
        ([1, 1, 1, 1], 1, 4),
        ([2, 2, 2, 2, 2], 1, 0),
        ([0, 0, 1, 1, 1], 1, 0),
    ]
    for each in test_cases:
        # GPT tokenizer uses left padding.
        assert trainer.get_left_padding_length(each[0], each[1]) == each[2]


def test_get_right_padding_length():
    """Test the get_right_padding_length function."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True
    )
    test_cases = [
        ([1, 1, 1, 3, 1], 1, 1),
        ([1, 1, 1, 1], 1, 4),
        ([2, 2, 2, 2, 2], 1, 0),
        ([0, 0, 1, 1, 1], 1, 3),
    ]
    for each in test_cases:
        # T5 tokenizer uses right padding.
        assert trainer.get_right_padding_length(each[0], each[1]) == each[2]


def test_gpt_model_trainer_tokenize():
    """Test the Trainer for GPT model give correct tokenization."""
    trainer = GenerationModelTrainer(
        "sshleifer/tiny-gpt2", has_encoder=False, tokenizer_max_length=64
    )
    training_dataset = datasets.Dataset.from_dict(
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
        attention_mask = tokenized_dataset["attention_mask"][idx]
        label = tokenized_dataset["labels"][idx]
        output_encoding_id = output_encodings["input_ids"][idx]
        # Test that each pad_token in input_id list corresponds to
        # a 0 in attention_mask.
        assert trainer.get_left_padding_length(
            input_id, trainer.model.config.pad_token_id
        ) == trainer.get_left_padding_length(attention_mask, 0)
        # Test that the last token of input_id is a eos_token.
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

        # The index -100 is ignored for the loss compute in Autoregressive model.
        # length_of_compute_loss_label is the length of labels that
        # are taken into account by the loss function.
        length_of_compute_loss_label = len(label) - trainer.get_left_padding_length(
            label, -100
        )

        # So length_of_output_encoding_id_without_padding
        # should be equal to length_of_compute_loss_label.

        assert (
            length_of_compute_loss_label == length_of_output_encoding_id_without_padding
        )
        # The tail of the label should be exactly the same as
        # the raw tokenized `model_output` without padding.
        assert (
            label[-length_of_compute_loss_label:]
            == output_encoding_id[-length_of_output_encoding_id_without_padding:]
            == input_id[-length_of_output_encoding_id_without_padding:]
        )
        # The end of the `model_input` is the `model_output`. And the end of
        # `model_output` is the eos_token. So the last token of input_id
        #  and output_encoding_id should both be the eos_token.
        assert (
            output_encoding_id[-1]
            == label[-1]
            == input_id[-1]
            == trainer.model.config.eos_token_id
        )
        # For GPT model, length of input_id, atattention_mask, label is the same.
        assert len(input_id) == len(attention_mask) == len(label)


def test_t5_model_trainer_tokenize():
    """Test the Trainer for T5 model give correct tokenization."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True, tokenizer_max_length=64
    )
    training_dataset = datasets.Dataset.from_dict(
        {
            "model_input": [
                "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",  # noqa: E501
                "<task 0>convert to text2text\nExample:\nfoo foo foo foo\nLabel:\n",  # noqa: E501
                "<task 0>convert to text2text\nExample:\nfoo foo\nLabel:\n",  # noqa: E501
            ],
            "model_output": [
                "baz",
                "baz baz baz baz",
                "baz baz",
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

    # For T5 model, the label of tokenized_dataset is the modified input_id
    # of output_encodings, where all the padding tokens are replaced by -100.
    modified_labels = [
        [
            -100 if element == trainer.tokenizer.pad_token_id else element
            for element in sublist
        ]
        for sublist in output_encodings["input_ids"]
    ]
    assert tokenized_dataset["labels"] == modified_labels
    # For T5 model，length of input_ids is the same as attention_mask.
    for idx, input_id in enumerate(tokenized_dataset["input_ids"]):
        label = tokenized_dataset["labels"][idx]
        attention_mask = tokenized_dataset["attention_mask"][idx]
        output_encoding_id = output_encodings["input_ids"][idx]

        # Test that the length of input_id is the same as attention_mask.
        assert len(input_id) == len(attention_mask)
        # Test each pad_token in input_id corresponds to a 0 in attention_mask.
        assert trainer.get_left_padding_length(
            input_id, trainer.model.config.pad_token_id
        ) == trainer.get_left_padding_length(attention_mask, 0)
        # The length of right padding tokens in output_encoding_id
        # equals to the length of right padding -100 of label.
        length_of_right_padding_in_label = trainer.get_right_padding_length(label, -100)
        length_of_right_padding_in_input_id = trainer.get_right_padding_length(
            output_encoding_id, trainer.tokenizer.pad_token_id
        )
        assert length_of_right_padding_in_label == length_of_right_padding_in_input_id
        # Test the last token of label and output_encoding_id (except the pad_token)
        # should both be the eos_token.
        length_of_label_without_padding = len(label) - length_of_right_padding_in_label
        length_of_output_encoding_id_without_padding = (
            len(output_encoding_id) - length_of_right_padding_in_input_id
        )
        assert (
            label[length_of_label_without_padding - 1]
            == output_encoding_id[length_of_output_encoding_id_without_padding - 1]
            == trainer.model.config.eos_token_id
        )
        # Test that label without right padding is the same as
        # output_encoding_id without right padding.
        assert (
            label[:length_of_label_without_padding]
            == output_encoding_id[:length_of_output_encoding_id_without_padding]
        )


def test_t5_trainer_with_tokenizer_max_length():
    """Train a encoder-decoder model with a specified tokenizer_max_length of 512."""
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
                    "model_input": [
                        "<task 0>I am learning French.\nExample:\ntranslate apple to french\nLabel:\n"  # noqa: E501
                    ]
                    * 2,
                    "model_output": ["pomme"] * 2,
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
                    "model_output": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "model_output": ["pomme", "pome"],
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
                    "model_output": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "model_output": ["pomme", "pome"],
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
                    "model_output": ["pomme"] * 2,
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
                    "model_output": ["pomme"] * 2,
                }
            ),
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "model_output": ["pomme"] * 2,
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
                    "model_output": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "model_output": ["pomme", "pome"],
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "model_output": ["pomme", "pome"],
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
                    "model_output": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "model_output": ["pomme", "pome"],
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
                    "model_output": ["pomme", "pome"],
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
                        "model_output": ["pomme"] * 2,
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
            "model_output": ["pomme"] * 2,
        }
    )
    with patch("logging.warning") as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logging.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. Consider increasing the tokenizer_max_length if possible. Otherwise, truncation may lead to unexpected results."  # noqa: E501
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
            "model_output": ["pomme"] * 2,
        }
    )
    with patch("logging.warning") as mock_warning:
        trainer.tokenize_dataset(training_dataset)
        # logging.warning was called for truncation.
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset. Consider increasing the tokenizer_max_length if possible. Otherwise, truncation may lead to unexpected results."  # noqa: E501
        )
