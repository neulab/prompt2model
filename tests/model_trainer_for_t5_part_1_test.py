"""Testing T5 (encoder-decoder) ModelTrainer with different configurations (part 2)."""

import os
import tempfile
from unittest.mock import patch

import datasets

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer_with_get_right_padding_length():
    """Test the get_right_padding_length function of T5 Trainer."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True
    )
    test_cases = [
        ([1, 1, 1, 3, 1], 1, 1),  # There is 1 `1` in the suffix.
        ([1, 1, 1, 1], 1, 4),  # There is 4 `1` in the suffix.
        ([2, 2, 2, 2, 2], 1, 0),  # There is 0 `1` in the suffix.
        ([0, 0, 1, 1, 1], 1, 3),  # There is 3 `1` in the suffix.
    ]
    for each in test_cases:
        # T5 tokenizer uses right padding.
        assert trainer.get_right_padding_length(each[0], each[1]) == each[2]


def test_t5_trainer_tokenize():
    """Test the Trainer for T5 mode correctly tokenize dataset."""
    trainer = GenerationModelTrainer(
        "patrickvonplaten/t5-tiny-random", has_encoder=True, tokenizer_max_length=64
    )
    training_dataset = datasets.Dataset.from_dict(
        # The T5 tokenizer automatically adds eos_token
        # in the end of sequence. So we don't need to
        # manually add eos_token in the end.
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

        length_of_label_without_padding = len(label) - length_of_right_padding_in_label
        length_of_output_encoding_id_without_padding = (
            len(output_encoding_id) - length_of_right_padding_in_input_id
        )
        # T5 tokenizer automatically adds eos token in the end of sequence.
        # Test the last token of label and output_encoding_id (except the pad_token)
        # should both be the eos_token. And for T5 model, eos_token is not pad_token.
        assert (
            label[length_of_label_without_padding - 1]
            == output_encoding_id[length_of_output_encoding_id_without_padding - 1]
            == trainer.model.config.eos_token_id
            != trainer.model.config.pad_token_id
        )

        # Test that label without right padding is the same as
        # output_encoding_id without right padding.
        assert (
            label[:length_of_label_without_padding]
            == output_encoding_id[:length_of_output_encoding_id_without_padding]
        )
    gc.collect()


def test_t5_trainer_with_tokenizer_max_length():
    """Test T5 Trainer with a specified tokenizer_max_length of 512."""
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
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nI have been using this every night for 6 weeks now. I do not see a change in my acne or blackheads. My skin is smoother and brighter. There is a glow. But that is it.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["3"],
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

            trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "no",
                },
                training_datasets,
            )

            # Though we did not pass in validation dataset, we set
            # evaluation_strategy to no. Check if logging.info was
            # called once for not setting the evaluation strategy.
            mock_info.assert_called_once_with(
                "The traning doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logging.warning wasn't called.
            mock_warning.assert_not_called()
        gc.collect()


def test_t5_trainer_without_tokenizer_max_length():
    """Train a encoder-decoder model without a specified tokenizer_max_length ."""
    # Test encoder-decoder GenerationModelTrainer implementation
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
                tokenizer_max_length=None,
            )
            num_train_epochs = 1
            trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "no",
                },
                training_datasets,
            )
            mock_info.assert_called_once_with(
                "The traning doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logging.warning was called once for
            # not setting the tokenizer_max_length.
            mock_warning.assert_called_once_with(
                "Set the tokenizer_max_length is preferable for finetuning model, which saves the cost of training."  # noqa 501
            )
        gc.collect()


def test_t5_trainer_with_epoch_evaluation():
    """Test T5 Trainer with validation datsets for epoch evaluation."""
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
        ]

        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nBroke me out and gave me awful texture all over my face. I typically have clear skin and after using this product my skin HATED it. Could work for you though.\nLabel:\n",  # noqa 501
                    ],
                    "model_output": ["2"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            trainer = GenerationModelTrainer(
                "patrickvonplaten/t5-tiny-random",
                has_encoder=True,
            )
            num_train_epochs = 1
            trainer.train_model(
                {
                    "output_dir": cache_dir,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": 1,
                    "evaluation_strategy": "epoch",
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

            mock_warning.assert_not_called()
        gc.collect()
