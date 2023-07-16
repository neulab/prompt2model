"""Testing GPT (autoregressive) ModelTrainer with different configurations."""

import os
import tempfile
from unittest.mock import patch

import datasets
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_gpt_trainer_with_get_left_padding_length():
    """Test the get_left_padding_length function of GPT Trainer."""
    trainer = GenerationModelTrainer("sshleifer/tiny-gpt2", has_encoder=False)
    test_cases = [
        ([1, 1, 1, 3, 1], 1, 3),  # There is 3 `1` in the prefix.
        ([1, 1, 1, 1], 1, 4),  # There is 4 `1` in the prefix.
        ([2, 2, 2, 2, 2], 1, 0),  # There is 0 `1` in the prefix.
        ([0, 0, 1, 1, 1], 1, 0),  # There is 0 `1` in the prefix.
    ]
    for each in test_cases:
        # GPT tokenizer uses left padding.
        assert trainer.get_left_padding_length(each[0], each[1]) == each[2]


def test_gpt_model_trainer_tokenize():
    """Test the Trainer for GPT model correctly tokenize dataset."""
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

        # We are using teaching force in training decoder-only model.
        # The end of the `model_input` is the `model_output`, only which
        # should be taken into account by the loss function.
        # length_of_output_encoding_id_without_padding is the length
        # of raw tokenized `model_output` without padding.
        length_of_output_encoding_id_without_padding = len(
            output_encoding_id
        ) - trainer.get_left_padding_length(
            output_encoding_id, trainer.model.config.pad_token_id
        )

        # The index -100 is the ignored index of cross-entropy loss.
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
        # Since we set the padding_token as eos_token, so the
        # pad_token_id should equal to eos_token_id.
        assert (
            output_encoding_id[-1]
            == label[-1]
            == input_id[-1]
            == trainer.model.config.eos_token_id
            == trainer.model.config.pad_token_id
        )
        # For GPT model, length of input_id, atattention_mask, label is the same.
        assert len(input_id) == len(attentent_mask) == len(label)
    del trainer


def test_gpt_trainer_with_tokenizer_max_length():
    """Test GPT Trainer with a specified tokenizer_max_length of 512."""
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
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n5<|endoftext|>",  # noqa 501
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nI have been using this every night for 6 weeks now. I do not see a change in my acne or blackheads. My skin is smoother and brighter. There is a glow. But that is it.\nLabel:\n3<|endoftext|>",  # noqa 501
                    ],
                    "model_output": ["5<|endoftext|>", "3<|endoftext|>"],
                }
            ),
        ]

        with patch("logging.info") as mock_info, patch(
            "logging.warning"
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
            # evaluation_strategy to no. Check if logging.info was
            # called once for not setting the evaluation strategy.
            mock_info.assert_called_once_with(
                "The traning doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logging.warning wasn't called.
            mock_warning.assert_not_called()

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
        del trainer


def test_gpt_trainer_without_tokenizer_max_length():
    """Test GPT Trainer without a specified tokenizer_max_length."""
    # Test auto-regressive GenerationModelTrainer implementation
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
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nBeen using for a week and have noticed a huuge difference.\nLabel:\n5<|endoftext|>",  # noqa 501
                        "<task 1>Given a product review, predict the sentiment score associated with it.\nExample:\nI have been using this every night for 6 weeks now. I do not see a change in my acne or blackheads. My skin is smoother and brighter. There is a glow. But that is it.\nLabel:\n3<|endoftext|>",  # noqa 501
                    ],
                    "model_output": ["5<|endoftext|>", "3<|endoftext|>"],
                }
            ),
        ]
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
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
            # evaluation_strategy to no. Check if logging.info was
            # called once for not setting the evaluation strategy.
            mock_info.assert_called_once_with(
                "The traning doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )

            # Check if logging.warning was called once for
            # not setting the tokenizer_max_length.
            mock_warning.assert_called_once_with(
                "Set the tokenizer_max_length is preferable for finetuning model, which saves the cost of training."  # noqa 501
            )

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
        del trainer


def test_gpt_trainer_with_epoch_evaluation():
    """Test GPT Trainer with validation datsets for epoch evaluation."""
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

            # Check if logging.warning was not called.
            mock_warning.assert_not_called()

        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
        del trainer
