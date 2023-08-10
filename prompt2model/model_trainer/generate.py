"""A trainer class to train generation models."""

from __future__ import annotations  # noqa FI58

import logging
import os
from itertools import takewhile
from typing import Any

import datasets
import torch
import transformers
from datasets import concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.model_trainer.callback import ValidationCallback
from prompt2model.utils import seed_generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)


class GenerationModelTrainer(BaseTrainer):
    """Trainer for T5 type (encoder-decoder) model and GPT type (deocder-only) model."""

    def __init__(
        self,
        pretrained_model_name: str,
        has_encoder: bool,
        tokenizer_max_length: int | None = 512,
    ):
        """Initializes a new instance of GenerationModelTrainer.

        Args:
            pretrained_model_name: HuggingFace pre-trained model name.
                Only supported encoder-decoder model or atuoregressive model.
            has_encoder: Whether the model has an encoder.
                If True, it's a T5-type model (encoder-decoder transformer).
                If fasle, it's a GPT-type model (atuoregressive transformer).
            tokenizer_max_length: this sets the maximum sentence length the tokenizer
                is allowed to generate. This can be customized for your specific use case. # noqa E501
        """
        self.has_encoder = has_encoder
        self.tokenizer_max_length = tokenizer_max_length
        if self.tokenizer_max_length is None:
            logging.warning(
                (
                    "Set the tokenizer_max_length is preferable for finetuning model,"
                    " which saves the cost of training."
                )
            )
        if self.has_encoder:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                pretrained_model_name
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name, padding_side="left"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # self.validation_callback is used for evaluate the model on
        # the validation dataset after each epoch.
        self.validation_callback = None
        self.training_seed = seed_generator.get_seed()

    def get_left_padding_length(cls, input_list, padding_token_id):
        """Get the left prefix length of the input list.

        Args:
            input_list: A list with the format of [prefix, ..., prefix, Others].
                The GPT tokenizer uses left padding.
            padding_token_id: The prefix of the input list.

        Returns:
            The length of [prefix, ..., prefix] in [prefix, ..., prefix, Others]
        """
        return len(list(takewhile(lambda x: x == padding_token_id, input_list)))

    def get_right_padding_length(cls, input_ids, padding_token_id):
        """Get the left prefix length of the input list.

        Args:
            input_list: A list with the format of [Others, suffix, ..., suffix].
                The T5 tokenizer uses right padding.
            padding_token_id: The suffix of the input list.

        Returns:
            The length of [suffix, ..., suffix] in [Others, suffix, ..., suffix].
        """
        # Reverse the input_ids to get the length of right padding.
        suffix_length = cls.get_left_padding_length(input_ids[::-1], padding_token_id)
        return suffix_length

    def tokenize_dataset(
        self, dataset: datasets.Dataset, shuffle: bool = True
    ) -> datasets.Dataset:
        """Tokenize the training/validation dataset.

        Args:
            dataset: Dataset.dataset with model_input and model_output columns.
            shuffle: Whether to shuffle the dataset.

        Returns:
            A `datasets.Dataset` object containing the preprocessed data with:
                "input_ids": numerical representations of input sequences to the model.
                "attention_mask": Mask to avoid performing attention on padding token
                    indices. Mask values selected in [0, 1]: 1 for tokens that are not
                    masked, 0 for masked tokens.
                "labels": Labels for language modeling. Indices are selected in
                    [-100, 0, ..., config.vocab_size - 1]. All labels set to -100 are
                    ignored (masked), the loss is only computed for labels in
                    [0, ..., config.vocab_size - 1].
        """
        if shuffle:
            dataset = dataset.shuffle(seed=seed_generator.get_seed())
        inputs = dataset["model_input"]
        outputs = dataset["model_output"]
        longest_input = max(inputs, key=len)
        longest_output = max(outputs, key=len)
        if self.tokenizer_max_length is not None and (
            len(self.tokenizer.tokenize(longest_input)) > self.tokenizer_max_length
            or len(self.tokenizer.tokenize(longest_output)) > self.tokenizer_max_length
        ):
            logging.warning(
                (
                    "Truncation happened when tokenizing dataset."
                    " You should consider increasing the tokenizer_max_length."
                    " Otherwise the truncation may lead to unexpected results."
                )
            )
        input_encodings = self.tokenizer.batch_encode_plus(
            inputs,
            truncation=True,
            max_length=self.tokenizer_max_length,
            padding=True,
        )
        output_encodings = self.tokenizer.batch_encode_plus(
            outputs,
            truncation=True,
            max_length=self.tokenizer_max_length,
            padding=True,
        )

        labels = []
        if not self.has_encoder:
            length_of_input_encoding_ids_with_padding = len(
                input_encodings["input_ids"][0]
            )
            length_of_output_encoding_ids_with_padding = len(
                output_encodings["input_ids"][0]
            )
            for idx, input_id in enumerate(input_encodings["input_ids"]):
                output_encoding_id = output_encodings["input_ids"][idx]
                length_of_padding_in_output_encoding_id = self.get_left_padding_length(
                    output_encoding_id, self.model.config.pad_token_id
                )

                # The index -100 is ignored for loss compute in Autoregressive model.
                # Reference: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2DoubleHeadsModel.forward.labels # noqa E501
                length_of_output_encoding_id_without_padding = (
                    length_of_output_encoding_ids_with_padding
                    - length_of_padding_in_output_encoding_id
                )
                assert (
                    length_of_output_encoding_id_without_padding != 0
                ), "One of the model_output is empty."
                label = [-100] * (
                    length_of_input_encoding_ids_with_padding
                    - length_of_output_encoding_id_without_padding
                ) + input_id[-length_of_output_encoding_id_without_padding:]
                assert (
                    len(label)
                    == length_of_input_encoding_ids_with_padding
                    == len(input_id)
                )
                labels.append(label)
        else:
            # For T5 model, right padding token id should not be taken into
            # account by the loss function. In PyTorch and Tensorflow, this can
            # be done by replacing them with -100, which is the ignore_index
            # of the CrossEntropyLoss.
            # Reference: https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/t5#training # noqa E501
            for output_encoding_id in output_encodings["input_ids"]:
                length_of_right_padding_in_output_encoding_id = (
                    self.get_right_padding_length(
                        output_encoding_id, self.tokenizer.pad_token_id
                    )
                )
                label = (
                    (
                        output_encoding_id[
                            :-length_of_right_padding_in_output_encoding_id
                        ]
                        + [-100] * length_of_right_padding_in_output_encoding_id
                    )
                    if length_of_right_padding_in_output_encoding_id != 0
                    else output_encoding_id
                )
                assert len(label) == len(output_encoding_id)
                labels.append(label)

        preprocessed_dict = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels,
        }
        return datasets.Dataset.from_dict(preprocessed_dict)

    def train_model(
        self,
        hyperparameter_choices: dict[str, Any],
        training_datasets: list[datasets.Dataset],
        validation_datasets: list[datasets.Dataset] | None = None,
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text generation model.

        Args:
            hyperparameter_choices: A dictionary of hyperparameters for training.
            training_datasets: Training datasets with `input_col` and `model_output`.
            validation_datasets: Validation datasets during training. If not provided,
                15% of training data will be spilt from training_datasets to validate.

        Returns:
            A trained HuggingFace model and tokenizer.
        """
        hyperparameter_choices_keys = set(hyperparameter_choices.keys())
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
        assert hyperparameter_choices_keys.issubset(
            supported_keys
        ), f"Only support {supported_keys} as training parameters."
        training_args = Seq2SeqTrainingArguments(
            output_dir=hyperparameter_choices.get("output_dir", "./result"),
            logging_steps=hyperparameter_choices.get("logging_steps", 1),
            save_strategy=hyperparameter_choices.get("save_strategy", "no"),
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 10),
            per_device_train_batch_size=hyperparameter_choices.get(
                "per_device_train_batch_size", 100
            ),
            warmup_steps=hyperparameter_choices.get("warmup_steps", 0),
            weight_decay=hyperparameter_choices.get("weight_decay", 0.01),
            logging_dir=hyperparameter_choices.get("logging_dir", "./result"),
            learning_rate=hyperparameter_choices.get("learning_rate", 1e-4),
            predict_with_generate=True,
        )
        evaluation_strategy = hyperparameter_choices.get("evaluation_strategy", "epoch")
        if evaluation_strategy == "epoch":
            evaluate_after_epoch = True
        elif evaluation_strategy == "no":
            logging.info(
                "The traning doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )
            evaluate_after_epoch = False
        else:
            logging.warning(
                (
                    "Only `epoch` evaluation strategy is supported"
                    + ", the evaluation strategy will be set to evaluate_after_epoch."
                )
            )
            evaluate_after_epoch = True

        concatenated_training_dataset = concatenate_datasets(training_datasets)

        if evaluate_after_epoch is True:
            if validation_datasets is None:
                if not self.has_encoder:
                    # The validation dataset for autoregressive model is missed.
                    logging.warning(
                        (
                            (
                                "The validation split for autoregressive model is missed"  # noqa E501
                                + ", which should not contain labels as the training spilt."  # noqa E501
                                + " Thus this evaluation will be skipped."
                            )
                        )
                    )
                    train_dataset = self.tokenize_dataset(concatenated_training_dataset)
                    val_dataset = None
                    evaluate_after_epoch = False
                else:
                    # The validation dataset for encoder-decoder model is missed.
                    logging.warning(
                        (
                            "The validation split for encoder-decoder model is missed."  # noqa E501
                            + " The training dataset will be split to create the validation dataset."  # noqa E501
                        )
                    )
                    test_size = hyperparameter_choices.get("test_size", 0.15)
                    assert (
                        len(concatenated_training_dataset) > 1
                    ), "Training dataset should be larger than 1 for train_test_split."
                    splited_dataset = concatenated_training_dataset.train_test_split(
                        test_size=test_size, seed=self.training_seed
                    )
                    train_dataset = self.tokenize_dataset(splited_dataset["train"])
                    # the training dataset will be tokenized to train the model.
                    # But we evaluate the model on the validation dataset in the
                    # call back with the model executor and model evaluator,
                    # the validation dataset should not be tokenized.
                    val_dataset = splited_dataset["test"]
            else:
                # the training dataset will be tokenized to train the model.
                # But we evaluate the model on the validation dataset in the
                # call back with the model executor and model evaluator,
                # the validation dataset should not be tokenized.
                train_dataset = self.tokenize_dataset(concatenated_training_dataset)
                val_dataset = concatenate_datasets(validation_datasets)
        else:
            if validation_datasets:
                logging.warning(
                    "The validation dataset is provided, but the evaluation is skipped."  # noqa E501
                )
            train_dataset = self.tokenize_dataset(concatenated_training_dataset)
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
            if self.has_encoder
            else transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            optimizers=[
                torch.optim.AdamW(
                    params=self.model.parameters(), lr=training_args.learning_rate
                ),
                None,
            ],
        )

        if evaluate_after_epoch:
            assert val_dataset is not None, "Validation dataset is None"
            self.validation_callback = ValidationCallback(
                trainer, self.tokenizer, val_dataset
            )
            trainer.add_callback(self.validation_callback)

        # Train the model
        trainer.train()
        # Return the trained model and tokenizer
        return self.model, self.tokenizer
