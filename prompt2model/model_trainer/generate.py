"""A trainer class to train generation models."""

from __future__ import annotations  # noqa FI58

import os
from itertools import takewhile
from typing import Any, Optional

import datasets
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.model_trainer.callback import ValidationCallback
from prompt2model.param_selector import OptunaParamSelector
from prompt2model.utils import get_formatted_logger, seed_generator
from prompt2model.utils.config import DEFAULT_HYPERPARAMETERS

logger = get_formatted_logger("ModelTrainer")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenerationModelTrainer(BaseTrainer):
    """Trainer for T5 type (encoder-decoder) model and GPT type (deocder-only) model."""

    def __init__(
        self,
        pretrained_model_name: str,
        has_encoder: bool,
        executor_batch_size: int = 10,
        tokenizer_max_length: int = 512,
        sequence_max_length: int = 1024,
        device: Optional[str] = None,
    ):
        """Initializes a new instance of GenerationModelTrainer.

        Args:
            pretrained_model_name: HuggingFace pre-trained model name.
                Only supported encoder-decoder model or atuoregressive model.
            has_encoder: Whether the model has an encoder.
                If True, it's a T5-type model (encoder-decoder transformer).
                If fasle, it's a GPT-type model (atuoregressive transformer).
            executor_batch_size: The batch size for model executor to
                make predictions.
            tokenizer_max_length: The maximum sentence length the tokenizer
                is allowed to generate.
            sequence_max_length: The maximum number of tokens the model is
                allowed to generate when being evaluated on validation dataset.
                Note that sequence_max_length might be scaled in the ModelExecutor
                if it exceeds the model's max_embedding.
            device: The device in which the operations should happen. It is ideal to
                use GPU. Hence this is Optional, where if no value is given, it will
                be assumed that GPU is being used.
        """
        self.has_encoder = has_encoder
        self.tokenizer_max_length = tokenizer_max_length
        self.sequence_max_length = sequence_max_length
        self.executor_batch_size = executor_batch_size
        if self.tokenizer_max_length is None:
            logger.warning(
                (
                    "Set the tokenizer_max_length is preferable for finetuning model,"
                    " which saves the cost of training."
                )
            )
        if self.has_encoder:
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
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
        self.validation_callback: ValidationCallback | None = None
        self.training_seed = seed_generator.get_seed()

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

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

                Note that -100 is the default ignore index for labels when computing
                    loss. You can check it by:
                    from torch import nn
                    loss_function = nn.CrossEntropyLoss()
                    IGNORE_INDEX = loss_function.ignore_index
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
            logger.warning(
                (
                    "Truncation happened when tokenizing dataset."
                    " Consider increasing the tokenizer_max_length if possible."
                    " Otherwise, truncation may lead to unexpected results."
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
        loss_function = nn.CrossEntropyLoss()
        IGNORE_INDEX = loss_function.ignore_index
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
                # The IGNORE_INDEX is ignored for loss compute in Autoregressive model.
                # Reference: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2DoubleHeadsModel.forward.labels # noqa E501
                length_of_output_encoding_id_without_padding = (
                    length_of_output_encoding_ids_with_padding
                    - length_of_padding_in_output_encoding_id
                )
                assert (
                    length_of_output_encoding_id_without_padding != 0
                ), "One of the model_output is empty."
                label = [IGNORE_INDEX] * (
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
            # For T5 model, right padding token id should ignored by the loss
            # function. In PyTorch and Tensorflow, this can be done by replacing
            # them with IGNORE_INDEX, which is the ignore_index of the
            # CrossEntropyLoss as demonstrated before.
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
                        + [IGNORE_INDEX] * length_of_right_padding_in_output_encoding_id
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

    def search_best_hyperparameters(
        self,
        hyperparameters: dict[str, Any],
        training_datasets: Dataset,
        validation_datasets: Dataset,
    ) -> transformers.Trainer:
        """Method to do automatic hyperparameter search.

        The hyperparameters in this argument will be the same as the hyperparameter
        choices in the `train_model()` method. However the hyperparameter_choices
        parameter should follow this schema as shown in this below example.

        ```python
        hyperparameter_choice = {
            "static_hyperparameters": {
                "output_dir": "./result",
                "logging_steps": 1,
                "save_strategy": "no",
                "num_train_epochs": 10,
                "per_device_train_batch_size": 100,
                "warmup_steps": 0,
                "weight_decay": 0.01,
                "logging_dir": "./result",
                "learning_rate": 1e-4,
                "evaluation_strategy": "epoch",
                "test_size": 0.15,
            },

            "optuna": {
                "min_num_train_epochs": 5,
                "max_num_train_epochs": 10,
                "save_strategy": ["epoch", "steps", "no"],
                "evaluation_strategy": ["epoch", "no"],
                "per_device_train_batch_size": [4, 8, 16, 32],
                "min_weight_decay": 1e-5,
                "max_weight_decay": 1e-1,
                "min_learning_rate": 1e-5,
                "max_learning_rate": 1e-1,
            },
        }
        ```
        Here all of the keys are optional. Here are the criterions laid out:

        :key `static_hyperparameters`: These are predefined hyperparameters solely
        for doing model training and not hyperparameter search. if this key is not
        provided then DEFAULT_HYPERPARAMETERS will be choosen to train model.

        :key `optuna`: The presense of key will signify that the user
        is interested in doing hyperparameter optimization using optuna library.
        User has two options here. Either pass "default" as the value or another dict.
        Passing "default" will mean that it will choose from the default hyperparameter
        space else, it will be customizing the hyperparameter space with the provided
        custom values.

        :key `gpt`: Not implemented.

        Args:
            hyperparameters (dict[str, Any]): A dictionary of hyperparameter choices
                for trainer and optimization
            training_datasets (Dataset): One or more training datasets for the trainer.
            validation_datasets (Dataset): A dataset for computing validation metrics.

        Returns:
            transformers.Trainer: _description_
        """
        allowed_keys = {"static_hyperparameters", "optuna", "gpt", "hybrid"}
        assert (
            set(hyperparameters.keys()) <= allowed_keys
        ), "Invalid keys in hyperparameter_choice"

        if "optuna" in hyperparameters:
            optuna_hp_selector = OptunaParamSelector(self)
            optuna_value = hyperparameters["optuna"]
            assert isinstance(
                optuna_value, (str, dict)
            ), "optuna value must be a string or a dictionary"

            if isinstance(optuna_value, dict):
                allowed_optuna_keys = set(
                    optuna_hp_selector._example_hyperparameter_space.keys()
                )
                assert set(optuna_value.keys()) <= allowed_optuna_keys
                optuna_hp_space_dict = optuna_value
            else:
                optuna_hp_space_dict = optuna_hp_selector._example_hyperparameter_space

            logger.info("Starting hyperparameter search using Optuna")
            return optuna_hp_selector.select_from_hyperparameters(
                training_sets=training_datasets,
                validation=validation_datasets,
                hyperparameters=optuna_hp_space_dict,
            )

        elif "gpt" in hyperparameters:
            raise NotImplementedError(
                "Hyperparameter search using gpt is not yet implemented"
            )

        elif "hybrid" in hyperparameters:
            raise NotImplementedError("Hybrid search is not yet implemented")

        else:
            raise KeyError("Invalid Key given")

    def train_model(
        self,
        hyperparameter_choices: dict[str, Any],
        training_datasets: list[datasets.Dataset],
        validation_datasets: list[datasets.Dataset] | None = None,
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text generation model with automated hyperparameter search.

        The hyperparameter_choices
        parameter should follow this schema as shown in this below example.

        ```python
        hyperparameter_choice = {
            "static_hyperparameters": {
                "output_dir": "./result",
                "logging_steps": 1,
                "save_strategy": "no",
                "num_train_epochs": 10,
                "per_device_train_batch_size": 100,
                "warmup_steps": 0,
                "weight_decay": 0.01,
                "logging_dir": "./result",
                "learning_rate": 1e-4,
                "evaluation_strategy": "epoch",
                "test_size": 0.15,
            },

            "optuna": {
                "min_num_train_epochs": 5,
                "max_num_train_epochs": 10,
                "save_strategy": ["epoch", "steps", "no"],
                "evaluation_strategy": ["epoch", "no"],
                "per_device_train_batch_size": [4, 8, 16, 32],
                "min_weight_decay": 1e-5,
                "max_weight_decay": 1e-1,
                "min_learning_rate": 1e-5,
                "max_learning_rate": 1e-1,
            },
        }
        ```
        Here all of the keys are optional. Here are the criterions laid out:

        :key `static_hyperparameters`: These are predefined hyperparameters solely
        for doing model training and not hyperparameter search. if this key is not
        provided then DEFAULT_HYPERPARAMETERS will be choosen to train model.

        :key `optuna`: The presense of key will signify that the user
        is interested in doing hyperparameter optimization using optuna library.
        User has two options here. Either pass "default" as the value or another dict.
        Passing "default" will mean that it will choose from the default hyperparameter
        space else, it will be customizing the hyperparameter space with the provided
        custom values.

        :key `gpt`: Not implemented.

        Args:
            hyperparameter_choices: A dictionary of hyperparameters for training.
            training_datasets: Training datasets with `input_col` and `model_output`.
            validation_datasets: Validation datasets during training. If not provided,
                15% of training data will be spilt from training_datasets to validate.

        Returns:
            A trained HuggingFace model and tokenizer.
        """
        top_level_supported_keys = {"static_hyperparameters", "optuna", "gpt", "hybrid"}
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

        if hyperparameter_choices is not None:
            assert set(hyperparameter_choices.keys()).issubset(
                top_level_supported_keys
            ), f"Only support {top_level_supported_keys} as training parameters."

        if "static_hyperparameters" in hyperparameter_choices:
            assert set(
                hyperparameter_choices["static_hyperparameters"].keys()
            ).issubset(supported_keys)
        else:
            # matter of discussion whether we want to keep static_hyperparams
            # as required arg

            hyperparameter_choices["static_hyperparameters"] = DEFAULT_HYPERPARAMETERS

        evaluation_strategy = hyperparameter_choices["static_hyperparameters"].get(
            "evaluation_strategy", "epoch"
        )
        if evaluation_strategy == "epoch":
            evaluate_after_epoch = True
        elif evaluation_strategy == "no":
            logger.info(
                "The trainer doesn't set the evaluation strategy, the evaluation will be skipped."  # noqa E501
            )
            evaluate_after_epoch = False
        else:
            logger.warning(
                (
                    "Only `epoch` evaluation strategy is supported"
                    + ", the evaluation strategy will be set to evaluate_after_epoch."
                )
            )
            evaluate_after_epoch = True

        # build the training dataset
        concatenated_training_dataset = concatenate_datasets(training_datasets)

        if evaluate_after_epoch is True:
            if validation_datasets is None:
                if not self.has_encoder:
                    # The validation dataset for autoregressive model is missing.
                    logger.warning(
                        (
                            (
                                "The validation split for autoregressive model is missing"  # noqa E501
                                + ", which should not contain labels as the training spilt."  # noqa E501
                                + " Thus this evaluation will be skipped."
                            )
                        )
                    )
                    train_dataset = self.tokenize_dataset(concatenated_training_dataset)
                    val_dataset = None
                    evaluate_after_epoch = False
                else:
                    # The validation dataset for encoder-decoder model is missing.
                    logger.warning(
                        (
                            "The validation split for encoder-decoder model is missing."  # noqa E501
                            + " The training dataset will be split to create the validation dataset."  # noqa E501
                        )
                    )
                    test_size = hyperparameter_choices["static_hyperparameters"].get(
                        "test_size", 0.15
                    )
                    assert (
                        len(concatenated_training_dataset) > 1
                    ), "Dataset should be larger than 1 to make train/test split."
                    splitted_dataset = concatenated_training_dataset.train_test_split(
                        test_size=test_size, seed=self.training_seed
                    )
                    train_dataset = self.tokenize_dataset(splitted_dataset["train"])
                    # The training dataset will be tokenized to train the model.
                    # We evaluate the model on the validation dataset in the
                    # callback with the model executor and model evaluator,
                    # so the validation dataset should not be pre-tokenized here.
                    val_dataset = splitted_dataset["test"]
            else:
                # the training dataset will be tokenized to train the model.
                # But we evaluate the model on the validation dataset in the
                # call back with the model executor and model evaluator,
                # the validation dataset should not be tokenized.
                train_dataset = self.tokenize_dataset(concatenated_training_dataset)
                val_dataset = concatenate_datasets(validation_datasets)
        else:
            if validation_datasets:
                logger.warning(
                    "The validation dataset is provided, but the evaluation is skipped."  # noqa E501
                )
            train_dataset = self.tokenize_dataset(concatenated_training_dataset)

        if "optuna" in hyperparameter_choices:
            trainer = self.search_best_hyperparameters(
                hyperparameters=hyperparameter_choices,
                training_datasets=train_dataset,
                validation_datasets=validation_datasets,
            )

        else:
            training_args = Seq2SeqTrainingArguments(
                output_dir=hyperparameter_choices["static_hyperparameters"].get(
                    "output_dir", DEFAULT_HYPERPARAMETERS.get("output_dir")
                ),
                logging_steps=hyperparameter_choices["static_hyperparameters"].get(
                    "logging_steps", DEFAULT_HYPERPARAMETERS.get("logging_steps")
                ),
                save_strategy=hyperparameter_choices["static_hyperparameters"].get(
                    "save_strategy", DEFAULT_HYPERPARAMETERS.get("save_strategy")
                ),
                num_train_epochs=hyperparameter_choices["static_hyperparameters"].get(
                    "num_train_epochs", DEFAULT_HYPERPARAMETERS.get("num_train_epochs")
                ),
                per_device_train_batch_size=hyperparameter_choices[
                    "static_hyperparameters"
                ].get(
                    "per_device_train_batch_size",
                    DEFAULT_HYPERPARAMETERS.get("per_device_train_batch_size"),
                ),
                warmup_steps=hyperparameter_choices["static_hyperparameters"].get(
                    "warmup_steps", DEFAULT_HYPERPARAMETERS.get("warmup_steps")
                ),
                weight_decay=hyperparameter_choices["static_hyperparameters"].get(
                    "weight_decay", DEFAULT_HYPERPARAMETERS.get("weight_decay")
                ),
                logging_dir=hyperparameter_choices["static_hyperparameters"].get(
                    "logging_dir", DEFAULT_HYPERPARAMETERS.get("logging_dir")
                ),
                learning_rate=hyperparameter_choices["static_hyperparameters"].get(
                    "learning_rate", DEFAULT_HYPERPARAMETERS.get("learning_rate")
                ),
                predict_with_generate=True,
            )
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer
                )
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
            print(val_dataset)
            assert val_dataset is not None, "Validation dataset is None"
            self.validation_callback = ValidationCallback(
                trainer,
                self.tokenizer,
                val_dataset,
                executor_batch_size=self.executor_batch_size,
                tokenizer_max_length=self.tokenizer_max_length,
                sequence_max_length=self.sequence_max_length,
            )

        # Not at all sure when uncommenting this
        # I am getting a test error of
        # NoneType' object has no attribute 'on_train_begin'
        # trainer.add_callback(self.validation_callback)

        # Train the model
        trainer.train()
        return self.model, self.tokenizer
