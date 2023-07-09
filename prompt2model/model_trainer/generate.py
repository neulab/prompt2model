"""A trainer class to train generation models."""

from __future__ import annotations  # noqa FI58

import logging
import os
from typing import Any

import datasets
import torch
import transformers
from datasets import concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.utils import seed_generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaluationCallback(TrainerCallback):
    """The evaluation will be conduted on validation set after each  epoch."""

    def __init__(self, trainer, tokenizer, val_dataset) -> None:
        """Initializes a new instance of AutoregressiveModelCallback.

        Args:
            trainer: Trainer instance.
                After each epoch of Training, this callback will be called.
            tokenizer: Tokenizer used for ModelExecutor.
            val_dataset: Validation dataset to be evaluated on.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        """After each  evaluation, this function will be called."""
        _ = (args, state, control, kwargs)
        # Pass the unused paramerters warning.
        logging.info("Coduct real evaluation on each epoch's ending.")

        model_executor = GenerationModelExecutor(
            self.trainer.model,
            self.tokenizer,
            self.val_dataset,
            "model_input",
            10,
        )
        model_outputs = model_executor.make_prediction()
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            self.val_dataset,
            "output_col",
            model_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        logging.info(metric_values)


class GenerationModelTrainer(BaseTrainer):
    """Trainer for T5 type (encoder-decoder) model and GPT type (deocder-only) model."""

    def __init__(
        self,
        pretrained_model_name: str,
        has_encoder: bool,
        model_max_length: int | None = None,
    ):
        """Initializes a new instance of GenerationModelTrainer.

        Args:
            pretrained_model_name: HuggingFace pre-trained model name.
                Only supported encoder-decoder model or atuoregressive model.
            has_encoder: Whether the model has an encoder.
                If True, it's a T5-type model (encoder-decoder transformer).
                If fasle, it's a GPT-type model (atuoregressive transformer).
            model_max_length: this sets the maximum sentence length allowed by an
            encoder-decoder model. This can be customized for your specific use case.
        """
        self.has_encoder = has_encoder
        self.model_max_length = model_max_length
        if self.has_encoder:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name
            )
            if model_max_length:
                self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                    pretrained_model_name, model_max_length=model_max_length
                )
            else:
                self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                    pretrained_model_name
                )
        else:
            if model_max_length is not None:
                logging.warning(
                    "model_max_length is only supported for encoder-decoder models"
                )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            # Save the pad_id to the model's config instead of the function

    def tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Tokenize the training/validation dataset.

        Args:
            dataset: Dataset.dataset with model_input and output_col columns.

        Returns:
            A `datasets.Dataset` object containing the preprocessed data with:
                "input_ids": A list of token IDs for the encoded input texts.
                "attention_mask": A list of 0/1 indicating which tokens are padding.
                "labels": A list of token IDs for the encoded output texts.
        """
        shuffled_dataset = dataset.shuffle(seed=seed_generator.get_seed())
        inputs = shuffled_dataset["model_input"]
        outputs = shuffled_dataset["output_col"]
        input_encodings = self.tokenizer.batch_encode_plus(
            inputs, truncation=True, max_length=self.model_max_length, padding=True
        )
        output_encodings = self.tokenizer.batch_encode_plus(
            outputs, truncation=True, max_length=self.model_max_length, padding=True
        )
        preprocessed_dict = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": output_encodings["input_ids"]
            if self.has_encoder
            else input_encodings["input_ids"],
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
            hyperparameter_choices: A dictionary of hyperparameter choices.
            training_datasets: Training datasets with `input_col` and `output_col`.
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
        ), f"Only support {supported_keys} as training parameters"
        training_args = Seq2SeqTrainingArguments(
            output_dir=hyperparameter_choices.get("output_dir", "./result"),
            logging_steps=hyperparameter_choices.get("logging_steps", 8),
            save_strategy=hyperparameter_choices.get("save_strategy", "no"),
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 10),
            per_device_train_batch_size=hyperparameter_choices.get(
                "per_device_train_batch_size", 100
            ),
            warmup_steps=hyperparameter_choices.get("warmup_steps", 0),
            weight_decay=hyperparameter_choices.get("weight_decay", 0.01),
            logging_dir=hyperparameter_choices.get("logging_dir", "./logs"),
            learning_rate=hyperparameter_choices.get("learning_rate", 1e-4),
            predict_with_generate=True,
        )
        evaluation_strategy = hyperparameter_choices.get("evaluation_strategy", "epoch")
        if evaluation_strategy == "epoch":
            evaluate_after_epoch = True
        elif evaluation_strategy is not None:
            logging.warning(
                (
                    "Only `epoch` evaluation strategy is supported"
                    + ", the evaluation strategy will be set to  evaluate_after_epoch"
                )
            )
            evaluate_after_epoch = True
        else:
            logging.info("The traning doesn't set the evaluation strategy.")
            evaluate_after_epoch = False

        concatenated_training_dataset = concatenate_datasets(training_datasets)

        if not validation_datasets:
            if not self.has_encoder:
                logging.warn(
                    (
                        (
                            "The validation split for autoregressive model is missed"
                            + ", which should not contain labels as the training spilt."
                            + "  Thus this evaluation will be skipped."
                        )
                    )
                )
                train_dataset = self.tokenize_dataset(concatenated_training_dataset)
                val_dataset = None
                evaluate_after_epoch = False
            else:
                logging.warn(
                    (
                        "The validation split for encoder-decoder model is missed."
                        + " The training dataset will be split to evaluate the model."
                    )
                )
                test_size = hyperparameter_choices.get("test_size", 0.15)
                splited_dataset = concatenated_training_dataset.train_test_split(
                    test_size=test_size, seed=seed_generator.get_seed()
                )
                train_dataset = self.tokenize_dataset(splited_dataset["train"])
                # the training dataset will be tokenized to train the model.
                # But we evaluate the model on the validation dataset with
                # the model executor and model evaluator, so the validation
                # dataset should not be tokenized.
                val_dataset = splited_dataset["test"]
        else:
            val_dataset = concatenate_datasets(validation_datasets)
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
            trainer.add_callback(
                EvaluationCallback(trainer, self.tokenizer, val_dataset)
            )

        # Train the model
        trainer.train()
        # Return the trained model and tokenizer
        return self.model, self.tokenizer
