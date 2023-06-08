"""A trainer class to train generation models."""

from __future__ import annotations  # noqa FI58

import logging
import os
from typing import Any

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.utils import seed_generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenerationModelTrainer(BaseTrainer):
    """Trainer for T5 type (encoder-decoder) model and GPT type (deocder-only) model."""

    def __init__(
        self,
        pretrained_model_name: str,
        has_encoder: bool,
        model_max_length: int | None = None,
    ):
        """Initializes a new instance of HuggingFace pre-trained model.

        Args:
            pretrained_model_name: HuggingFace pre-trained model name.
                Only supported encoder-decoder model or atuoregressive model.
            has_encoder: Whether the model has an encoder.
                If True, it's a T5-type model (encoder-decoder transformer).
                If fasle, it's a GPT-type model (atuoregressive transformer).
            model_max_length: model_max_length allows model to handle
                longer sequences, and customize sequence lengths as required
                for your specific use case.
        """
        self.has_encoder = has_encoder
        self.training_args = Seq2SeqTrainingArguments(
            output_dir="./result",
            logging_steps=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )
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
                logging.warning("model_max_length is only supported for T5 models")
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

    def preprocess_dataset(
        self, dataset_list: list[datasets.Dataset]
    ) -> datasets.Dataset:
        """Concatenate and preprocess the training/validation datasets.

        Args:
            dataset_list: List of datasets wit model_input and output_col columns.

        Returns:
            A `datasets.Dataset` object containing the preprocessed data with:
                "input_ids": A list of token IDs for the encoded input texts.
                "attention_mask": A list of 0/1 indicating which tokens are padding.
                "labels": A list of token IDs for the encoded output texts.
        """
        concatenated_dataset = concatenate_datasets(dataset_list)
        shuffled_dataset = concatenated_dataset.shuffle(seed=seed_generator.get_seed())
        inputs = shuffled_dataset["model_input"]
        outputs = shuffled_dataset["output_col"]
        input_encodings = self.tokenizer(
            inputs, truncation=True, max_length=self.model_max_length, padding=True
        )
        output_encodings = self.tokenizer(
            outputs, truncation=True, max_length=self.model_max_length, padding=True
        )
        attention_masks = (
            torch.tensor(input_encodings["input_ids"]) != self.model.config.pad_token_id
        ).tolist()
        # If the model has an encoder, calculate the length of the labels and
        # set the ids of the original input's condition to -100
        if self.has_encoder:
            labels = output_encodings["input_ids"]
            for i, label in enumerate(labels):
                labels[i] = [-100 for _ in label]
        else:
            labels = input_encodings["input_ids"]
        preprocessed_dict = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": attention_masks,
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
            hyperparameter_choices: A dictionary of hyperparameter choices.
            training_datasets: Training datasets with `input_col` and `output_col`.
            validation_datasets: Validation datasets during training. If not provided,
                15% of training data will be splited from training_datasets to validate.

        Returns:
            A trained HuggingFace model and tokenizer.
        """

        def compute_metrics(eval_preds):
            metrics = [
                evaluate.load("chrf"),
                evaluate.load("exact_match"),
                evaluate.load("bertscore"),
            ]
            logits, ground_truth = eval_preds
            predicted_strings = self.tokenizer.batch_decode(
                logits, skip_special_tokens=True
            )
            ground_truth = np.where(
                ground_truth != -100, ground_truth, self.tokenizer.pad_token_id
            )
            # -100 is a special value used in PyTorch and Hugging Face Transformers
            # to indicate tokens that should be ignored in the loss computation.
            ground_strings = self.tokenizer.batch_decode(
                ground_truth, skip_special_tokens=True
            )
            metric_values = {}
            for metric in metrics:
                metric_name = metric.name
                assert metric_name in ["chr_f", "exact_match", "bert_score"]
                if metric_name == "chr_f":
                    metric.add_batch(
                        predictions=predicted_strings, references=ground_strings
                    )
                    metric_values["chr_f++"] = metric.compute(word_order=2)["score"]
                elif metric_name == "exact_match":
                    metric.add_batch(
                        predictions=predicted_strings, references=ground_strings
                    )
                    metric_values[metric_name] = metric.compute()["exact_match"]
                elif metric_name == "bert_score":
                    metric.add_batch(
                        predictions=predicted_strings, references=ground_strings
                    )
                    metric_values[metric_name] = metric.compute(
                        model_type="xlm-roberta-base"
                    )["f1"]
            return metric_values

        self.training_args.output_dir = hyperparameter_choices.get(
            "output_dir", "./result"
        )
        self.training_args.num_train_epochs = hyperparameter_choices.get(
            "num_train_epochs", 10
        )
        self.training_args.per_device_train_batch_size = hyperparameter_choices.get(
            "batch_size", 100
        )
        self.training_args.warmup_steps = hyperparameter_choices.get("warmup_steps", 0)
        self.training_args.weight_decay = hyperparameter_choices.get(
            "weight_decay", 0.01
        )
        self.training_args.logging_dir = hyperparameter_choices.get(
            "logging_dir", "./logs"
        )
        self.training_args.learning_rate = hyperparameter_choices.get(
            "learning_rate", 1e-4
        )
        self.training_args.predict_with_generate = True

        preprocessed_training_dataset = self.preprocess_dataset(training_datasets)
        if not validation_datasets:
            preprocessed_training_dataset = (
                preprocessed_training_dataset.train_test_split(
                    test_size=0.15, seed=seed_generator.get_seed()
                )
            )
            train_dataset = preprocessed_training_dataset["train"]
            val_dataset = preprocessed_training_dataset["test"]
        else:
            val_dataset = self.preprocess_dataset(validation_datasets)
            train_dataset = preprocessed_training_dataset
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
            if self.has_encoder
            else transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            optimizers=[
                torch.optim.AdamW(
                    params=self.model.parameters(), lr=self.training_args.learning_rate
                ),
                None,
            ],
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
