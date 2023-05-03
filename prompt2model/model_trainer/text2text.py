from typing import Any

import torch
import datasets
import transformers
from datasets import concatenate_datasets
from transformers import TrainingArguments, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from prompt2model.model_trainer import ModelTrainer


class TextToTextTrainer(ModelTrainer):
    """This is a simple trainer for a T5 based text2text model."""

    def __init__(self, pretrained_model_name: str):
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text2text T5 based model.

        Args:
            training_datasets: Training `Dataset`s, with `input_col` and `output_col`.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model.
        """
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 1),
            per_device_train_batch_size=hyperparameter_choices.get("batch_size", 1),
            warmup_steps=hyperparameter_choices.get("warmup_steps", 0),
            weight_decay=hyperparameter_choices.get("weight_decay", 0.01),
            logging_dir="./logs",
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
        )

        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42)

        for example in shuffled_dataset:
            print(example)
            input_ids = self.tokenizer(example["input_col"], return_tensors="pt").input_ids
            labels = self.tokenizer(example["output_col"], return_tensors="pt").input_ids

            # the forward function automatically creates the correct decoder_input_ids
            loss = self.model(input_ids=input_ids, labels=labels).loss
            loss.item()
