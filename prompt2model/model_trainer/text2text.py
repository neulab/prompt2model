"""A trainer for text2text models."""

from typing import Any

import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments

from prompt2model.model_trainer import ModelTrainer


class TextToTextTrainer(ModelTrainer):
    """This is a simple trainer for a T5 based text2text model."""

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

        def preprocess_function(examples):
            inputs = self.tokenizer(
                examples["input_col"],
                padding="max_length",
                truncation=True,
                max_length=hyperparameter_choices.get("max_length", 128),
                return_tensors="pt",
            )
            targets = self.tokenizer(
                examples["output_col"],
                padding="max_length",
                truncation=True,
                max_length=hyperparameter_choices.get("max_length", 128),
                return_tensors="pt",
            )
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": targets.input_ids,
            }

        processed_dataset = shuffled_dataset.map(
            preprocess_function, batched=True, num_proc=4
        )
        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=processed_dataset
        )
        trainer.train()
        return self.model, self.tokenizer
