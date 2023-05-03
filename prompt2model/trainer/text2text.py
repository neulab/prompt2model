"""This module provides a T5 model trainer."""

from typing import Any

import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments

from prompt2model.trainer import ModelTrainer


def preprocess_dataset(
    dataset: datasets.Dataset, tokenizer: transformers.AutoTokenizer
):
    """Preprocesses the given dataset using the given tokenizer.

    Args:
    - dataset: A dictionary-like object containing the input and output columns.
    - tokenizer: A tokenizer object that can encode the input and output texts.

    Returns:
    - A `datasets.Dataset` object containing the preprocessed data with:
      - "input_ids": A list of token IDs for the encoded input texts.
      - "attention_mask": A list of 0s and 1s indicating which tokens are padding.
      - "labels": A list of token IDs for the encoded output texts.
    """
    inputs = dataset["input_col"]
    outputs = dataset["output_col"]
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=128)
    output_encodings = tokenizer(outputs, truncation=True, padding=True, max_length=128)

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": output_encodings["input_ids"],
        }
    )


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
            num_train_epochs=hyperparameter_choices["num_train_epochs"],
            per_device_train_batch_size=hyperparameter_choices["batch_size"],
            warmup_steps=hyperparameter_choices["warmup_steps"],
            weight_decay=hyperparameter_choices["weight_decay"],
            logging_dir="./logs",
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
        )

        # Concatenate and preprocess the training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42, buffer_size=1000)
        preprocessed_dataset = preprocess_dataset(shuffled_dataset, self.tokenizer)

        # Create the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            data_collator=lambda data: {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "labels": data["labels"],
            },
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
