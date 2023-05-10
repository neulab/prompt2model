"""This module provides a T5 model trainer."""

from typing import Any

import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments

from prompt2model.model_trainer import ModelTrainer


def preprocess_dataset(
    dataset: datasets.Dataset, tokenizer: transformers.AutoTokenizer
):
    """Preprocesses the given dataset using the given tokenizer.

    Args:
        dataset: A dictionary-like object containing the input and output columns.
        tokenizer: A tokenizer object that can encode the input and output texts.

    Returns:
        A `datasets.Dataset` object containing the preprocessed data with:
            "input_ids": A list of token IDs for the encoded input texts.
            "attention_mask": A list of 0s and 1s indicating which tokens are padding.
            "labels": A list of token IDs for the encoded output texts.
    """
    inputs = dataset["input_col"]
    input_encodings = tokenizer(inputs)

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": input_encodings["input_ids"],
        }
    )


class GPTTrainer(ModelTrainer):
    """This is a simple trainer for a GPT (decoder-only) based text generation model."""

    def __init__(self, pretrained_model_name: str):
        """Initializes a new instance of a decoder-only model.

        Args:
            pretrained_model_name: decoder-only model name.
        """
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name
        )

    def preprocess_dataset(self, dataset: datasets.Dataset):
        """Preprocesses the given dataset using self.tokenizer.

        Args:
            dataset: A dictionary-like object containing the input and output columns.

        Returns:
            A `datasets.Dataset` object containing the preprocessed data with:
                "input_ids": A list of tokenIds for the encoded input texts.
                "attention_mask": A list of 0/1 indicating which tokens are padding.
                "labels": A list of token IDs for the encoded output texts.
        """
        inputs = dataset["input_col"]
        input_encodings = self.tokenizer(inputs)

        return datasets.Dataset.from_dict(
            {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": input_encodings["input_ids"],
            }
        )

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
            output_dir=hyperparameter_choices.get("output_dir", "./result"),
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 1),
            per_device_train_batch_size=hyperparameter_choices.get("batch_size", 1),
            warmup_steps=hyperparameter_choices.get("warmup_steps", 0),
            weight_decay=hyperparameter_choices.get("weight_decay", 0.01),
            logging_dir=hyperparameter_choices.get("logging_dir", "./logs"),
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
        )

        # Concatenate and preprocess the training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42)
        preprocessed_dataset = self.preprocess_dataset(shuffled_dataset)

        # Create the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
