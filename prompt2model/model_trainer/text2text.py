"""A simple trainer for text-to-text models."""

from typing import Any

import datasets
import torch
import transformers
from datasets import concatenate_datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer

from prompt2model.model_trainer import ModelTrainer


class TextToTextTrainer(ModelTrainer):
    """This is a simple trainer for a T5-based text-to-text model."""

    def __init__(self, pretrained_model_name: str):
        """Initialize a model trainer.

        Args:
            pretrained_model_name: A HuggingFace model name to use for training.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        self.wandb = None

    def prepare_inputs(
        self,
        input_text: str,
        output_text: str,
        max_length: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess input and output text using the model's tokenizer.

        Args:
            input_text: Input text.
            output_text: Output text.
            max_length: Maximum length of the encoded text.

        Returns:
            A tuple containing the input and output tensors.
        """
        input_encoding = self.tokenizer(
            input_text,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            output_text,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            input_encoding.input_ids,
            input_encoding.attention_mask,
        )
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text-to-text T5-based model.

        Args:
            training_datasets: Training `Dataset`s, with `input_col` and `output_col`.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model and its tokenizer.
        """
        # Set up training arguments
        training_args = transformers.TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameter_choices.get("num_train_epochs", 1),
        )

        # Set up optimizer
        optimizer = transformers.AdamW(self.model.parameters())

        # Concatenate and shuffle training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42)

        # Train the model
        for epoch in range(training_args.num_train_epochs):
            for step, example in enumerate(shuffled_dataset):
                # Prepare inputs and labels
                input_ids, attention_mask, labels = self.prepare_inputs(
                    example["input_col"],
                    example["output_col"],
                    max_length=hyperparameter_choices.get("max_length", 100),
                )

                # Compute loss and update parameters
                optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()
                optimizer.step()

                # Print loss every 1000 steps
                if step % 1000 == 0:
                    print(f"Epoch {epoch}, Step {step}: Loss {loss.item()}")

        # Return trained model and its tokenizer
        return self.model, self.tokenizer
