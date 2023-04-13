"""An interface for trainers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, dict

import transformers
import wandb
from datasets import DatasetDict
from prompt_parser import PromptSpec
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, TrainingArguments

# Input:
#    1) training dataset (datasets.Dataset)
#    2) Dictionary consisting of hyperparameter values to use
#       (e.g. base model, optimizer, LR, etc)
#
# Output:
#    transformers.PreTrainedModel


# pylint: disable=too-few-public-methods


class Trainer(ABC):
    """Train a model with a fixed set of hyperparameters."""

    def __init__(
        self,
        training_datasets: DatasetDict,
        hyperparameter_choices: dict[str, Any],
        prompt_spec: PromptSpec,
    ) -> None:
        """This is the base Constructor.

        Initialize trainer with training dataset(s),
        hyperparameters, and a prompt specification.
        """
        self.training_datasets = training_datasets
        self.hyperparameter_choices = hyperparameter_choices
        self.prompt_spec = prompt_spec
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        wandb.init(project="my-project", config=self.hyperparameter_choices)
        self.wandb = wandb
        raise NotImplementedError

    @abstractmethod
    def train_model(self) -> transformers.PreTrainedModel:
        """Train a model with the given hyperparameters and return it."""


class BaseTrainer(Trainer):
    """This dummy trainer does not actually train anything."""

    def train_model(self) -> transformers.PreTrainedModel:
        """This dummy trainer returns an untrained BERT-base model."""
        # Set up checkpointing
        output_dir = Path(self.hyperparameter_choices.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.hyperparameter_choices.get("checkpoint_file")
        if checkpoint_file is not None:
            checkpoint_dict = self.hyperparameter_choices.copy()
            checkpoint_dict["output_dir"] = str(output_dir)
            parser = HfArgumentParser(TrainingArguments)
            checkpoint = parser.parse_dict(checkpoint_dict)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                checkpoint_file
            )
            optimizer = transformers.AdamW(
                model.parameters(), lr=checkpoint.learning_rate
            )
            start_epoch = checkpoint.epoch + 1
            global_step = checkpoint.global_step
        else:
            # Load the tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.hyperparameter_choices["model"]
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.hyperparameter_choices["model"]
            )

            # Set up the optimizer
            optimizer = transformers.AdamW(
                model.parameters(), lr=self.hyperparameter_choices["learning_rate"]
            )

            checkpoint = TrainingArguments(output_dir=output_dir)
            start_epoch = 0
            global_step = 0

        # Train the model
        batch_size = self.hyperparameter_choices.get("batch_size", 32)
        train_data_loader = DataLoader(
            self.training_datasets["train"], batch_size=batch_size, shuffle=True
        )

        for epoch in range(start_epoch, checkpoint.num_train_epochs):
            for batch in train_data_loader:
                inputs = tokenizer(
                    batch["input_ids"],
                    batch["attention_mask"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                labels = batch["label"]

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()

                # Save checkpoint every save_steps steps
                if (global_step + 1) % checkpoint.save_steps == 0:
                    checkpoint.epoch = epoch
                    checkpoint.global_step = global_step + 1
                    checkpoint_dir = output_dir / f"checkpoint-{global_step + 1}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    checkpoint.save((checkpoint_dir / "checkpoint").as_posix())

                global_step += 1

        # Save final model if checkpointing was enabled
        if checkpoint_file is None:
            model.save_pretrained(output_dir)

        return model
