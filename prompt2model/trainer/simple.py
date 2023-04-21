"""This module provides a simple trainer."""

from pathlib import Path
from typing import Any

import datasets
import transformers
from torch.utils.data import ConcatDataset, DataLoader

from prompt2model.trainer import Trainer


class SimpleTrainer(Trainer):
    """This is a simple trainer."""

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> transformers.PreTrainedModel:
        """Train a sequence classification model.

        Use the self.model as a pretrained feature extractor
        retrain the final layer from scratch to To solve the
        inconsistency of pretraining and finetuning.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model.
        """

        # Extract hyperparameters
        batch_size = hyperparameter_choices["batch_size"]
        num_epochs = hyperparameter_choices["num_epochs"]
        learning_rate = hyperparameter_choices["learning_rate"]
        device = hyperparameter_choices["device"]

        # Concatenate all training datasets
        concatenated_dataset = ConcatDataset(training_datasets)

        # Create dataloader
        train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

        # Clear all parameters in the final layer and retrain it
        self.model.classifier = transformers.Identity()
        self.model.classifier.out_proj = transformers.Linear(self.model.config.hidden_size, self.model.config.num_labels)

        # Set up optimizer and scheduler
        optimizer = transformers.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * num_epochs,
        )

        # Set device and precision
        self.model.to(device)
        self.model.train()
        transformers.set_seed(42)

        # Train the model
        for epoch in range(num_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Return the trained model
        return self.model