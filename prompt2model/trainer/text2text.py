"""This module provides a simple trainer."""

from typing import Any

import datasets
import transformers
from transformers import AutoModel
from transformers import Trainer
from transformers import TrainingArguments

from prompt2model.trainer import Model_Trainer


class SimpleTrainer(Model_Trainer):
    """This is a simple trainer."""

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> transformers.PreTrainedModel:
        """Train a sequence classification model.

        Clear all parameters in the final layer and retrain it.

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
        weight_decay = hyperparameter_choices["weight_decay"]

        # Concatenate all training datasets

        # Why we need a list of datasets? I guess currently in our project, we
        # the first dataset is retrieved form `DatasetRetriever`, the scond is
        # from LLM's generation. At least, they should have a shared target
        # like 3-class classificaton. But could we concatenate them up in the
        # main pipeline into a single dataset, then pass it in?
        concatenated_dataset = datasets.concatenate_datasets(training_datasets)

        # Clear all parameters in the final layer and retrain it
        model = AutoModel.from_pretrained(self.model.base_model_prefix)
        model.classifier = transformers.Identity()
        model.classifier.out_proj = transformers.Linear(
            model.config.hidden_size, self.num_labels
        )

        # Set up training arguments and trainer
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            push_to_hub=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=concatenated_dataset,
            tokenizer=self.tokenizer,
        )

        # Train the model
        trainer.train()

        # Return the trained model
        return trainer.model
