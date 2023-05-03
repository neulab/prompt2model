from typing import Any

import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments

from prompt2model.model_trainer import ModelTrainer
from datasets import DatasetDict

from functools import partial


def preprocess_dataset(
    example, tokenizer: transformers.AutoTokenizer
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
    inputs = example["input_col"]
    outputs = example["output_col"]
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=128)
    output_encodings = tokenizer(outputs, truncation=True, padding=True, max_length=128)
    example["input_ids"] = input_encodings["input_ids"]
    example["attention_mask"] = input_encodings["attention_mask"]
    example["labels"] = output_encodings["input_ids"]
    return example



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

        # Concatenate and preprocess the training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=42)
        preprocess_dataset = shuffled_dataset.map(partial(preprocess_dataset, tokenizer=self.tokenizer), batched=True)

        def data_collator(batch):
            print(batch)
            inputs = [example["input_ids"] for example in batch]
            attention_masks = [example["attention_mask"] for example in batch]

            return{
                "input_ids": inputs,
                "attention_mask": attention_masks,
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=preprocess_dataset,
            data_collator=partial(data_collator),
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
