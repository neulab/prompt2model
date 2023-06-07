"""A trainer class to train generation models."""

from typing import Any

import datasets
import transformers
from datasets import concatenate_datasets
from transformers import Trainer, TrainingArguments

from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.utils import seed_generator


class GenerationModelTrainer(BaseTrainer):
    """Trainer for T5 type (encoder-decoder) model and GPT type (deocder-only) model."""

    def __init__(self, pretrained_model_name: str, has_encoder: bool):
        """Initializes a new instance of HuggingFace pre-trained model.

        Args:
            pretrained_model_name: HuggingFace pre-trained model name.
                Only supported encoder-decoder model or atuoregressive model.
            has_encoder: Whether the model has an encoder.
                If True, it's a T5-type model (encoder-decoder transformer).
                If fasle, it's a GPT-type model (atuoregressive transformer).
        """
        self.has_encoder = has_encoder
        self.training_args = TrainingArguments(
            output_dir="./result",
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
        )
        if self.has_encoder:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                pretrained_model_name
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = len(self.tokenizer)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.attention_mask_fn = lambda input_ids: (
                input_ids != self.model.config.pad_token_id
            ).float()

    def preprocess_dataset(self, dataset: datasets.Dataset):
        """Preprocesses the given dataset using self.tokenizer.

        Args:
            dataset: A dictionary-like object containing the input and output columns.

        Returns:
            A `datasets.Dataset` object containing the preprocessed data with:
                "input_ids": A list of token IDs for the encoded input texts.
                "attention_mask": A list of 0/1 indicating which tokens are padding.
                "labels": A list of token IDs for the encoded output texts.
        """
        inputs = dataset["model_input"]
        outputs = dataset["output_col"]
        input_encodings = self.tokenizer(inputs, truncation=False, padding=True)
        output_encodings = self.tokenizer(outputs, truncation=False, padding=True)

        return datasets.Dataset.from_dict(
            {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": output_encodings["input_ids"]
                if self.has_encoder
                else input_encodings["input_ids"],
            }
        )

    def train_model(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
    ) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Train a text generation model.

        Args:
            training_datasets: Training datasets with `input_col` and `output_col`.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A trained HuggingFace model.
        """
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

        # Concatenate and preprocess the training datasets
        training_dataset = concatenate_datasets(training_datasets)
        shuffled_dataset = training_dataset.shuffle(seed=seed_generator.get_seed())
        preprocessed_dataset = self.preprocess_dataset(
            shuffled_dataset
        ).train_test_split(test_size=0.2, seed=seed_generator.get_seed())
        # Create the trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=preprocessed_dataset["train"],
            eval_dataset=preprocessed_dataset["test"],
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
            if self.has_encoder
            else transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )

        # Train the model
        trainer.train()

        # Return the trained model and tokenizer
        return self.model, self.tokenizer
