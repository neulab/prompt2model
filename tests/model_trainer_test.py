"""Testing GenerationModelTrainer with different configurations."""

import os
import tempfile

import datasets
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer

os.environ["WANDB_MODE"] = "dryrun"


def test_t5_trainer_with_model_max_length():
    """Train a encoder-decoder model with a specified model_max_length of 512 ."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer(
            "t5-small", has_encoder=True, model_max_length=512
        )
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
            training_datasets,
        )

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_t5_trainer_without_model_max_length():
    """Train a encoder-decoder model without a specified model_max_length ."""
    # Test encoder-decoder GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("t5-small", has_encoder=True)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": ["translate apple to french"] * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
            training_datasets,
        )
        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)


def test_gpt_trainer_without_validation_datasets():
    """Train an autoregressive model without validation datasets."""
    # Test decoder-only GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("gpt2", has_encoder=False)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
            training_datasets,
        )
        trained_model.save_pretrained(cache_dir)
        trained_tokenizer.save_pretrained(cache_dir)
        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)


def test_gpt_trainer_with_validation_datasets():
    """Train an autoregressive model with validation datasets."""
    # Test decoder-only GenerationModelTrainer implementation
    with tempfile.TemporaryDirectory() as cache_dir:
        trainer = GenerationModelTrainer("gpt2", has_encoder=False)
        training_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]
        validation_datasets = [
            datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "translate English to French.",
                        "translate English to Kinyarwanda.",
                    ],
                    "output_col": ["pomme", "pome"],
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
            training_datasets,
            validation_datasets,
        )

        assert isinstance(trained_model, transformers.GPT2LMHeadModel)
        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)