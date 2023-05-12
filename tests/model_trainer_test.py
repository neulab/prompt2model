"""Testing GenerationModelTrainer with different configurations."""

import tempfile

import datasets
import transformers

from prompt2model.model_trainer.generate import GenerationModelTrainer


def test_trainer():
    """Test the GenerationModelTrainer class.

    This function tests the GenerationModelTrainer class by training two
    different models using different configurations. The first model is an
    encoder-decoder model implemented with the T5 architecture, and the
    second is a decoder-only model implemented with the GPT-2 architecture.
    The function creates temporary directories to store the trained models
    and their respective tokenizers. It then creates two training datasets
    with synthetic data, one for each model. Finally, it trains each model
    using the GenerationModelTrainer class and tests whether the output
    models and tokenizers are of the expected type.
    """
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
            training_datasets,
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
        )

        assert isinstance(trained_model, transformers.T5ForConditionalGeneration)
        assert isinstance(trained_tokenizer, transformers.T5Tokenizer)

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
                        "translate English to French. Example: apple. Label: pomme"
                    ]
                    * 2,
                    "output_col": ["pomme"] * 2,
                }
            ),
        ]

        trained_model, trained_tokenizer = trainer.train_model(
            training_datasets,
            {"output_dir": cache_dir, "num_train_epochs": 1, "batch_size": 1},
        )

        assert isinstance(trained_model, transformers.GPT2LMHeadModel)

        assert isinstance(trained_tokenizer, transformers.PreTrainedTokenizerFast)
