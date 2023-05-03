import pytest
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from prompt2model.trainer.text2text import TextToTextTrainer


@pytest.fixture(scope="module")
def training_datasets():
    # Define a simple training dataset with two examples
    return [
        {"input_col": "translate apple to french", "output_col": "pomme"},
        {"input_col": "translate orange to french", "output_col": "orange"},
    ]


@pytest.fixture(scope="module")
def small_t5_model():
    # Define a small T5-based model for testing
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer


def test_text_to_text_trainer(training_datasets, small_t5_model):
    # Unpack the small T5-based model and tokenizer
    model, tokenizer = small_t5_model

    # Initialize a TextToTextTrainer instance with the small T5-based model and tokenizer
    trainer = TextToTextTrainer(model, tokenizer)

    # Train the TextToTextTrainer instance on the training dataset
    trained_model, trained_tokenizer = trainer.train_model(training_datasets, {})

    # Verify that the trained model is a T5ForConditionalGeneration model
    assert isinstance(trained_model, T5ForConditionalGeneration)

    # Verify that the trained tokenizer is a T5Tokenizer model
    assert isinstance(trained_tokenizer, T5Tokenizer)

    # Generate a prediction using the trained model and tokenizer
    inputs = "translate banana to french"
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    outputs = trained_model.generate(input_ids)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Verify that the prediction is correct
    assert output_text == "banane"
