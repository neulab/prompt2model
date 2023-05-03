"""Testing TextToTextTrainer."""

import datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer

from prompt2model.model_trainer.text2text import TextToTextTrainer


def test_text_to_text_trainer():
    # Initialize a TextToTextTrainer instance with the small T5
    trainer = TextToTextTrainer("t5-small")

    training_datasets = [
        datasets.Dataset.from_dict(
            {"input_col": ["translate apple to french"] * 1000, "output_col": ["pomme"] * 1000}
        ),
    ]

    # Train the TextToTextTrainer instance on the training dataset
    trained_model, trained_tokenizer = trainer.train_model(training_datasets, {})

    # Verify that the trained model is a T5ForConditionalGeneration model
    assert isinstance(trained_model, T5ForConditionalGeneration)

    # Verify that the trained tokenizer is a T5Tokenizer model
    assert isinstance(trained_tokenizer, T5Tokenizer)

    # Generate a prediction using the trained model and tokenizer
    inputs = "translate banana to french"
    input_ids = trained_tokenizer.encode(inputs, return_tensors="pt")
    outputs = trained_model.generate(input_ids)
    output_text = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Verify that the prediction is correct
    assert output_text == "banane"
