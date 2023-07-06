"""Testing GenerationModelExecutor with different configurations."""

from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from test_helpers import create_gpt2_model_and_tokenizer


def test_make_prediction_t5_model():
    """Test the `make_prediction` method with a T5 model."""
    # Create T5 model and tokenizer.
    t5_model_name = "t5-small"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    # Create test dataset.
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "This is the first test input.",
                "Another example for testing.",
                "One more test input.",
            ]
        }
    )

    # Create GenerationModelExecutor.
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, test_dataset, "model_input"
    )

    # Test T5 model.
    t5_outputs = model_executor.make_prediction()
    assert isinstance(t5_outputs, list)
    assert len(t5_outputs) == len(test_dataset)

    for output in t5_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert output.confidence is not None
        assert list(output.auxiliary_info.keys()) == [
            "example",
            "input_text",
            "logits",
            "probs",
        ]
        assert isinstance(output.auxiliary_info, dict)


def test_make_prediction_gpt2_model():
    """Test the `make_prediction` method with a GPT-2 model."""
    # Create GPT2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create test dataset.
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "This is the first test input.",
                "Another example for testing.",
                "One more test input.",
            ]
        }
    )

    # Create GenerationModelExecutor.
    model_executor = GenerationModelExecutor(
        gpt2_model, gpt2_tokenizer, test_dataset, "model_input"
    )

    # Test GPT-2 model.
    gpt2_outputs = model_executor.make_prediction()
    assert isinstance(gpt2_outputs, list)
    assert len(gpt2_outputs) == len(test_dataset)

    for output in gpt2_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert output.confidence is not None
        assert list(output.auxiliary_info.keys()) == [
            "example",
            "input_text",
            "logits",
            "probs",
        ]
        assert isinstance(output.auxiliary_info, dict)


def test_make_single_prediction_t5_model():
    """Test the `make_single_prediction` method with a T5 model."""
    # Create T5 model and tokenizer.
    t5_model_name = "t5-small"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    # Create GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)

    # Test T5 model single prediction.
    test_input = "This is a test input."
    t5_output = model_executor.make_single_prediction(test_input)
    assert isinstance(t5_output, ModelOutput)
    assert t5_output.prediction is not None
    assert t5_output.confidence is not None
    assert list(t5_output.auxiliary_info.keys()) == [
        "example",
        "input_text",
        "logits",
        "probs",
    ]
    assert isinstance(t5_output.auxiliary_info, dict)


def test_make_single_prediction_gpt2_model():
    """Test the `make_single_prediction` with a GPT-2 model."""
    # Create GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)

    # Test GPT-2 model single prediction.
    test_input = "This is a test input."
    gpt2_output = model_executor.make_single_prediction(test_input)
    assert isinstance(gpt2_output, ModelOutput)
    assert gpt2_output.prediction is not None
    assert gpt2_output.confidence is not None
    assert list(gpt2_output.auxiliary_info.keys()) == [
        "example",
        "input_text",
        "logits",
        "probs",
    ]
    assert isinstance(gpt2_output.auxiliary_info, dict)


def test_wrong_init_for_model_excutor():
    """Test the input_column and test_set should be provided simultaneously."""
    with pytest.raises(AssertionError) as exc_info:
        gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
        gpt2_model = gpt2_model_and_tokenizer.model
        gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

        # Create test dataset.
        test_dataset = Dataset.from_dict(
            {
                "model_input": [
                    "This is the first test input.",
                    "Another example for testing.",
                    "One more test input.",
                ]
            }
        )

        # Create GenerationModelExecutor.
        _ = GenerationModelExecutor(gpt2_model, gpt2_tokenizer, test_set=test_dataset)
        assert str(exc_info.value) == (
            "input_column and test_set should be provided simultaneously."
        )


def test_the_max_new_tokens_init_for_gpt2():
    """Test the max_new_tokens are correctly set for gpt2."""
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer
    test_input = "Hello world."
    # The max_seq_length is 1024, and test_input is 3 tokens.
    with patch("logging.warning") as mock_warning:
        gpt2_executor = GenerationModelExecutor(
            gpt2_model,
            gpt2_tokenizer,
            expected_max_new_tokens=10000,
        )
        gpt2_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(
            (
                "The input_ids together with the expected_max_new_tokens"
                " (10003) are longerthan the max_sequence_length (1024). "
                "The max_new_tokens will be set to the "
                "(max_sequence_length - input_ids)."
            )
        )


def test_the_max_new_tokens_init_for_t5():
    """Test the max_new_tokens are correctly set for t5."""
    t5_model_name = "t5-small"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    # Create test dataset.
    test_input = "translate English to Spanish: What's your name?"

    T5_executor = GenerationModelExecutor(
        t5_model,
        t5_tokenizer,
        expected_max_new_tokens=10000,
    )
    T5_executor.make_single_prediction(test_input)
