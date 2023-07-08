"""Testing GenerationModelExecutor with different configurations."""

from unittest.mock import patch

import pytest
from datasets import Dataset

from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from test_helpers import create_gpt2_model_and_tokenizer, create_t5_model_and_tokenizer


def test_make_prediction_t5_model():
    """Test the `make_prediction` method with a T5 model."""
    # Create T5 model and tokenizer.
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

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
                "What's your name? Please reply in 10 words.",
                "Hello! Just tell me your name.",
                "How are you today? Please reply in 10 words.",
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
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

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
    test_input = "Hello World! What is your name?"
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


def test_make_single_prediction_t5_model_without_length_constraints():
    """Test GenerationModelExecutor for a T5 model without length constraints."""
    # Create T5 model and tokenizer.
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create GenerationModelExecutor.
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, tokenizer_max_length=None, sequence_max_length=None
    )

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


def test_make_single_prediction_gpt2_model_without_length_constraints():
    """Test GenerationModelExecutor for a GPT2 model without length constraints."""
    # Create GPT2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    with patch("logging.warning") as mock_warning:
        model_executor = GenerationModelExecutor(
            gpt2_model, gpt2_tokenizer, tokenizer_max_length=None, sequence_max_length=None
        )
        test_input = "Hello World! What is your name?"
        expected_warining = "The `max_length` in `self.model.generate` will default to `self.model.config.max_length` (20) if `sequence_max_length` is `None`."  # noqa: E501
        gpt2_output = model_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(expected_warining)
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


def test_sequence_max_length_init_for_gpt2():
    """Test the sequence_max_length is correctly set for gpt2."""
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer
    test_input = "Hello World!"
    # The max_seq_length is 1024, and test_input is 3 tokens.
    with patch("logging.warning") as mock_warning:
        gpt2_executor = GenerationModelExecutor(
            gpt2_model,
            gpt2_tokenizer,
            sequence_max_length=10043,
        )
        gpt2_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(
            (
                "The sequence_max_length (10043) is larger"
                " than the max_position_embeddings (1024)."
                " So the sequence_max_length will be set to 1024."
            )
        )


def test_sequence_max_length_init_for_t5():
    """Test the sequence_max_length is correctly set for t5."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer
    # Create test dataset.
    test_input = "translate English to Spanish: What's your name?"

    T5_executor = GenerationModelExecutor(
        t5_model,
        t5_tokenizer,
        sequence_max_length=10000,
    )
    T5_executor.make_single_prediction(test_input)


def test_truncation_warning_for_gpt2():
    """Test that truncation warning is raised for gpt2."""
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer
    test_input = "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
    # The max_seq_length is 1024, and test_input is 3 tokens.
    with patch("logging.warning") as mock_warning:
        gpt2_executor = GenerationModelExecutor(
            gpt2_model,
            gpt2_tokenizer,
        )
        gpt2_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing 'In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things.'. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )
