"""Testing the autoregressive GenerationModelExecutor with different configurations."""

import gc
import logging
from unittest.mock import patch

from datasets import Dataset

from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from test_helpers import create_gpt2_model_and_tokenizer

logger = logging.getLogger("ModelExecutor")


def test_make_prediction_gpt2():
    """Test the `make_prediction` method with a GPT-2 model."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a test dataset.
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "What's your name? Please reply in 10 words.",
                "Hello! Just tell me your name.",
                "How are you today? Please reply in 10 words.",
            ]
        }
    )

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)

    # Test the GPT-2 model.
    gpt2_outputs = model_executor.make_prediction(test_dataset, "model_input")
    assert isinstance(gpt2_outputs, list)
    assert len(gpt2_outputs) == len(test_dataset)

    for output in gpt2_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert list(output.auxiliary_info.keys()) == [
            "input_text",
            "logits",
        ]
        assert isinstance(output.auxiliary_info, dict)
    gc.collect()


def test_make_single_prediction_gpt2():
    """Test the `make_single_prediction` with a GPT-2 model."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)

    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    gpt2_output = model_executor.make_single_prediction(test_input)
    assert isinstance(gpt2_output, ModelOutput)
    assert gpt2_output.prediction is not None
    assert list(gpt2_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(gpt2_output.auxiliary_info, dict)


def test_make_single_prediction_gpt2_without_length_constraints():
    """Test GenerationModelExecutor for a GPT2 model without length constraints."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        model_executor = GenerationModelExecutor(
            gpt2_model,
            gpt2_tokenizer,
            tokenizer_max_length=None,
            sequence_max_length=None,
        )
        test_input = "Hello World! What is your name?"
        expected_warining = "The `max_length` in `self.model.generate` will default to `self.model.config.max_length` (20) if `sequence_max_length` is `None`."  # noqa: E501
        gpt2_output = model_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(expected_warining)
        mock_info.assert_not_called()
    assert isinstance(gpt2_output, ModelOutput)
    assert gpt2_output.prediction is not None
    assert list(gpt2_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(gpt2_output.auxiliary_info, dict)
    gc.collect()


def test_sequence_max_length_init_for_gpt2():
    """Test that the sequence_max_length is correctly set for gpt2."""
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer
    test_input = "What's your name? Please reply in 10 words."
    # The max_seq_length is 1024, and test_input is 3 tokens.
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
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
        mock_info.assert_not_called()
        # The max_position_embeddings is 1024, so the
        # sequence_max_length will be scaled to 1024.
        assert (
            gpt2_executor.sequence_max_length
            == gpt2_executor.model.config.max_position_embeddings
            == 1024
        )
    gc.collect()


def test_truncation_warning_for_gpt2_executor():
    """Test that truncation warning is raised for GPT2 ModelExecutor."""
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer
    test_input = "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
    # The default tokenizer_max_length is 256, and test_input is 332 tokens.
    gpt2_executor = GenerationModelExecutor(
        gpt2_model,
        gpt2_tokenizer,
    )
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        gpt2_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset / input string. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )
        mock_info.assert_not_called()
    gc.collect()


def test_beam_search_for_gpt2_executor():
    """Test the beam search for GPT2 ModelExecutor."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)
    hyperparameter_choices = {"generate_strategy": "beam", "num_beams": 4}
    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    model_output = model_executor.make_single_prediction(
        test_input, hyperparameter_choices
    )
    assert isinstance(model_output, ModelOutput)
    assert model_output.prediction is not None
    assert list(model_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(model_output.auxiliary_info, dict)
    gc.collect()


def test_greedy_search_for_gpt2_executor():
    """Test the greedy search for GPT2 ModelExecutor."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "greedy",
    }
    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    model_output = model_executor.make_single_prediction(
        test_input, hyperparameter_choices
    )
    assert isinstance(model_output, ModelOutput)
    assert model_output.prediction is not None
    assert list(model_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(model_output.auxiliary_info, dict)


def test_top_k_sampling_for_gpt2_executor():
    """Test the top_k sampling for GPT2 ModelExecutor."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "top_k",
        "top_k": 20,
    }
    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    model_output = model_executor.make_single_prediction(
        test_input, hyperparameter_choices
    )
    assert isinstance(model_output, ModelOutput)
    assert model_output.prediction is not None
    assert list(model_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(model_output.auxiliary_info, dict)
    gc.collect()


def test_top_p_sampling_for_gpt2_executor():
    """Test the top_p sampling for GPT2 ModelExecutor."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "top_p",
        "top_p": 0.7,
    }
    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    model_output = model_executor.make_single_prediction(
        test_input, hyperparameter_choices
    )
    assert isinstance(model_output, ModelOutput)
    assert model_output.prediction is not None
    assert list(model_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(model_output.auxiliary_info, dict)


def test_intersect_sampling_for_gpt2_executor():
    """Test the intersect sampling for GPT2 ModelExecutor."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(gpt2_model, gpt2_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "intersect",
        "top_p": 0.7,
        "top_k": 20,
    }
    # Test making a single prediction from the GPT-2 model.
    test_input = "What's your name? Please reply in 10 words."
    model_output = model_executor.make_single_prediction(
        test_input, hyperparameter_choices
    )
    assert isinstance(model_output, ModelOutput)
    assert model_output.prediction is not None
    assert list(model_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(model_output.auxiliary_info, dict)
    gc.collect()
