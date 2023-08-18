"""Testing encoder-decoder GenerationModelExecutor with different configurations."""

import gc
import logging
from unittest.mock import patch

from datasets import Dataset

from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from test_helpers import create_t5_model_and_tokenizer

logger = logging.getLogger("ModelExecutor")


def test_make_prediction_t5():
    """Test the `make_prediction` method with a T5 model."""
    # Create a T5 model and tokenizer.
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a test dataset.
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "Translate French to English: cher",
                "Translate French to English: Bonjour",
                "Translate French to English: raisin",
            ]
        }
    )

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)

    # Test T5 model.
    t5_outputs = model_executor.make_prediction(test_dataset, "model_input")
    assert isinstance(t5_outputs, list)
    assert len(t5_outputs) == len(test_dataset)

    for output in t5_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert list(output.auxiliary_info.keys()) == [
            "input_text",
            "logits",
        ]
        assert isinstance(output.auxiliary_info, dict)
    gc.collect()


def test_make_single_prediction_t5():
    """Test the `make_single_prediction` method with a T5 model."""
    # Create a T5 model and tokenizer.
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)

    # Test T5 model single prediction.
    test_input = "Translate French to English: cher"
    t5_output = model_executor.make_single_prediction(test_input)
    assert isinstance(t5_output, ModelOutput)
    assert t5_output.prediction is not None
    assert list(t5_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(t5_output.auxiliary_info, dict)
    gc.collect()


def test_make_single_prediction_t5_without_length_constraints():
    """Test GenerationModelExecutor for a T5 model without length constraints."""
    # Create a T5 model and tokenizer.
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, tokenizer_max_length=None, sequence_max_length=None
    )

    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        model_executor = GenerationModelExecutor(
            t5_model, t5_tokenizer, tokenizer_max_length=None, sequence_max_length=None
        )
        test_input = "Translate French to English: cher"
        expected_warining = "The `max_length` in `self.model.generate` will default to `self.model.config.max_length` (20) if `sequence_max_length` is `None`."  # noqa: E501
        t5_output = model_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(expected_warining)
        mock_info.assert_not_called()
    assert isinstance(t5_output, ModelOutput)
    assert t5_output.prediction is not None
    assert list(t5_output.auxiliary_info.keys()) == [
        "input_text",
        "logits",
    ]
    assert isinstance(t5_output.auxiliary_info, dict)
    gc.collect()


def test_sequence_max_length_init_for_t5():
    """Test that the sequence_max_length is correctly set for t5."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer
    # Create a test dataset.
    test_input = "translate English to Spanish: What's your name?"

    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        t5_executor = GenerationModelExecutor(
            t5_model,
            t5_tokenizer,
            sequence_max_length=10000,
        )
        t5_executor.make_single_prediction(test_input)
        mock_warning.assert_not_called()
        # T5 model has no max_position_embeddings,
        # so the sequence_max_length will not be affected.
        assert t5_executor.sequence_max_length == 10000
        mock_info.assert_not_called()
    gc.collect()


def test_truncation_warning_for_t5_executor():
    """Test that truncation warning is raised for T5 ModelExecutor."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer
    test_input = "In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things. In the shimmering golden hues of a breathtaking sunset, as the radiant orb of the sun slowly descends beyond the distant horizon, casting its warm and ethereal glow upon the rippling surface of the tranquil ocean, a myriad of vibrant colors dance and intertwine, painting a mesmerizing tableau that captivates the senses, evoking a profound sense of wonder and awe, while the gentle breeze whispers its melodious secrets through the swaying branches of towering trees, carrying with it the fragrant scent of blooming flowers, creating a symphony of nature that envelops the very essence of existence, reminding us of the boundless beauty that surrounds us, beckoning us to embrace the fleeting moments of life's ephemeral tapestry and find solace in the profound interconnectedness of all living things."  # noqa: E501
    # The default tokenizer_max_length is 256, and test_input is 406 tokens.
    t5_executor = GenerationModelExecutor(
        t5_model,
        t5_tokenizer,
    )
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        t5_executor.make_single_prediction(test_input)
        mock_warning.assert_called_once_with(
            "Truncation happened when tokenizing dataset / input string. You should consider increasing the tokenizer_max_length. Otherwise the truncation may lead to unexpected results."  # noqa: E501
        )
        mock_info.assert_not_called()
    gc.collect()


def test_beam_search_for_T5_executor():
    """Test the beam search for T5 ModelExecutor."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)
    hyperparameter_choices = {"generate_strategy": "beam", "num_beams": 4}

    test_input = "Translate French to English: cher"
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


def test_greedy_search_for_T5_executor():
    """Test the greedy search for T5 ModelExecutor."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "greedy",
    }

    test_input = "Translate French to English: cher"
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


def test_top_k_sampling_for_T5_executor():
    """Test the top_k sampling for T5 ModelExecutor."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "top_k",
        "top_k": 20,
    }

    test_input = "Translate French to English: cher"
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


def test_top_p_sampling_for_T5_executor():
    """Test the top_p sampling for T5 ModelExecutor."""
    T5_model_and_tokenizer = create_t5_model_and_tokenizer()
    T5_model = T5_model_and_tokenizer.model
    T5_tokenizer = T5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(T5_model, T5_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "top_p",
        "top_p": 0.7,
    }

    test_input = "Translate French to English: cher"
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


def test_intersect_sampling_for_T5_executor():
    """Test the intersect sampling for T5 ModelExecutor."""
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    model_executor = GenerationModelExecutor(t5_model, t5_tokenizer)
    hyperparameter_choices = {
        "generate_strategy": "intersect",
        "top_p": 0.7,
        "top_k": 20,
    }

    test_input = "Translate French to English: cher"
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
