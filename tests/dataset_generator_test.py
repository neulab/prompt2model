"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import os
import tempfile
from functools import partial
from unittest.mock import patch

import pytest

from prompt2model.utils.tevatron_utils import (  # noqa: F401
    encode_search_corpus, encode_text, retrieve_objects)
from prompt2model.prompt_parser import MockPromptSpec


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_encode_text(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    This function first test the unlimited generation. Then test generation
    when num_examples >= max_api_calls. Thus the API agent will only be
    called max_api_calls times.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    api_key = None
    unlimited_dataset_generator = OpenAIDatasetGenerator(api_key)
    check_generate_dataset_dict(unlimited_dataset_generator)
    check_generate_dataset(unlimited_dataset_generator)
    assert mocked_generate_example.call_count == 11
    limited_dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    check_generate_dataset(limited_dataset_generator)
    assert mocked_generate_example.call_count == 14
    limited_dataset_generator.api_call_counter = 0
    # refresh the api_call_counter of limited_dataset_generator for futher test.
    check_generate_dataset_dict(limited_dataset_generator)
    assert mocked_generate_example.call_count == 17


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=MOCK_WRONG_KEY_EXAMPLE,
)
def test_wrong_key_example(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    api_key = None
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec()
    num_examples = 1
    split = DatasetSplit.TRAIN
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 3
    assert dataset["input_col"] == dataset["output_col"] == []


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=MOCK_INVALID_JSON,
)
def test_invalid_json_response(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    api_key = None
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec()
    num_examples = 1
    split = DatasetSplit.VAL
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 3
    assert dataset["input_col"] == dataset["output_col"] == []


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=UNKNOWN_GPT3_EXCEPTION(),
)
def test_unexpected_examples_of_GPT(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    api_key = None
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with pytest.raises(UNKNOWN_GPT3_EXCEPTION):
        dataset_generator = OpenAIDatasetGenerator(api_key, 3)
        prompt_spec = MockPromptSpec()
        num_examples = 1
        split = DatasetSplit.TEST
        _ = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 1
