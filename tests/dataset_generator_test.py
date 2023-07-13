"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import os
import tempfile
from functools import partial
from unittest.mock import patch

import pytest

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import mock_openai_response

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_openai_response,
    content='{"sample": "This is a great movie!", "annotation": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_openai_response,
    content='{"sample": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_openai_response,
    content='{"sample": "This is a great movie!", "annotation": "1}',
)


class UNKNOWN_GPT3_EXCEPTION(Exception):
    """This is a newly-defined exception for testing purposes."""

    pass


def check_generate_dataset(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_split()` function of `OpenAIDatasetGenerator`.

    This function generates a Dataset for a specified split of the data
    (train, validation, or test) using a simple prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example is not empty.

    Args:
        dataset_generator: The dataset_generator will be tested with limited
            max_api_calls or unlimited max_api_calls.
    """
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    split = DatasetSplit.TRAIN
    num_examples = 5
    # if num_examples >= max_api_calls, the returned dataset's
    # length will be less or equal than max_api_calls.
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert len(dataset) <= num_examples
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns


def check_generate_dataset_dict(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_dict()` function of `OpenAIDatasetGenerator`.

    This function generates movie comments datasets by creating a specified number of
    examples for each split of the data, which includes train, validation, and test.
    It uses a simple prompt specification and saves the generated datasets to a
    temporary directory. Afterward, the function checks whether the dataset dictionary
    contains all the expected keys, each split has the anticipated number of examples,
    every dataset has the anticipated columns, each example is not empty, and whether
    the dataset dictionary is saved to the output directory.

    Args:
        dataset_generator: The dataset_generator will be tested with limited
            max_api_calls or unlimited max_api_calls.
    """
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = {DatasetSplit.TRAIN: 3, DatasetSplit.VAL: 2, DatasetSplit.TEST: 1}
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_dataset_dict(
            prompt_spec=prompt_spec, num_examples=num_examples, output_dir=output_dir
        )

        assert set(dataset_dict.keys()) == {"train", "val", "test"}
        for split, num in num_examples.items():
            assert len(dataset_dict[split.value]) <= num, (
                f"Expected less than {num} examples for {split.value} split, but"
                f" got {len(dataset_dict[split.value])}"
            )
        expected_columns = {"input_col", "output_col"}
        for dataset in dataset_dict.values():
            assert (
                set(dataset.column_names) == expected_columns
            ), f"Expected columns {expected_columns}, but got {dataset.column_names}"
        assert os.path.isdir(output_dir)
        assert set(os.listdir(output_dir)) == {
            "dataset_dict.json",
            "test",
            "train",
            "val",
        }


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
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    unlimited_dataset_generator = OpenAIDatasetGenerator()
    check_generate_dataset_dict(unlimited_dataset_generator)
    check_generate_dataset(unlimited_dataset_generator)
    assert mocked_generate_example.call_count == 11
    limited_dataset_generator = OpenAIDatasetGenerator(max_api_calls=3)
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
    api_key = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
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
    api_key = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
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
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with pytest.raises(UNKNOWN_GPT3_EXCEPTION):
        dataset_generator = OpenAIDatasetGenerator(max_api_calls=3)
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
        num_examples = 1
        split = DatasetSplit.TEST
        _ = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 1


def test_openai_key_init():
    """Test openai key initialization."""
    api_key = None
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(AssertionError) as exc_info:
        _ = OpenAIDatasetGenerator()
        assert str(exc_info.value) == (
            "API key must be provided or set the environment variable"
            + " with `export OPENAI_API_KEY=<your key>`"
        )
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    environment_key_generator = OpenAIDatasetGenerator()
    assert environment_key_generator.api_key == os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = ""
    api_key = "qwertwetyriutytwreytuyrgtwetrueytttr"
    explicit_api_key_generator = OpenAIDatasetGenerator(api_key)
    assert explicit_api_key_generator.api_key == api_key
