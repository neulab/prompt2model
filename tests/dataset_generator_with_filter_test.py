"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import os
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path
from unittest.mock import patch

import datasets
import pytest
from datasets import Dataset

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import Example, OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import (
    MockBatchDifferentCompletions,
    UnknownGpt3Exception,
    are_dataset_dicts_identical,
    are_datasets_identical,
    mock_batch_openai_response_identical_completions,
)

# Create partial functions to simulate different API responses.
# MOCK_EXAMPLE: Represents a mock example with identical completions.
# The content contains an input ("6") and the corresponding output ("f").
MOCK_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "6", "output": "f"}',
)

# MOCK_WRONG_KEY_EXAMPLE: Represents a mock example with identical completions,
# but the content contains an incorrect key "label" instead of "output".
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)

# MOCK_INVALID_JSON: Represents a mock example with an invalid JSON content.
# The content is missing a closing double-quote for the "output" value.
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_WRONG_KEY_EXAMPLE,
)
def test_wrong_key_example(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dictionary.

    This test case is designed to verify the behavior of OpenAIDatasetGenerator
    when the ChatGPTAgent returns a dictionary with a wrong key, i.e., "label" instead
    of "output".

    Args:
        mocked_generate_example: The function represents the @patch function and
        provides the mocked behavior for API calls.

    Note: The test function assumes the existence of 'MOCK_WRONG_KEY_EXAMPLE',
    which represents a mock example with identical completions but an incorrect key
    in the content.

    """
    api_key = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            api_key, 3, filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a mock prompt specification.
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)

        # Set the expected number of examples and dataset split for testing.
        expected_num_examples = 1
        split = DatasetSplit.TRAIN

        # Generate the dataset split using OpenAIDatasetGenerator.
        dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions to verify the test results.
        assert mocked_generate_example.call_count == 3
        assert (
            dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_INVALID_JSON,
)
def test_invalid_json_response(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns an invalid JSON response.

    This test case is designed to verify the behavior of OpenAIDatasetGenerator
    when the ChatGPTAgent returns a response with invalid JSON content. The @patch
    decorator replaces the 'generate_batch_openai_chat_completion' function with
    the 'MOCK_INVALID_JSON' side effect.

    Args:
        mocked_generate_example: The function represents the @patch function and
        provides the mocked behavior for API calls.

    Note: The test function assumes the existence of 'MOCK_INVALID_JSON',
    which represents a mock example with an invalid JSON content.

    """
    api_key = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            api_key, 3, filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a mock prompt specification.
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)

        # Set the expected number of examples and dataset split for testing.
        expected_num_examples = 1
        split = DatasetSplit.VAL

        # Generate the dataset split using OpenAIDatasetGenerator.
        dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions to verify the test results.
        assert mocked_generate_example.call_count == 3
        assert (
            dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=UnknownGpt3Exception(),
)
def test_unexpected_examples_of_gpt(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns an unknown GPT-3 exception.

    This test case is designed to verify the behavior of OpenAIDatasetGenerator
    when the ChatGPTAgent returns an unknown GPT-3 exception. The @patch decorator
    replaces the 'generate_batch_openai_chat_completion' function with the
    'UnknownGpt3Exception' side effect, simulating an unexpected exception.

    Args:
        mocked_generate_example: The function represents the @patch function and
        provides the mocked behavior for API calls.

    Note: The test function assumes the existence of 'UnknownGpt3Exception',
    which represents an unknown GPT-3 exception raised during API calls.

    """
    api_key = "fake_api_key"

    # Set the fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize the OpenAIDatasetGenerator with `max_api_calls = 3`.
    # Use pytest.raises() to assert that an UnknownGpt3Exception is raised.
    with pytest.raises(
        UnknownGpt3Exception
    ), tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            max_api_calls=3, filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a mock prompt specification.
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)

        # Set the expected number of examples and dataset split for testing.
        expected_num_examples = 1
        split = DatasetSplit.TEST

        # Generate the dataset split using OpenAIDatasetGenerator and expect the
        # unknown GPT-3 exception to be raised.
        _ = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

    # Assertions to verify the test results.
    assert mocked_generate_example.call_count == 1

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_openai_key_init():
    """Test OpenAIDatasetGenerator API key initialization.

    This test case verifies the behavior of OpenAIDatasetGenerator when initializing
    the API key. It checks different scenarios, including the absence of the API key,
    setting the API key through the environment variable, and explicitly providing
    the API key as an argument during initialization.

    """
    api_key = None

    # Test case when the API key is not provided or set in the environment variable.
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(
        AssertionError
    ) as exc_info, tempfile.TemporaryDirectory() as cache_dir:
        _ = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        assert str(exc_info.value) == (
            "API key must be provided or set the environment variable"
            + " with `export OPENAI_API_KEY=<your key>`"
        )

    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Test case when the API key is provided through the environment variable.
    with tempfile.TemporaryDirectory() as cache_dir:
        environment_key_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )

    # Assertions to verify that the API key is
    # initialized from the environment variable.
    assert environment_key_generator.api_key == os.environ["OPENAI_API_KEY"]

    # Reset the API key in the environment variable.
    os.environ["OPENAI_API_KEY"] = ""

    # Set a fake API key explicitly for testing purposes.
    api_key = "qwertwetyriutytwreytuyrgtwetrueytttr"

    # Test case when the API key is provided
    # explicitly as an argument during initialization.
    with tempfile.TemporaryDirectory() as cache_dir:
        explicit_api_key_generator = OpenAIDatasetGenerator(
            api_key, cache_root=cache_dir
        )

    # Assertions to verify that the API key is initialized explicitly.
    assert explicit_api_key_generator.api_key == api_key

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_construct_map_with_duplicate_inputs_unique_outputs():
    """Test constructing a map with duplicate inputs but unique outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when there are duplicate inputs but
    unique outputs in the generated examples.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with duplicate inputs and unique outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map afte
        # r constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_duplicate_inputs_duplicate_outputs():
    """Test constructing a map with duplicate inputs and duplicate outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when there are duplicate inputs and
    duplicate outputs in the generated examples.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with
        # duplicate inputs and duplicate outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="C"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="G"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="F"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map after
        # constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when all generated examples have unique
    inputs and outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with unique inputs and outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map after
        # constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_empty_examples_list():
    """Test constructing a map with an empty list of inputs and outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when no generated examples are available.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create an empty list of generated examples.
        generated_examples = []

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The input-output map should be empty
        # when there are no generated examples.
        assert input_output_map == {}

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_multi_vote_with_duplicate_inputs_unique_outputs():
    """Test multi-voting with duplicate inputs but unique outputs.

    This test case verifies the application of multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of
    OpenAIDatasetGenerator. It specifically tests the scenario when
    the input-output map contains duplicate inputs but unique outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide an input-output map with duplicate inputs but unique outputs.
        input_output_map = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_duplicate_inputs_duplicate_outputs():
    """Test multi-voting with duplicate inputs and duplicate outputs.

    This test case verifies the application of multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of
    OpenAIDatasetGenerator. It specifically tests the scenario when
    the input-output map contains duplicate inputs and duplicate outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide an input-output map with duplicate inputs and duplicate outputs.
        input_output_map = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_unique_inputs_outputs():
    """Test multi-voting with unique inputs and outputs.

    This test case verifies the application of the multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of OpenAIDatasetGenerator.
    It specifically tests the scenario when the input-output map contains unique
    inputs and outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with an empty input-output map.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)

        # Provide an input-output map with unique inputs and outputs.
        input_output_map = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_empty_examples_list():
    """Test multi-voting with empty inputs and outputs.

    This test case verifies the application of the multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of OpenAIDatasetGenerator.
    It specifically tests the scenario when the input-output map is empty.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with an empty input-output map.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )

        # Set the input-output map to be empty.
        input_output_map = {}

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting (empty dataset).
        expected_dataset = Dataset.from_dict({})

        # Verify that the generated dataset matches
        # the expected dataset (empty dataset).
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_duplicate_inputs_unique_outputs():  # noqa 501
    """Test constructing generated dataset with duplicate inputs but unique outputs.

    This test case verifies the construction of the generated dataset with duplicate
    inputs but unique outputs. The OpenAIDatasetGenerator object is initialized with
    `filter_duplicated_examples=True` to ensure that duplicates are filtered.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with `filter_duplicated_examples=True`.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide generated examples with duplicate inputs but unique outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
        ]

        # Convert the generated examples to the generated dataset.
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )

        # Define the expected dataset after conversion (duplicates are filtered).
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        expected_all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)
        assert are_datasets_identical(
            all_generated_examples_dataset, expected_all_generated_examples_dataset
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_duplicate_inputs_duplicate_outputs():  # noqa 501
    """Test constructing a map with duplicate inputs and duplicate outputs.

    This test case verifies the construction of the generated dataset with duplicate
    inputs and duplicate outputs. The OpenAIDatasetGenerator object is initialized with
    `filter_duplicated_examples=True` to ensure that duplicates are filtered.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with `filter_duplicated_examples=True`.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide generated examples with duplicate inputs and duplicate outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="C"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="G"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="F"),
        ]

        # Convert the generated examples to the generated dataset.
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )

        # Define the expected dataset after conversion (duplicates are filtered).
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        expected_all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)
        assert are_datasets_identical(
            all_generated_examples_dataset, expected_all_generated_examples_dataset
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs.

    This test case verifies the construction of the generated dataset with unique
    inputs and outputs. The OpenAIDatasetGenerator object is initialized with
    `filter_duplicated_examples=True` to ensure that duplicates are filtered.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with `filter_duplicated_examples=True`.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide generated examples with unique inputs and outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
        ]

        # Convert the generated examples to the generated dataset.
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )

        # Define the expected dataset after conversion (no duplicates to filter).
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        expected_all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)
        assert are_datasets_identical(
            all_generated_examples_dataset, expected_all_generated_examples_dataset
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_empty_examples_list():
    """Test constructing a map with empty inputs and outputs.

    This test case verifies the construction of the generated dataset when the
    generated_examples list is empty. The OpenAIDatasetGenerator object is initialized
    with `filter_duplicated_examples=True` to ensure that duplicates are filtered.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with `filter_duplicated_examples=True`.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide an empty list of generated examples.
        generated_examples = []

        # Convert the empty generated examples to the generated dataset.
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )

        # Define the expected dataset (empty dataset when there are no examples).
        expected_dataset = Dataset.from_dict({})

        expected_all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)
        assert are_datasets_identical(
            all_generated_examples_dataset, expected_all_generated_examples_dataset
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_load_cache_dataset_with_filter_duplicated_examples():
    """Test the cached dataset loading with filtering duplicated examples.

    This test case verifies the loading of the cached dataset and its filtering
    to eliminate duplicated examples. The OpenAIDatasetGenerator object is
    initialized with `filter_duplicated_examples=True`.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set up a temporary directory for cache.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )

        # Create a cached dataset and save it to the disk.
        examples_cache_path = Path(
            data_generator.cache_root / f"generated_examples_{DatasetSplit.TEST.value}"
        )
        cached_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "1", "1", "1", "2", "3"],
                "output_col": ["a", "a", "b", "c", "a", "d"],
            }
        )
        cached_dataset.save_to_disk(examples_cache_path)

        # The generate_dataset_split would first load the cached dataset into
        # generated_examples. Then, in the while loop,
        # create_all_examples_dataset_and_generated_dataset would be called to
        # construct the generated_dataset. Note that filter_duplicated_examples
        # is True, so the generated_examples will be filtered to 3 examples
        # in generated_dataset. Since expected_num_examples is 3, the while loop
        # would exit immediately. So the generated_dataset would be the filtered
        # cached dataset.
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            generated_dataset = data_generator.generate_dataset_split(
                expected_num_examples=3,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )

            # Verify that logging.info was called with the correct message.
            mock_info.assert_called_once_with(
                f"Loading cache from {str(examples_cache_path)}."
            )
            mock_warning.assert_not_called()

        # Define the expected filtered dataset after loading the cache.
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "d"],
            }
        )

        # Verify that the generated dataset matches the expected filtered dataset.
        assert are_datasets_identical(generated_dataset, excepted_generated_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_EXAMPLE,
)
def test_load_cache_dataset_with_filter_duplicated_examples_and_continue_generation(
    mocked_generate_example,
):
    """Test OpenAIDatasetGenerator can load cache and continue generation.

    This test case verifies the ability of OpenAIDatasetGenerator to
    load a cached dataset and continue generation when
    `filter_duplicated_examples` is True. The OpenAIDatasetGenerator
    object is initialized with `filter_duplicated_examples=True`.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set up a temporary directory for cache.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )

        # Create cached examples and save them to the disk.
        examples_cache_path = (
            Path(cache_dir) / f"generated_examples_{DatasetSplit.TEST.value}"
        )
        cached_examples = Dataset.from_dict(
            {
                "input_col": ["1", "1", "1", "1", "2", "3"],
                "output_col": ["a", "a", "b", "c", "a", "d"],
            }
        )
        cached_examples.save_to_disk(examples_cache_path)

        # The generate_dataset_split would first load the cached dataset into
        # generated_examples. Then, in the while loop,
        # create_all_examples_dataset_and_generated_dataset would be called to
        # construct the generated_dataset. Note that filter_duplicated_examples
        # is True, so the generated_examples will be filtered to 3 examples
        # in generated_dataset. Since expected_num_examples is 4, the generation
        # would continue, and the max_batch_size = 1. After one batch of API calls,
        # generated_dataset meets the requirement and stop generation.
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            generated_dataset = data_generator.generate_dataset_split(
                expected_num_examples=4,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )

            # Verify that logging.info was called with
            # the correct message for loading cache.
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert info_list[0] == f"Loading cache from {str(examples_cache_path)}."
            # The first logging.info is for loading cache, and there are
            # 5 * 2 additional logging.info messages in extract_responses.
            assert len(info_list) == 1 + 5 * 2
            mock_warning.assert_not_called()

        # Define the expected generated dataset after continuing generation.
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3", "6"],
                "output_col": ["a", "a", "d", "f"],
            }
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, excepted_generated_dataset)

        # Verify that the API was called once to generate responses.
        assert mocked_generate_example.call_count == 1

    # Collect garbage to release memory resources after the test.
    gc.collect()


"""
These tests validate the generation process with `filter_duplicated_examples=True`.

These tests collaborate with the `MockBatchDifferentCompletions().mock_completions`
function to imitate the generation process of the OpenAIDataSetGenerator.

The first five tests check the generation of a single dataset split using
a shared OpenAIDataSetGenerator with the following settings:
    - max_batch_size = 2
    - responses_per_request = 3
    - filter_duplicated_examples = True
    - expected_num_examples = 5

In the first API call, the generator produces 2 * 3 = 6 responses.
After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2"],
        "output_col": ["a", "a"],
    })

max_batch_size = (expected_num_examples - len(generated_dataset))
/ responses_per_request = (5 - 2) / 3 = 1.

The second API call reduces max_batch_size to 1 and generates 3 more responses.
After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3"],
        "output_col": ["a", "a", "a"],
    })

The third API call again uses max_batch_size = 1 and generates another 3 responses.
After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3"],
        "output_col": ["b", "a", "a"],
    })

The fourth API call also uses max_batch_size = 1 and generates 3 responses.
After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3", "4", "5"],
        "output_col": ["b", "a", "a", "c", "a"],
    })

The test suite contains five test cases, each using a different OpenAIDataSetGenerator.
These generators have the same settings (max_batch_size = 2, responses_per_request = 3,
expected_num_examples = 5, filter_duplicated_examples = True), but their max_api_calls
attribute is set to 2, 3, 4, 5, and unlimited, respectively.

Each test runs the generation of its generator and verifies that the
generated dataset matches the expected result.
"""


api_key = "fake_api_key"
prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
split = DatasetSplit.TRAIN
filter_duplicated_examples = True
expected_num_examples = 5
max_batch_size = 2
responses_per_request = 3


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_first_batch(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods in the first batch.

    This test verifies the behavior of the OpenAIDatasetGenerator with
    filter methods in the first batch of API calls. It initializes an
    OpenAIDatasetGenerator with specific settings, limiting the number
    of API calls to 2. After running the generation process, the test
    checks whether the generated dataset matches the expected
    result after the second API call. The test also ensures that the
    number of calls to the API mock matches the expected number.

    Note: The first API call's max_batch_size is 2, generating 6 responses.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Initialize the OpenAIDatasetGenerator with specific settings.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            max_api_calls=2,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
        )

        # Generate the dataset split using the initialized generator.
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions for API call count and dataset matching the expected result.
        assert mocked_generate_example.call_count == 1
        assert dataset_generator.api_call_counter == 2

        # Define the expected dataset based on the given mock responses.
        expected_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2"],
                "output_col": ["a", "a"],
            }
        )

        # Verify the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_second_batch(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods in the second batch.

    This test verifies the behavior of the OpenAIDatasetGenerator with filter
    methods in the second batch of API calls. It initializes an
    OpenAIDatasetGenerator with specific settings, limiting the number of
    API calls to 3. After running the generation process, the test checks
    whether the generated dataset matches the expected result after the
    second API call. The test also ensures that the number of calls to the
    API mock matches the expected number.

    Note: The first API call's max_batch_size is 2, generating 6 responses.
    The second API call's max_batch_size is 1, generating 3 responses.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Initialize the OpenAIDatasetGenerator with specific settings.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            max_api_calls=3,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
        )

        # Generate the dataset split using the initialized generator.
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions for API call count and dataset matching the expected result.
        assert mocked_generate_example.call_count == 2
        assert dataset_generator.api_call_counter == 3

        # Define the expected dataset based on the given mock responses.
        expected_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "a"],
            }
        )

        # Verify the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_third_batch(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods in the third batch.

    This test verifies the behavior of the OpenAIDatasetGenerator with
    filter methods in the third batch of API calls. It initializes an
    OpenAIDatasetGenerator with specific settings, limiting the number
    of API calls to 4. After running the generation process, the test
    checks whether the generated dataset matches the expected
    result after the third API call. The test also ensures that the
    number of calls to the API mock matches the expected number.

    Note: The first API call's max_batch_size is 2, generating 6 responses.
    The second API call's max_batch_size is 1, generating 3 responses.
    The third API call's max_batch_size is 1, generating 3 responses.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Reset the mock responses to ensure predictable behavior.

        # Initialize the OpenAIDatasetGenerator with specific settings.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            max_api_calls=4,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
        )

        # Generate the dataset split using the initialized generator.
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions for API call count and dataset matching the expected result.
        assert mocked_generate_example.call_count == 3
        assert dataset_generator.api_call_counter == 4

        # Define the expected dataset based on the given mock responses.
        expected_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["b", "a", "a"],
            }
        )

        # Verify the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_forth_batch(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods in the forth batch.

    This test verifies the behavior of the OpenAIDatasetGenerator with
    filter methods in the forth batch of API calls. It initializes an
    OpenAIDatasetGenerator with specific settings, limiting the number
    of API calls to 5. After running the generation process, the test checks
    whether the generated dataset matches the expected result after the
    forth API call. The test also ensures that the number of calls to the
    API mock matches the expected number.

    Note: The first API call's max_batch_size is 2, generating 6 responses.
    The second API call's max_batch_size is 1, generating 3 responses.
    The third API call's max_batch_size is 1, generating 3 responses.
    The forth and last API call's max_batch_size is 1, generating 3 responses.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Reset the mock responses to ensure predictable behavior.

        # Initialize the OpenAIDatasetGenerator with specific settings.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            max_api_calls=5,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
        )

        # Generate the dataset split using the initialized generator.
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions for API call count and dataset matching the expected result.
        assert mocked_generate_example.call_count == 4
        assert dataset_generator.api_call_counter == 5

        # Define the expected dataset based on the given mock responses.
        expected_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3", "4", "5"],
                "output_col": ["b", "a", "a", "c", "a"],
            }
        )

        # Verify the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_unlimited_api_calls(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods and unlimited API calls.

    This test verifies the behavior of the OpenAIDatasetGenerator with
    filter methods and unlimited API calls. It initializes the generator
    with specific settings but does not limit the number of API calls.
    After running the generation process, the test checks whether
    the generated dataset matches the expected result after the
    final API call. The test also ensures that the number of calls
    to the API mock matches the expected number.

    Note: The first API call's max_batch_size is 2, generating 6 responses.
    The second API call's max_batch_size is 1, generating 3 responses.
    The third API call's max_batch_size is 1, generating 3 responses.
    The forth and last API call's max_batch_size is 1, generating 3 responses.
    After the forth batch, the generation ends. No further API calls are required.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Reset the mock responses to ensure predictable behavior.

        # Initialize the OpenAIDatasetGenerator with
        # specific settings and unlimited API calls.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
        )

        # Generate the dataset split using the initialized generator.
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )

        # Assertions for API call count and dataset matching the expected result.
        assert mocked_generate_example.call_count == 4
        assert dataset_generator.api_call_counter == 5

        # Define the expected dataset based on the given mock responses.
        expected_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3", "4", "5"],
                "output_col": ["b", "a", "a", "c", "a"],
            }
        )

        # Verify the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MockBatchDifferentCompletions(length=5).mock_completions,
)
def test_generator_with_filter_to_generate_datasetdict(mocked_generate_example):
    """Test OpenAIDatasetGenerator with filter methods to generate a DatasetDict.

    This test checks the generation of a DatasetDict using the OpenAIDatasetGenerator
    with filter methods. It initializes the generator with specific settings and
    expected_num_examples for each split. The test verifies the generated dataset
    dictionaries for the train, val, and test splits match the expected results.

    The generation involves multiple API calls, and filtering duplicates is applied to
    the generated examples at each step. The API calls continue until the number of
    generated examples meets the expected_num_examples for each split or reaches the
    maximum allowed API calls.

    Args:
        mocked_generate_example (MagicMock): The patched function representing the
            @patch decorator for generating example responses.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        # Reset the mock responses to ensure predictable behavior.

        # Initialize the OpenAIDatasetGenerator with
        # specific settings and limited API calls.
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            max_batch_size=max_batch_size,
            responses_per_request=responses_per_request,
            max_api_calls=7,
        )

        # Generate the DatasetDict using the initialized generator.
        generated_dataset_dict = dataset_generator.generate_dataset_dict(
            prompt_spec,
            expected_num_examples={
                DatasetSplit.TRAIN: 4,
                DatasetSplit.VAL: 4,
                DatasetSplit.TEST: 2,
            },
        )

        # Assertions for API call count and dataset
        # dictionaries matching the expected results.
        assert mocked_generate_example.call_count == 5
        assert dataset_generator.api_call_counter == 7

        # Define the expected dataset dictionaries
        # based on the given mock responses.
        expected_dataset_dict = datasets.DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "input_col": ["1", "2", "3", "4", "5"],
                        "output_col": ["b", "a", "a", "c", "a"],
                    }
                ),
                "val": Dataset.from_dict(
                    {
                        "input_col": ["1", "2"],
                        "output_col": ["a", "a"],
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "input_col": [],
                        "output_col": [],
                    }
                ),
            }
        )

        # Verify the generated DatasetDict matches the expected DatasetDict.
        assert are_dataset_dicts_identical(
            generated_dataset_dict, expected_dataset_dict
        )

    # Collect garbage to release memory resources after the test.
    gc.collect()
