"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import logging
import os
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import Example, OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import (
    MockCompletion,
    UnknownGpt3Exception,
    are_datasets_identical,
    mock_batch_openai_response_identical_completions,
)

logger = logging.getLogger("DatasetGenerator")

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)


def check_generate_dataset(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_split()` function of `OpenAIDatasetGenerator`.

    This function generates a Dataset for a specified split of the data
    (train, validation, or test) using a simple prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example is not empty.

    Args:
        dataset_generator: The dataset_generator will be tested
            with limited max_api_calls or unlimited max_api_calls.
    """
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    split = DatasetSplit.TRAIN
    expected_num_examples = 29
    # If expected_num_examples >= max_api_calls, the returned dataset's
    # length will be less than or equal to max_api_calls.
    dataset = dataset_generator.generate_dataset_split(
        prompt_spec, expected_num_examples, split
    )
    # Since each API call would return one completion object with 5 responses
    # and some of the responses are invalid JSON objects, the upper bound of
    # the length of the dataset is expected_num_examples + 5, where 5 is the
    # default number of responses per API call.
    assert len(dataset) < expected_num_examples + 5
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns
    return dataset


def check_generate_dataset_dict(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_dict()` function of `OpenAIDatasetGenerator`.

        This function generates movie comments datasets by creating a specified
        number of examples for each split of the data, which includes train,
        validation, and test. It uses a simple prompt specification and saves the
        generated datasets to a temporary directory. Afterward, the function
        checks whether the dataset dictionary contains all the expected keys,
        each split has the anticipated number of examples, every dataset has
        the anticipated columns, each example is not empty, and whether
        the dataset dictionary is saved to the output directory.

    Args:
        dataset_generator: The dataset_generator will be tested
            with limited max_api_calls or unlimited max_api_calls.
    """
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    expected_num_examples = {
        DatasetSplit.TRAIN: 50,
        DatasetSplit.VAL: 24,
        DatasetSplit.TEST: 26,
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_dataset_dict(
            prompt_spec=prompt_spec,
            expected_num_examples=expected_num_examples,
            output_dir=output_dir,
        )

        assert set(dataset_dict.keys()) == {"train", "val", "test"}
        for split, num in expected_num_examples.items():
            # As explained previously, the upper bound of the length of
            # generated dataset is expected_num_examples + 5, where
            # 5 is the default number of responses per API call.
            assert len(dataset_dict[split.value]) < num + 5
        expected_columns = {"input_col", "output_col"}
        for dataset in dataset_dict.values():
            assert set(dataset.column_names) == expected_columns
        assert os.path.isdir(output_dir)
        assert set(os.listdir(output_dir)) == {
            "dataset_dict.json",
            "test",
            "train",
            "val",
        }
    return dataset_dict


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generator_without_filter(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    This function first tests unlimited generation. Then, it tests generation
    when expected_num_examples >= max_api_calls. In the second test, the API agent
    will only be called max_api_calls times.

    Args:
        mocked_generate_example: The function representing the @patch function.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        unlimited_dataset_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        unlimited_generated_dataset = check_generate_dataset(
            unlimited_dataset_generator
        )
        # The default responses_per_request is 5. So each API call will return
        # 5 responses, i.e. 5 choices in openai.Completion.choices.
        # Each API call will return 5 responses, and each response is a valid JSON.
        # So the unlimited_dataset_generator will call the API (29 // 5 + 1) times.
        assert unlimited_dataset_generator.api_call_counter == (29 // 5 + 1)
        # The default batch_size is 5. So generate_batch_openai_chat_completion
        # will be called 2 times with  first batch_size = 5 and second batch_size = 1.
        assert mocked_generate_example.call_count == 2
        # Since all the responses are valid JSON and the api_call_counter is 6,
        # the unlimited_generated_dataset will contain 30 examples.
        assert len(unlimited_generated_dataset) == 30

    # Refresh the call_count and dataset_generator.
    with tempfile.TemporaryDirectory() as cache_dir:
        mocked_generate_example.call_count = 0
        unlimited_dataset_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        unlimited_generated_dataset_dict = check_generate_dataset_dict(
            unlimited_dataset_generator
        )

        # Each API call returns five responses. So the unlimited_dataset_generator will
        # call the API (50 // 5 + 24 // 5 + 1 + 26 // 5 + 1) = 21 times.
        assert unlimited_dataset_generator.api_call_counter == (
            50 // 5 + 24 // 5 + 1 + 26 // 5 + 1
        )
        # The default batch_size is 5. So generate_batch_openai_chat_completion
        # will be called 2 times for 50 examples in the train split,
        # 1 time for 24 examples in the validation split,
        # and 2 times for 26 examples in the test split.
        assert mocked_generate_example.call_count == 2 + 1 + 2

        # Each API call returns 5 responses, and each response is a valid JSON.
        # So the unlimited_generated_dataset_dict will contain (50, 25, 30) examples.
        assert len(unlimited_generated_dataset_dict["train"]) == 50
        assert len(unlimited_generated_dataset_dict["val"]) == 25
        assert len(unlimited_generated_dataset_dict["test"]) == 30

    with tempfile.TemporaryDirectory() as cache_dir:
        # Refresh the call_count.
        mocked_generate_example.call_count = 0

        limited_dataset_generator = OpenAIDatasetGenerator(
            max_api_calls=3, filter_duplicated_examples=False, cache_root=cache_dir
        )
        limited_generated_dataset = check_generate_dataset(limited_dataset_generator)
        # The max_api_calls is 3. So the limited_dataset_generator calls the
        # API 3 times. Each API call returns 5 responses. So the
        # limited_dataset_generator will have 3 * 5 = 15 examples.
        assert len(limited_generated_dataset) == 15

        # The default batch_size is 5. So generate_batch_openai_chat_completion
        # will be called only once.
        assert mocked_generate_example.call_count == 1

        # Each API call returns 5 responses, so the limited_dataset_generator
        # will use up all the available API calls.
        assert limited_dataset_generator.api_call_counter == 3

        # Each API call returns 5 responses, and each response is a valid JSON.
        # So the limited_generated_dataset will contain 15 examples.
        assert len(limited_generated_dataset) == 15

    with tempfile.TemporaryDirectory() as cache_dir:
        # Refresh the call_count and create a new limited_dataset_generator.
        mocked_generate_example.call_count = 0
        limited_dataset_generator = OpenAIDatasetGenerator(
            max_api_calls=13, filter_duplicated_examples=False, cache_root=cache_dir
        )

        limited_generated_dataset_dict = check_generate_dataset_dict(
            limited_dataset_generator
        )
        # Since the max_api_calls is 13, the limited_dataset_generator cannot
        # generate the whole dataset_dict and will call the API 13 times.
        assert limited_dataset_generator.api_call_counter == 13

        # The train split has 50 examples, so it will call the API 10 times and call
        # generate_batch_openai_chat_completion 2 times.
        # The validation split has 24 examples, but there are only 3 API calls
        # left, so it will call the API 3 times and call
        # generate_batch_openai_chat_completion 1 time.
        # The test split has 26 examples, but there are no more API calls left,
        # so it will not call generate_batch_openai_chat_completion.
        assert mocked_generate_example.call_count == 2 + 1 + 0

        # Each API call returns 5 responses, and each response is a valid JSON.
        # So the generated_dataset_dict will contain (50, 15, 0) examples.
        assert len(limited_generated_dataset_dict["train"]) == 50
        assert len(limited_generated_dataset_dict["val"]) == 15
        assert len(limited_generated_dataset_dict["test"]) == 0
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_WRONG_KEY_EXAMPLE,
)
def test_wrong_key_example(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a dictionary with wrong keys.

    Args:
        mocked_generate_example: The function representing the @patch function.
    """
    api_key = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            api_key, 3, filter_duplicated_examples=False, cache_root=cache_dir
        )
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
        expected_num_examples = 1
        split = DatasetSplit.TRAIN
        generated_dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )
        assert mocked_generate_example.call_count == 3
        expected_dataset = Dataset.from_dict({"input_col": [], "output_col": []})
        assert are_datasets_identical(expected_dataset, generated_dataset)
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_INVALID_JSON,
)
def test_invalid_json_response(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns invalid JSON responses.

    Args:
        mocked_generate_example: The function representing the @patch function.
    """
    api_key = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            api_key, 3, filter_duplicated_examples=False, cache_root=cache_dir
        )
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
        expected_num_examples = 1
        split = DatasetSplit.VAL
        dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )
        assert mocked_generate_example.call_count == 3
        expected_dataset = Dataset.from_dict({"input_col": [], "output_col": []})
        assert are_datasets_identical(dataset, expected_dataset)
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=UnknownGpt3Exception(),
)
def test_unexpected_examples_of_gpt(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns unexpected examples.

    This function tests the scenario when the agent raises an UnknownGpt3Exception
    during dataset generation. The test ensures that the exception is correctly raised.

    Args:
        mocked_generate_example: The function representing the @patch function.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with pytest.raises(
        UnknownGpt3Exception
    ), tempfile.TemporaryDirectory() as cache_dir:
        dataset_generator = OpenAIDatasetGenerator(
            max_api_calls=3, filter_duplicated_examples=False, cache_root=cache_dir
        )
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
        expected_num_examples = 1
        split = DatasetSplit.TEST
        _ = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )
    assert mocked_generate_example.call_count == 1
    gc.collect()


def test_openai_key_init():
    """Test OpenAI API key initialization.

    This function tests the initialization of the OpenAI API key. It verifies that
    the API key can be provided directly or through the environment variable
    `OPENAI_API_KEY`, and the generator successfully uses the provided API key.

    It tests three cases:
    1. When the API key is not provided and the environment variable is empty, the
       generator should raise an AssertionError.
    2. When the API key is provided through the environment variable, the generator
       should use the provided API key.
    3. When the API key is provided directly, the generator should use the provided
       API key.

    The test uses the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`
    for each case.

    Note: For security reasons, it is recommended to set the API key through the
    environment variable rather than hardcoding it in the code.

    Raises:
        AssertionError: If the API key is not provided, either directly or through the
                        environment variable.
    """
    api_key = None
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(
        ValueError
    ) as exc_info, tempfile.TemporaryDirectory() as cache_dir:
        _ = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        assert str(exc_info.value) == (
            "API key must be provided or set the environment variable"
            + " with `export OPENAI_API_KEY=<your key>`."
        )
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        environment_key_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
    assert environment_key_generator.api_key == os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = ""
    api_key = "qwertwetyriutytwreytuyrgtwetrueytttr"
    with tempfile.TemporaryDirectory() as cache_dir:
        explicit_api_key_generator = OpenAIDatasetGenerator(
            api_key, cache_root=cache_dir
        )
    assert explicit_api_key_generator.api_key == api_key
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_duplicate_inputs_unique_outputs():  # noqa: 501
    """Test constructing the generated dataset with duplicate inputs but unique outputs.

    This function tests the scenario when the generator has generated examples with
    duplicate inputs but unique outputs. It ensures that the generator successfully
    converts the generated examples into a generated dataset while preserving the
    correct mappings between input and output.

    The test uses the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`.
    The `generating_split` attribute of the generator is set to `DatasetSplit.TEST`,
    and the `generated_examples` list contains examples with some duplicate inputs but
    unique outputs.

    The function then calls the `create_all_examples_dataset_and_generated_dataset()`
    method to create the generated dataset.

    Finally, the function checks whether the generated dataset matches the expected
    dataset constructed from the input examples.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the generated dataset does not match the expected dataset.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
        ]
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )
        assert are_datasets_identical(all_generated_examples_dataset, expected_dataset)
        assert are_datasets_identical(generated_dataset, expected_dataset)
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_duplicate_inputs_duplicate_outputs():  # noqa: 501
    """Test constructing a map with duplicate inputs and duplicate outputs.

    This function tests the scenario when the generator has generated examples with
    duplicate inputs and duplicate outputs. It ensures that the generator successfully
    converts the generated examples into a generated dataset while preserving the
    correct mappings between input and output.

    The test uses the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`.
    The `generating_split` attribute of the generator is set to `DatasetSplit.TEST`,
    and the `generated_examples` list contains examples with both duplicate inputs and
    duplicate outputs. The function then calls the
    `create_all_examples_dataset_and_generated_dataset()` method to create the generated
    dataset.

    Finally, the function checks whether the generated dataset matches the expected
    dataset constructed from the input examples.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the generated dataset does not match the expected dataset.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
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
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )
        assert are_datasets_identical(all_generated_examples_dataset, expected_dataset)
        assert are_datasets_identical(generated_dataset, expected_dataset)
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs.

    This function tests the scenario when the generator has generated examples with
    unique inputs and unique outputs. It ensures that the generator successfully
    converts the generated examples into a generated dataset while preserving the
    correct mappings between input and output.

    The test uses the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`.
    The `generating_split` attribute of the generator is set to `DatasetSplit.TEST`,
    and the `generated_examples` list contains examples with unique inputs and
    unique outputs. The function then calls the
    `create_all_examples_dataset_and_generated_dataset()` method to create the generated
    dataset.

    Finally, the function checks whether the generated dataset matches the expected
    dataset constructed from the input examples.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the generated dataset does not match the expected dataset.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
        ]
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )
        assert are_datasets_identical(all_generated_examples_dataset, expected_dataset)
        assert are_datasets_identical(generated_dataset, expected_dataset)
    gc.collect()


def test_create_all_examples_dataset_and_generated_dataset_with_empty_examples_list():
    """Test constructing a map with empty inputs and outputs.

    This function tests the scenario when the generator has an empty list of generated
    examples. It ensures that the generator successfully converts the empty examples
    list into an empty generated dataset.

    The test uses the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`.
    The `generating_split` attribute of the generator is set to `DatasetSplit.TEST`,
    and the `generated_examples` list is empty. The function then calls the
    `create_all_examples_dataset_and_generated_dataset()` method to create the generated
    dataset.

    Finally, the function checks whether the generated dataset is empty.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the generated dataset is not empty.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        generated_examples = []
        (
            all_generated_examples_dataset,
            generated_dataset,
        ) = data_generator.create_all_examples_dataset_and_generated_dataset(
            generated_examples
        )
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )
        assert are_datasets_identical(all_generated_examples_dataset, expected_dataset)
        assert are_datasets_identical(generated_dataset, expected_dataset)
    gc.collect()


def test_compute_batch_size_with_limited_max_api_calls():
    """Test the batch size computation with limited max API calls.

    This function tests the computation of batch size when the generator has limited
    max API calls. It covers scenarios where the API calls are close to reaching the
    maximum limit and when the API calls are far from the maximum limit.

    The test uses the `OpenAIDatasetGenerator` with `max_api_calls=28`. The
    `api_call_counter` attribute of the generator is set to `26`, and the
    `generated_dataset` contains 110 examples. The function then calls the
    `compute_batch_size()` method with an `expected_num_examples` of `125`.

    Finally, the function checks whether the computed batch size matches the expected
    batch size based on the remaining API calls and the number of examples needed to
    reach the expected number of examples.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the computed batch size does
            not match the expected batch size.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(max_api_calls=28, cache_root=cache_dir)
        data_generator.api_call_counter = 26
        generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        # Default batch size and responses_per_request are both 5.
        # So each batch should contain 25 examples.

        # At least (125 - 110) / 5 = 3 API calls needed to get
        # more than 125 examples.

        batch_size = data_generator.compute_batch_size(
            expected_num_examples=125, generated_dataset=generated_dataset
        )
        assert (
            batch_size
            == data_generator.max_api_calls - data_generator.api_call_counter
            == 28 - 26
        )

        data_generator.api_call_counter = 20
        batch_size = data_generator.compute_batch_size(125, generated_dataset)
        assert (
            batch_size
            == ((125 - len(generated_dataset))) / data_generator.responses_per_request
            == (125 - 110) / 5
        )

        data_generator.api_call_counter = 0
        generated_dataset = Dataset.from_dict(
            {
                "input_col": [1] * 50,
                "output_col": [2] * 50,
            }
        )
        batch_size = data_generator.compute_batch_size(125, generated_dataset)
        assert batch_size == data_generator.max_batch_size
    gc.collect()


def test_compute_batch_size_with_unlimited_max_api_calls():
    """Test the batch size computation with unlimited max API calls.

    This function tests the computation of batch size when the generator has unlimited
    max API calls. It covers scenarios where the number of examples needed to reach the
    expected number of examples is greater than the default batch size.

    The test uses the `OpenAIDatasetGenerator` with default `max_api_calls`. The
    `generated_dataset` contains 110 examples. The function then calls the
    `compute_batch_size()` method with an `expected_num_examples` of `125`.

    Finally, the function checks whether the computed batch size matches the expected
    batch size based on the number of examples needed to reach the expected number of
    examples.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the computed batch size
         ddoes not match the expected batch size.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)
        generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        # Default batch size and responses_per_request are both 5.
        # So each batch should contain 25 examples.

        # At least (125 - 110) / 5 = 3 API calls needed to get
        # more than 125 examples.

        batch_size = data_generator.compute_batch_size(125, generated_dataset)
        assert (
            batch_size
            == (125 - len(generated_dataset)) / data_generator.responses_per_request
            == (125 - 110) / 5
        )

        generated_dataset = Dataset.from_dict(
            {
                "input_col": [1] * 50,
                "output_col": [2] * 50,
            }
        )
        batch_size = data_generator.compute_batch_size(125, generated_dataset)
        assert batch_size == data_generator.max_batch_size == 5
    gc.collect()


def test_load_cache_dataset_without_filter_duplicated_examples():
    """Test the cached dataset loading without filtering duplicated examples.

    This function tests the cached dataset loading without filtering duplicated
    examples. It first saves a dataset to the cache directory and then initializes
    the `OpenAIDatasetGenerator` with `filter_duplicated_examples=False`.

    The `generate_dataset_split()` method is then called with an
    `expected_num_examples` of `110`, which is equal to the size of the cached
    dataset. The function checks that the cached dataset is successfully loaded,
    and the generation stops because the expected number of examples is
    already met.

    Note: The test uses a temporary directory as the cache root to ensure that the cache
    directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the cached dataset is not loaded correctly, or if the
            generation does not stop when the expected number of examples is met.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=False
        )
        examples_cache_path = Path(
            data_generator.cache_root / f"generated_examples_{DatasetSplit.TEST.value}"
        )
        cached_examples = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        cached_examples.save_to_disk(examples_cache_path)
        # The generate_dataset_split would first load the cached
        # dataset into generated_examples. Then in the while
        # loop, create_all_examples_dataset_and_generated_dataset
        # would be called to construct the generated_dataset.
        # Note that filter_duplicated_examples is False, so the
        # generated_examples won't be filtered. And since the
        # expected_num_examples is 110, the while loop would exit
        # immediately. So the generated_dataset would be the
        # same as the cached dataset.
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_dataset = data_generator.generate_dataset_split(
                expected_num_examples=110,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )
            mock_info.assert_called_once_with(
                f"Loading cache from {str(examples_cache_path)}."
            )
            mock_warning.assert_not_called()
        assert are_datasets_identical(generated_dataset, cached_examples)
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_load_cache_dataset_without_filter_duplicated_examples_and_continue_generation(
    mocked_generate_example,
):
    """Test OpenAIDatasetGenerator can load cache and continue generation.

    This function tests that the `OpenAIDatasetGenerator` can load the cached
    dataset and continue generation if the expected number of examples is
    greater than the size of the cached dataset. The test first saves a dataset
    to the cache directory and then initializes the `OpenAIDatasetGenerator`
    with `filter_duplicated_examples=False`. The `generate_dataset_split()`
    method is then called with an `expected_num_examples` of `117`, which
    is greater than the size of the cached dataset. The function checks that
    the cached dataset is successfully loaded, and the generation continues
    to meet the expected number of examples.

    Note: The test uses a temporary directory as the cache root to ensure
        that the cache directory is cleaned up after the test finishes.

    Raises:
        AssertionError: If the cached dataset is not loaded correctly, or if the
            generation does not continue to meet the expected number of examples.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=False
        )
        examples_cache_path = Path(
            data_generator.cache_root / f"generated_examples_{DatasetSplit.TEST.value}"
        )
        cached_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        cached_dataset.save_to_disk(examples_cache_path)
        # The generate_dataset_split would first load the cached
        # dataset into generated_examples. Then in the while
        # loop, create_all_examples_dataset_and_generated_dataset
        # would be called to construct the generated_dataset.
        # Note that filter_duplicated_examples is False, so the
        # generated_examples won't be filtered. And since the
        # expected_num_examples is 117, the generation would
        # continue and the batch_size = 2. After one batch of API
        # calls, generated_dataset meets the requirement and
        # stop generation.
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_dataset = data_generator.generate_dataset_split(
                expected_num_examples=117,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert info_list[0] == f"Loading cache from {str(examples_cache_path)}."
            # The first logger.info is loaded cache, and there is
            # another 2 * 5 * 2 logger.info in extract_responses.
            assert len(info_list) == 1 + 2 * 5 * 2
            mock_warning.assert_not_called()
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110 + ["This is a great movie!"] * 10,
                "output_col": ["2"] * 110 + ["1"] * 10,
            }
        )
        assert are_datasets_identical(generated_dataset, excepted_generated_dataset)
        assert mocked_generate_example.call_count == 1
    gc.collect()


def test_extract_responses():
    """Test the extract_responses function of DatasetGenerator."""
    mock_completion_1 = MockCompletion()
    mock_completion_1.choices = [
        {"message": {"content": '{"input": "1", "output": "a"}'}},
        {"message": {"content": '{"input": "1", "output": "b"}'}},
        {"message": {"content": '{"input": "1", "output": "a"}'}},
    ]
    mock_completion_2 = MockCompletion()
    mock_completion_2.choices = [
        {"message": {"content": '{"input": "3", "output": "a"}'}},
        # Note that the following choice miss the right quote of JSON.
        # So it should be discarded. And will log a warning.
        {"message": {"content": '{"input": "3", "output": "a}'}},
        {"message": {"content": '{"input": "3", "output": "b"}'}},
    ]
    mock_completion_3 = MockCompletion()
    mock_completion_3.choices = [
        {"message": {"content": '{"input": "4", "output": "c"}'}},
        {"message": {"content": '{"input": "4", "output": "c"}'}},
        {"message": {"content": '{"input": "5", "output": "a"}'}},
    ]
    # choices should be list of dicts. So mock_completion_4
    # is invalid. Which will be discarded and log a warning.
    mock_completion_4 = MockCompletion()
    mock_completion_4.choices = None

    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
        generated_examples = []
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_examples = data_generator.extract_responses(
                [mock_completion_1, mock_completion_2], generated_examples
            )
            mock_warning.assert_called_once_with(
                'Error happened parsing API choice: {\'message\': {\'content\': \'{"input": "3", "output": "a}\'}}'  # noqa E501
            )
            # There are 5 valid examples. Each input
            # and output will be logged once as info.
            assert mock_info.call_count == 5 * 2

        # The second choice in mock_completion_2
        # is invalid. So it should be discarded.
        assert generated_examples == [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
        ]
        generated_examples = data_generator.extract_responses(
            [mock_completion_3], generated_examples
        )
        assert generated_examples == [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
            Example(input_col="4", output_col="c"),
            Example(input_col="4", output_col="c"),
            Example(input_col="5", output_col="a"),
        ]
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_examples = data_generator.extract_responses(
                [mock_completion_4], generated_examples
            )
            mock_warning.assert_called_once_with(
                "Error happened when parsing API completion: <MockObject choices=None>"
            )
            mock_info.assert_not_called()
            # The generated_examples should be the same.
            assert generated_examples == [
                Example(input_col="1", output_col="a"),
                Example(input_col="1", output_col="b"),
                Example(input_col="1", output_col="a"),
                Example(input_col="3", output_col="a"),
                Example(input_col="3", output_col="b"),
                Example(input_col="4", output_col="c"),
                Example(input_col="4", output_col="c"),
                Example(input_col="5", output_col="a"),
            ]
    gc.collect()


def test_extract_some_empty_responses():
    """Test the extract_responses function correctly handle empty responses."""
    mock_completion_1 = MockCompletion()
    mock_completion_1.choices = [
        # Note that this choice's input is empty. So it should be discarded.
        {"message": {"content": '{"input": "", "output": "a"}'}},
        {"message": {"content": '{"input": "5", "output": "b"}'}},
        # Note that this choice's output is empty. So it should be discarded.
        {"message": {"content": '{"input": "1", "output": ""}'}},
    ]
    mock_completion_2 = MockCompletion()
    mock_completion_2.choices = [
        {"message": {"content": '{"input": "3", "output": "a"}'}},
        # Note that the following choice misses the right quote of JSON.
        # So it should be discarded. And will log a warning.
        {"message": {"content": '{"input": "3", "output": "a}'}},
        {"message": {"content": '{"input": "3", "output": "b"}'}},
    ]
    mock_completion_3 = MockCompletion()
    mock_completion_3.choices = [
        {"message": {"content": '{"input": "4", "output": "c"}'}},
        {"message": {"content": '{"input": "4", "output": "c"}'}},
        {"message": {"content": '{"input": "5", "output": "a"}'}},
    ]
    # choices should be list of dicts. So mock_completion_4
    # is invalid. Which will be discarded and log a warning.
    mock_completion_4 = MockCompletion()
    mock_completion_4.choices = None

    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
        generated_examples = []
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_examples = data_generator.extract_responses(
                [mock_completion_1, mock_completion_2], generated_examples
            )
            mock_warning.assert_called_once_with(
                'Error happened parsing API choice: {\'message\': {\'content\': \'{"input": "3", "output": "a}\'}}'  # noqa E501
            )
            # There are 3 valid examples in [mock_completion_1,
            # mock_completion_2] Each input
            # and output will be logged once as info.
            # And there are 2 examples with empty
            # input or output, which should be discarded
            # and be logged as info.
            assert mock_info.call_count == 3 * 2 + 2

        # The second choice in mock_completion_2
        # is invalid. So it should be discarded.
        assert generated_examples == [
            Example(input_col="5", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
        ]
        generated_examples = data_generator.extract_responses(
            [mock_completion_3], generated_examples
        )
        assert generated_examples == [
            Example(input_col="5", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
            Example(input_col="4", output_col="c"),
            Example(input_col="4", output_col="c"),
            Example(input_col="5", output_col="a"),
        ]
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            generated_examples = data_generator.extract_responses(
                [mock_completion_4], generated_examples
            )
            mock_warning.assert_called_once_with(
                "Error happened when parsing API completion: <MockObject choices=None>"
            )
            mock_info.assert_not_called()
            # The generated_examples should be the same.
            assert generated_examples == [
                Example(input_col="5", output_col="b"),
                Example(input_col="3", output_col="a"),
                Example(input_col="3", output_col="b"),
                Example(input_col="4", output_col="c"),
                Example(input_col="4", output_col="c"),
                Example(input_col="5", output_col="a"),
            ]
    gc.collect()


def test_initialize_dataset_generator_with_dynamic_temperature():
    """Test the correct initialization of the dynamic temperature strategy."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        with pytest.raises(ValueError) as exc_info:
            _ = OpenAIDatasetGenerator(cache_root=cache_dir, initial_temperature=-0.2)
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "initial_temperature must be >= 0, but self.initial_temperature=-0.2"
        )
        with pytest.raises(ValueError) as exc_info:
            _ = OpenAIDatasetGenerator(cache_root=cache_dir, max_temperature=2.3)
            error_info = exc_info.value.args[0]
            assert (
                error_info
                == "max_temperature must be <= 2,0, but self.max_temperature=2.3"
            )

        with pytest.raises(ValueError) as exc_info:
            _ = OpenAIDatasetGenerator(
                cache_root=cache_dir, max_temperature=1.2, initial_temperature=1.5
            )
            error_info = exc_info.value.args[0]
            assert (
                error_info
                == "self.initial_temperature=1.5 must be <= self.max_temperature=1.2"
            )
