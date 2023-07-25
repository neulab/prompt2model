"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import os
import tempfile
from collections import Counter, namedtuple
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import (
    MockCompletion,
    are_datasets_identical,
    mock_batch_openai_response_with_different_completions,
    mock_batch_openai_response_with_identical_completions,
)

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

example = namedtuple("example", ["input_col", "output_col"])


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
    expected_num_examples = 29
    # if expected_num_examples >= max_api_calls, the returned dataset's
    # length will be less or equal than max_api_calls.
    dataset = dataset_generator.generate_dataset_split(
        prompt_spec, expected_num_examples, split
    )
    # Since each API call would return one completion object with 5 responses
    # and some of the responses are invalid JSON objects, the upper bound of
    # the length of the dataset is expected_num_examples + 5.
    assert len(dataset) < expected_num_examples + 5
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns
    return dataset


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
            # generated dataset is expected_num_examples + 5.
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
def tes_generator_without_filter(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    This function first test the unlimited generation. Then test generation
    when expected_num_examples >= max_api_calls. Thus the API agent will only be
    called max_api_calls times.

    Args:
        mocked_generate_example: The function represents the @patch function.
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
        # Each api call will return 5 responses, and each response is valid JSON.
        # So the unlimited_dataset_generator will call API (29 // 5 + 1) times.
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

        # Each API call returns five responses. So unlimited_dataset_generator will
        # call API (50 // 5 + 24 // 5 + 1 + 26 // 5 + 1) = 21 times.
        assert unlimited_dataset_generator.api_call_counter == (
            50 // 5 + 24 // 5 + 1 + 26 // 5 + 1
        )
        # The default batch_size is 5. So generate_batch_openai_chat_completion
        # will be called 2 times for 50 examples in train split, 1 times for 24 examples
        # in the validation split, and 2 times for 26 examples in test split.
        assert mocked_generate_example.call_count == 2 + 1 + 2

        # Each API call returns 5 responses, and each response is valid JSON.
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
        # The max_api_calls is 3. So the limited_dataset_generator call API 3 times.
        # Each API call returns 5 responses. So the limited_dataset_generator will
        # have 3 * 5 = 15 examples.
        assert len(limited_generated_dataset) == 15

        # The default batch_size is 5. So generate_batch_openai_chat_completion
        # will be called only once.
        assert mocked_generate_example.call_count == 1

        # Each API call returns 5 responses, so the limited_dataset_generator
        # will use up all the available API calls.
        assert limited_dataset_generator.api_call_counter == 3

        # Each API call returns 5 responses, and each response is valid JSON.
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
        # Since the max_api_calls is 13, the limited_dataset_generator can not
        # generate the whole dataset_dict, and will call API 13 times.
        assert limited_dataset_generator.api_call_counter == 13

        # The train split has 50 examples, so it will call API 10 times and call
        # generate_batch_openai_chat_completion 2 times.
        # The validation split has 24 examples, but there is only 3 API calls
        # left, so it will call API 3 times and call
        # generate_batch_openai_chat_completion 1 time.
        # The test split has 26 examples, but there is no more API calls left,
        # so it will not generate_batch_openai_chat_completion.
        assert mocked_generate_example.call_count == 2 + 1 + 0

        # Each API call returns 5 responses, and each response is valid JSON.
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
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
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
        dataset = dataset_generator.generate_dataset_split(
            prompt_spec, expected_num_examples, split
        )
        assert mocked_generate_example.call_count == 3
        assert (
            dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
        )
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_INVALID_JSON,
)
def test_invalid_json_response(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
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
        assert (
            dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
        )
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=UNKNOWN_GPT3_EXCEPTION(),
)
def test_unexpected_examples_of_GPT(mocked_generate_example):
    """Test OpenAIDatasetGenerator when the agent returns a wrong key dict.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    with pytest.raises(
        UNKNOWN_GPT3_EXCEPTION
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
    """Test openai key initialization."""
    api_key = None
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(
        AssertionError
    ) as exc_info, tempfile.TemporaryDirectory() as cache_dir:
        _ = OpenAIDatasetGenerator(
            filter_duplicated_examples=False, cache_root=cache_dir
        )
        assert str(exc_info.value) == (
            "API key must be provided or set the environment variable"
            + " with `export OPENAI_API_KEY=<your key>`"
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


def test_construct_map_with_duplicate_inputs_unique_outputs():
    """Test constructing a map with duplicate inputs but unique outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="apple", output_col="E"),
            example(input_col="orange", output_col="O"),
            example(input_col="apple", output_col="D"),
        ]
        data_generator.construct_input_output_map()

        expected_output = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }
        assert data_generator.input_output_map == expected_output
    gc.collect()


def test_construct_map_with_duplicate_inputs_duplicate_outputs():
    """Test constructing a map with duplicate inputs and duplicate outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="C"),
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="apple", output_col="G"),
            example(input_col="apple", output_col="A"),
            example(input_col="orange", output_col="O"),
            example(input_col="apple", output_col="D"),
            example(input_col="banana", output_col="B"),
            example(input_col="orange", output_col="F"),
        ]
        data_generator.construct_input_output_map()

        expected_output = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }
        assert data_generator.input_output_map == expected_output
    gc.collect()


def test_construct_map_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="orange", output_col="O"),
        ]
        data_generator.construct_input_output_map()

        expected_output = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }
        assert data_generator.input_output_map == expected_output
    gc.collect()


def test_construct_map_with_empty_examples_list():
    """Test constructing a map with empty inputs and outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generated_examples = []
        data_generator.construct_input_output_map()

        # if self.generated_examples is empty, self.input_output_map is None.
        assert data_generator.input_output_map == {}
    gc.collect()


def test_multi_vote_with_duplicate_inputs_unique_outputs():
    """Test multi-voting with duplicate inputs but unique outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.input_output_map = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }
        data_generator.apply_multi_vote_to_construct_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_multi_vote_with_duplicate_inputs_duplicate_outputs():
    """Test multi-voting with duplicate inputs and duplicate outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.input_output_map = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }
        data_generator.apply_multi_vote_to_construct_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_multi_vote_with_unique_inputs_outputs():
    """Test multi-voting with unique inputs and outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)
        data_generator.input_output_map = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }
        data_generator.apply_multi_vote_to_construct_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_multi_vote_with_empty_examples_list():
    """Test multi-voting with empty inputs and outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)
        data_generator.input_output_map = {}
        data_generator.apply_multi_vote_to_construct_generated_dataset()

        expected_dataset = Dataset.from_dict({})

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_convert_generated_examples_to_generated_dataset_with_duplicate_inputs_unique_outputs():  # noqa: 501
    """Test constructing generated dataset with duplicate inputs but unique outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generating_split = DatasetSplit.TEST
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="apple", output_col="E"),
            example(input_col="orange", output_col="O"),
            example(input_col="apple", output_col="D"),
        ]
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
        data_generator.filter_duplicated_examples = False
        data_generator.convert_generated_examples_to_generated_dataset()
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )


def test_convert_generated_examples_to_generated_dataset_with_duplicate_inputs_duplicate_outputs():  # noqa: 501
    """Test constructing a map with duplicate inputs and duplicate outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generating_split = DatasetSplit.TEST
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="C"),
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="apple", output_col="G"),
            example(input_col="apple", output_col="A"),
            example(input_col="orange", output_col="O"),
            example(input_col="apple", output_col="D"),
            example(input_col="banana", output_col="B"),
            example(input_col="orange", output_col="F"),
        ]
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
        data_generator.filter_duplicated_examples = False
        data_generator.convert_generated_examples_to_generated_dataset()
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_convert_generated_examples_to_generated_dataset_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generating_split = DatasetSplit.TEST
        data_generator.generated_examples = [
            example(input_col="apple", output_col="A"),
            example(input_col="banana", output_col="B"),
            example(input_col="orange", output_col="O"),
        ]
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
        data_generator.filter_duplicated_examples = False
        data_generator.convert_generated_examples_to_generated_dataset()
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_convert_generated_examples_to_generated_dataset_with_empty_examples_list():
    """Test constructing a map with empty inputs and outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generating_split = DatasetSplit.TEST
        data_generator.generated_examples = []
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict({})

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
        data_generator.filter_duplicated_examples = False
        data_generator.convert_generated_examples_to_generated_dataset()
        expected_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_compute_batch_size_with_limited_max_api_calls():
    """Test the batch size computation with limited max API calls."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(max_api_calls=28, cache_root=cache_dir)
        data_generator.api_call_counter = 26
        data_generator.generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        # Default batch size and responses_per_request are both 5.
        # So each batch should contain 25 examples.

        # At least (125 - 110) / 5 = 3 API calls needed to get
        # more than 125 examples.

        # The left API calls allowed are 28 - 26 = 2.

        batch_size = data_generator.compute_batch_size(expected_num_examples=125)
        assert (
            batch_size
            == data_generator.max_api_calls - data_generator.api_call_counter
            == 28 - 26
        )

        data_generator.api_call_counter = 20
        batch_size = data_generator.compute_batch_size(expected_num_examples=125)
        assert (
            batch_size
            == ((125 - len(data_generator.generated_dataset)))
            / data_generator.responses_per_request
            == (125 - 110) / 5
        )

        data_generator.api_call_counter = 0
        data_generator.generated_dataset = Dataset.from_dict(
            {
                "input_col": [1] * 50,
                "output_col": [2] * 50,
            }
        )
        batch_size = data_generator.compute_batch_size(expected_num_examples=125)
        assert batch_size == data_generator.batch_size


def test_compute_batch_size_with_unlimited_max_api_calls():
    """Test the batch size computation with unlimited max API calls."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)
        data_generator.generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        # Default batch size and responses_per_request are both 5.
        # So each batch should contain 25 examples.

        # At least (125 - 110) / 5 = 3 API calls needed to get
        # more than 125 examples.

        batch_size = data_generator.compute_batch_size(expected_num_examples=125)
        assert (
            batch_size
            == (125 - len(data_generator.generated_dataset))
            / data_generator.responses_per_request
            == (125 - 110) / 5
        )

        data_generator.generated_dataset = Dataset.from_dict(
            {
                "input_col": [1] * 50,
                "output_col": [2] * 50,
            }
        )
        batch_size = data_generator.compute_batch_size(expected_num_examples=125)
        assert batch_size == data_generator.batch_size == 5


def test_load_cache_dataset_without_filter_duplicated_examples():
    """Test the cached dataset loading without filtering duplicated examples."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=False
        )
        dataset_cache_path = Path(
            data_generator.cache_root / f"{DatasetSplit.TEST.value}"
        )
        cached_dataset = Dataset.from_dict(
            {
                "input_col": ["1"] * 110,
                "output_col": ["2"] * 110,
            }
        )
        cached_dataset.save_to_disk(dataset_cache_path)
        # The generate_dataset_split would first load the cached
        # dataset into self.generated_examples. Then in the while
        # loop, convert_generated_examples_to_generated_dataset
        # would be called to construct the self.generated_dataset.
        # Note that filter_duplicated_examples is False, so the
        # self.generated_examples won't be filtered. And since the
        # expected_num_examples is 0, the while loop would exit
        # immediately. So the self.generated_dataset would be the
        # same as the cached dataset.
        data_generator.generate_dataset_split(
            expected_num_examples=0, prompt_spec=MockPromptSpec, split=DatasetSplit.TEST
        )
        assert are_datasets_identical(data_generator.generated_dataset, cached_dataset)
        directly_constructed_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(directly_constructed_dataset, cached_dataset)


def test_load_cache_dataset_with_filter_duplicated_examples():
    """Test the cached dataset loading with filtering duplicated examples."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
        dataset_cache_path = Path(
            data_generator.cache_root / f"{DatasetSplit.TEST.value}"
        )
        cached_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "1", "1", "1", "2", "3"],
                "output_col": ["a", "a", "b", "c", "a", "d"],
            }
        )
        cached_dataset.save_to_disk(dataset_cache_path)
        # The generate_dataset_split would first load the cached
        # dataset into self.generated_examples. Then in the while
        # loop, convert_generated_examples_to_generated_dataset
        # would be called to construct the self.generated_dataset.
        # Note that filter_duplicated_examples is False, so the
        # self.generated_examples won't be filtered. And since the
        # expected_num_examples is 0, the while loop would exit
        # immediately. So the self.generated_dataset would be the
        # same as the cached dataset.
        data_generator.generate_dataset_split(
            expected_num_examples=0, prompt_spec=MockPromptSpec, split=DatasetSplit.TEST
        )
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "d"],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, excepted_generated_dataset
        )
        directly_constructed_dataset = Dataset.from_dict(
            {
                "input_col": [
                    example.input_col for example in data_generator.generated_examples
                ],
                "output_col": [
                    example.output_col for example in data_generator.generated_examples
                ],
            }
        )
        assert are_datasets_identical(directly_constructed_dataset, cached_dataset)


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
        # So it should be discarded. And will log an warning.
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
    # is invalid. Which will be discarded and log an warning.
    mock_completion_4 = MockCompletion()
    mock_completion_4.choices = None

    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
        assert data_generator.generated_examples == []
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            data_generator.extract_responses([mock_completion_1, mock_completion_2])
            mock_warning.assert_called_once_with(
                'Error happened parsing API choice: {\'message\': {\'content\': \'{"input": "3", "output": "a}\'}}'  # noqa E501
            )
            # There are 5 valid examples. Each input
            # and output will be logged once as info.
            assert mock_info.call_count == 5 * 2

        # The second choice in mock_completion_2
        # is invalid. So it should be discarded.
        assert data_generator.generated_examples == [
            example(input_col="1", output_col="a"),
            example(input_col="1", output_col="b"),
            example(input_col="1", output_col="a"),
            example(input_col="3", output_col="a"),
            example(input_col="3", output_col="b"),
        ]
        data_generator.extract_responses([mock_completion_3])
        assert data_generator.generated_examples == [
            example(input_col="1", output_col="a"),
            example(input_col="1", output_col="b"),
            example(input_col="1", output_col="a"),
            example(input_col="3", output_col="a"),
            example(input_col="3", output_col="b"),
            example(input_col="4", output_col="c"),
            example(input_col="4", output_col="c"),
            example(input_col="5", output_col="a"),
        ]
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            data_generator.extract_responses([mock_completion_4])
            mock_warning.assert_called_once_with(
                "Error happened when parsing API completion: <MockObject choices=None>"
            )
            mock_info.assert_not_called()
            # The generated_examples should be the same.
            assert data_generator.generated_examples == [
                example(input_col="1", output_col="a"),
                example(input_col="1", output_col="b"),
                example(input_col="1", output_col="a"),
                example(input_col="3", output_col="a"),
                example(input_col="3", output_col="b"),
                example(input_col="4", output_col="c"),
                example(input_col="4", output_col="c"),
                example(input_col="5", output_col="a"),
            ]


def test_generator_with_filter():
    """Test the generation with filter_duplicated_examples=True.

    This function is a carefully designed test togher with the iterator created
    in mock_batch_openai_response_with_different_completions.

    Initialize an OpenAIDatasetGenerator. Set batch_size = 2, responses_per_request
    = 3, expected_num_examples = 5, filter_duplicated_examples = True.

    In the first API call, the ChatGPTAgent will generate 2 * 3 = 6 responses.
    After filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2"],
                "output_col": ["a", "a"],
            }
        )

    The second API call's batch_size = 1. And generate 3 responses.
    batch_size = (expected_num_examples - len(generated_dataset))
    / responses_per_request = 1.

    After the filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "a"],
            }
        )

    The third API call's batch_size = 1. And generate 3 responses.

    After the filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["b", "a", "a"],
            }
        )

    The fourth and last API call's batch_size = 1. And generate 3 responses.
    After the filtering the duplicated responses, the generated_dataset will be
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3", "4", "5"],
            "output_col": ["b", "a", "a", "c", "a"],
        }
    )

    Then the generator will be exhausted and the generation also ends.

    The function contain four test cases with four limited dataset generators.
    Their batch_size = 2, responses_per_request = 3, expected_num_examples
    = 5, filter_duplicated_examples = True. But the number of max_api_calls are
    1, 2, 3, 4. Then we run the generation for each of them and check that each
    generated dataset is correct as expected.
    """
    api_key = "fake_api_key"
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    split = DatasetSplit.TRAIN
    filter_duplicated_examples = True
    expected_num_examples = 5
    batch_size = 2
    responses_per_request = 3

    def reset_mock_batch_openai_response_with_different_completions():
        if hasattr(
            mock_batch_openai_response_with_different_completions, "mock_completions"
        ):
            del mock_batch_openai_response_with_different_completions.mock_completions
        if hasattr(
            mock_batch_openai_response_with_different_completions, "current_index"
        ):
            del mock_batch_openai_response_with_different_completions.current_index

    @patch(
        "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
        side_effect=mock_batch_openai_response_with_different_completions,
    )
    def test_generator_with_filter_first_batch(mocked_generate_example):
        # The first API call's batch_size = 2. And generate 6 responses.
        # Init the OpenAIDatasetGenerator with `max_api_calls = 2`.
        with tempfile.TemporaryDirectory() as cache_dir:
            reset_mock_batch_openai_response_with_different_completions()
            dataset_generator = OpenAIDatasetGenerator(
                api_key,
                max_api_calls=2,
                filter_duplicated_examples=filter_duplicated_examples,
                cache_root=cache_dir,
                batch_size=batch_size,
                responses_per_request=responses_per_request,
            )
            generated_dataset = dataset_generator.generate_dataset_split(
                prompt_spec, expected_num_examples, split
            )
            assert mocked_generate_example.call_count == 1
            assert dataset_generator.api_call_counter == 2
            excepted_dataset = Dataset.from_dict(
                {
                    "input_col": ["1", "2"],
                    "output_col": ["a", "a"],
                }
            )
            assert are_datasets_identical(generated_dataset, excepted_dataset)

    @patch(
        "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
        side_effect=mock_batch_openai_response_with_different_completions,
    )
    def test_generator_with_filter_second_batch(mocked_generate_example):
        # The first API call's batch_size = 1. And generate 6 responses.
        # The second API call's batch_size = 1. And generate 3 responses.
        # Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
        with tempfile.TemporaryDirectory() as cache_dir:
            reset_mock_batch_openai_response_with_different_completions()
            dataset_generator = OpenAIDatasetGenerator(
                api_key,
                max_api_calls=3,
                filter_duplicated_examples=filter_duplicated_examples,
                cache_root=cache_dir,
                batch_size=batch_size,
                responses_per_request=responses_per_request,
            )
            generated_dataset = dataset_generator.generate_dataset_split(
                prompt_spec, expected_num_examples, split
            )
            assert mocked_generate_example.call_count == 2
            assert dataset_generator.api_call_counter == 3
            excepted_dataset = Dataset.from_dict(
                {
                    "input_col": ["1", "2", "3"],
                    "output_col": ["a", "a", "a"],
                }
            )
            assert are_datasets_identical(generated_dataset, excepted_dataset)

    @patch(
        "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
        side_effect=mock_batch_openai_response_with_different_completions,
    )
    def test_generator_with_filter_third_batch(mocked_generate_example):
        # The first API call's batch_size = 1. And generate 6 responses.
        # The second API call's batch_size = 1. And generate 3 responses.
        # The third API call's batch_size = 1. And generate 3 responses.
        # Init the OpenAIDatasetGenerator with `max_api_calls = 4`.
        with tempfile.TemporaryDirectory() as cache_dir:
            reset_mock_batch_openai_response_with_different_completions()
            dataset_generator = OpenAIDatasetGenerator(
                api_key,
                max_api_calls=4,
                filter_duplicated_examples=filter_duplicated_examples,
                cache_root=cache_dir,
                batch_size=batch_size,
                responses_per_request=responses_per_request,
            )
            generated_dataset = dataset_generator.generate_dataset_split(
                prompt_spec, expected_num_examples, split
            )
            assert mocked_generate_example.call_count == 3
            assert dataset_generator.api_call_counter == 4
            excepted_dataset = Dataset.from_dict(
                {
                    "input_col": ["1", "2", "3"],
                    "output_col": ["b", "a", "a"],
                }
            )
            assert are_datasets_identical(generated_dataset, excepted_dataset)

    @patch(
        "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
        side_effect=mock_batch_openai_response_with_different_completions,
    )
    def test_generator_with_filter_forth_batch(mocked_generate_example):
        # The first API call's batch_size = 1. And generate 6 responses.
        # The second API call's batch_size = 1. And generate 3 responses.
        # The third API call's batch_size = 1. And generate 3 responses.
        # The forth and last API call's batch_size = 1. And generate 3 responses.
        # Init the OpenAIDatasetGenerator with `max_api_calls = 5`.
        with tempfile.TemporaryDirectory() as cache_dir:
            reset_mock_batch_openai_response_with_different_completions()
            dataset_generator = OpenAIDatasetGenerator(
                api_key,
                max_api_calls=5,
                filter_duplicated_examples=filter_duplicated_examples,
                cache_root=cache_dir,
                batch_size=batch_size,
                responses_per_request=responses_per_request,
            )
            generated_dataset = dataset_generator.generate_dataset_split(
                prompt_spec, expected_num_examples, split
            )
            assert mocked_generate_example.call_count == 4
            assert dataset_generator.api_call_counter == 5
            excepted_dataset = Dataset.from_dict(
                {
                    "input_col": ["1", "2", "3", "4", "5"],
                    "output_col": ["b", "a", "a", "c", "a"],
                }
            )
            assert are_datasets_identical(generated_dataset, excepted_dataset)

    @patch(
        "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
        side_effect=mock_batch_openai_response_with_different_completions,
    )
    def test_generator_with_filter_unlimited_api_calls(mocked_generate_example):
        # The first API call's batch_size = 1. And generate 6 responses.
        # The second API call's batch_size = 1. And generate 3 responses.
        # The third API call's batch_size = 1. And generate 3 responses.
        # The forth and last API call's batch_size = 1. And generate 3 responses.
        # After the forth batch, the generation ends. No need for further API call.
        # Init the OpenAIDatasetGenerator with unlimited `max_api_calls`.
        with tempfile.TemporaryDirectory() as cache_dir:
            reset_mock_batch_openai_response_with_different_completions()
            dataset_generator = OpenAIDatasetGenerator(
                api_key,
                filter_duplicated_examples=filter_duplicated_examples,
                cache_root=cache_dir,
                batch_size=batch_size,
                responses_per_request=responses_per_request,
            )
            generated_dataset = dataset_generator.generate_dataset_split(
                prompt_spec, expected_num_examples, split
            )
            assert mocked_generate_example.call_count == 4
            assert dataset_generator.api_call_counter == 5
            excepted_dataset = Dataset.from_dict(
                {
                    "input_col": ["1", "2", "3", "4", "5"],
                    "output_col": ["b", "a", "a", "c", "a"],
                }
            )
            assert are_datasets_identical(generated_dataset, excepted_dataset)

    test_generator_with_filter_first_batch()
    test_generator_with_filter_second_batch()
    test_generator_with_filter_third_batch()
    test_generator_with_filter_forth_batch()
    test_generator_with_filter_unlimited_api_calls()

    """Test the generation with filter_duplicated_examples=True.

    This function is a carefully designed test togher with the iterator created
    in mock_batch_openai_response_with_different_completions.

    Initialize an OpenAIDatasetGenerator. Set batch_size = 2, responses_per_request
    = 3, expected_num_examples = 5, filter_duplicated_examples = True.

    In the first API call, the ChatGPTAgent will generate 2 * 3 = 6 responses.
    After filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2"],
                "output_col": ["a", "a"],
            }
        )

    The second API call's batch_size = 1. And generate 3 responses.
    batch_size = (expected_num_examples - len(generated_dataset))
    / responses_per_request = 1.

    After the filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "a"],
            }
        )

    The third API call's batch_size = 1. And generate 3 responses.

    After the filtering the duplicated responses, the generated_dataset will be
            Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["b", "a", "a"],
            }
        )

    The fourth and last API call's batch_size = 1. And generate 3 responses.
    After the filtering the duplicated responses, the generated_dataset will be
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3", "4", "5"],
            "output_col": ["b", "a", "a", "c", "a"],
        }
    )

    Then the generator will be exhausted and the generation also ends.

    The function contain four test cases with four limited dataset generators.
    Their batch_size = 2, responses_per_request = 3, expected_num_examples
    = 5, filter_duplicated_examples = True. But the number of max_api_calls are
    1, 2, 3, 4. Then we run the generation for each of them and check that each
    generated dataset is correct as expected.
    """