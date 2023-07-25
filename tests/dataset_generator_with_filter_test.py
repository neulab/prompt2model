"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import os
import tempfile
from collections import Counter, namedtuple
from functools import partial
from pathlib import Path
from unittest.mock import patch

import datasets
import pytest
from datasets import Dataset

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import (
    are_datasets_identical,
    mock_batch_openai_response_with_different_completions,
    mock_batch_openai_response_with_identical_completions,
    reset_mock_batch_openai_response_with_different_completions,
)

MOCK_EXAMPLE = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "6", "output": "f"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_openai_response_with_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

Example = namedtuple("Example", ["input_col", "output_col"])


class UNKNOWN_GPT3_EXCEPTION(Exception):
    """This is a newly-defined exception for testing purposes."""

    pass


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
            api_key, 3, filter_duplicated_examples=True, cache_root=cache_dir
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
            api_key, 3, filter_duplicated_examples=True, cache_root=cache_dir
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
            max_api_calls=3, filter_duplicated_examples=True, cache_root=cache_dir
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
            filter_duplicated_examples=True, cache_root=cache_dir
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
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
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
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
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
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
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
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
        ]
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        assert are_datasets_identical(
            data_generator.generated_dataset, expected_dataset
        )
    gc.collect()


def test_convert_generated_examples_to_generated_dataset_with_duplicate_inputs_duplicate_outputs():  # noqa: 501
    """Test constructing a map with duplicate inputs and duplicate outputs."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )
        data_generator.generating_split = DatasetSplit.TEST
        data_generator.generated_examples = [
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
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
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
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
        ]
        data_generator.convert_generated_examples_to_generated_dataset()

        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
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
    gc.collect()


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
        # Note that filter_duplicated_examples is True, so the
        # self.generated_examples will be filtered to 3 examples
        # in self.generated_dataset. Since expected_num_examples
        # is 3, the while loop would exit immediately. So the
        # self.generated_dataset would be the filtered cached dataset.
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            data_generator.generate_dataset_split(
                expected_num_examples=3,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )
            mock_info.assert_called_once_with(
                f"Loading cache from {str(dataset_cache_path)}."
            )
            mock_warning.assert_not_called()
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3"],
                "output_col": ["a", "a", "d"],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, excepted_generated_dataset
        )
        assert data_generator.generated_examples == [
            Example("1", "a"),
            Example("1", "a"),
            Example("1", "b"),
            Example("1", "c"),
            Example("2", "a"),
            Example("3", "d"),
        ]
        assert data_generator.input_output_map == {
            "1": Counter({"a": 2, "b": 1, "c": 1}),
            "2": Counter({"a": 1}),
            "3": Counter({"d": 1}),
        }
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
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=MOCK_EXAMPLE,
)
def test_load_cache_dataset_with_filter_duplicated_examples_and_continue_generation(
    mocked_generate_example,
):
    """Test OpenAIDatasetGenerator can load cache and continue generation."""
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
        # Note that filter_duplicated_examples is True, so the
        # self.generated_examples will be filtered to 3 examples
        # in self.generated_dataset. Since expected_num_examples
        # is 4, the generation would continue and the batch_size = 1.
        # After one batch of API calls, self.generated_dataset meets
        # the requirement and stop generation.
        with patch("logging.info") as mock_info, patch(
            "logging.warning"
        ) as mock_warning:
            data_generator.generate_dataset_split(
                expected_num_examples=4,
                prompt_spec=MockPromptSpec,
                split=DatasetSplit.TEST,
            )
            info_list = [each.args[0] for each in mock_info.call_args_list]
            assert info_list[0] == f"Loading cache from {str(dataset_cache_path)}."
            # The first logging.info is loaded cache, and there is
            # another 5 * 2 logging.info in extract_responses.
            assert len(info_list) == 1 + 5 * 2
            mock_warning.assert_not_called()
        excepted_generated_dataset = Dataset.from_dict(
            {
                "input_col": ["1", "2", "3", "6"],
                "output_col": ["a", "a", "d", "f"],
            }
        )
        assert are_datasets_identical(
            data_generator.generated_dataset, excepted_generated_dataset
        )
        assert data_generator.generated_examples == [
            Example("1", "a"),
            Example("1", "a"),
            Example("1", "b"),
            Example("1", "c"),
            Example("2", "a"),
            Example("3", "d"),
            Example("6", "f"),
            Example("6", "f"),
            Example("6", "f"),
            Example("6", "f"),
            Example("6", "f"),
        ]
        assert data_generator.input_output_map == {
            "1": Counter({"a": 2, "b": 1, "c": 1}),
            "2": Counter({"a": 1}),
            "3": Counter({"d": 1}),
            "6": Counter({"f": 5}),
        }
        assert mocked_generate_example.call_count == 1
    gc.collect()


"""
These tests validate the generation process with filter_duplicated_examples set to True.

These tests work together with `mock_batch_openai_response_with_different_completions`
function to simulate the generation process of the OpenAIDataSetGenerator.

The first five tests check the generation of a single dataset spilit, they used
a shared OpenAIDataSetGenerator with batch_size = 2, responses_per_request=3,
filter_duplicated_examples=True, and expected_num_examples = 5.

In the first API call, the generator produce 2 * 3 = 6 responses. After filtering
duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2"],
        "output_col": ["a", "a"],
    })

batch_size = (expected_num_examples - len(generated_dataset))
/ responses_per_request = (5 - 2) / 3 = 1.

The second API call reduces batch_size to 1 and generates 3 more responses.

After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3"],
        "output_col": ["a", "a", "a"],
    })

The third API call again uses batch_size = 1 and generates another 3 responses.
After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3"],
        "output_col": ["b", "a", "a"],
    })

The fourth and final API call also uses batch_size = 1 and generates a final 3
responses. After filtering duplicates, the generated_dataset will be:
    Dataset.from_dict(
    {
        "input_col": ["1", "2", "3", "4", "5"],
        "output_col": ["b", "a", "a", "c", "a"],
    })

The generator will then be exhausted, and the generation process will end.

The test suite contains five test cases, each using a different OpenAIDataSetGenerator.
These generators have the same settings (batch_size = 2, responses_per_request = 3,
expected_num_examples = 5, filter_duplicated_examples = True), but their max_api_calls
attribute is  2, 3, 4, 5, and unlimited respectively. Each test runs the generation of
its generator and verifies that the generated dataset matches the expected result.
"""

api_key = "fake_api_key"
prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
split = DatasetSplit.TRAIN
filter_duplicated_examples = True
expected_num_examples = 5
batch_size = 2
responses_per_request = 3


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_first_batch(mocked_generate_example):
    """Tests the filter methods in OpenAIDataSetGenerator in the first batch.

    This test initializes an OpenAIDataSetGenerator with the same settings as
    the suite's description but limits the number of API calls to 2. After running
    the generation process, it checks whether the generated dataset matches the
    expected result after the second API call. The test also asserts the number
    of calls to the API mock matches the expected number.

    Note that the first API call's batch_size = 2, generating 6 responses.
    """
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
        assert are_datasets_identical(
            dataset_generator.generated_dataset, excepted_dataset
        )
        excepted_examples = [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="c"),
            Example(input_col="2", output_col="a"),
            Example(input_col="2", output_col="b"),
        ]
        assert dataset_generator.generated_examples == excepted_examples
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_second_batch(mocked_generate_example):
    """Tests the filter methods in OpenAIDataSetGenerator in the second batch.

    This test initializes an OpenAIDataSetGenerator with the same settings as
    the suite's description but limits the number of API calls to 3. After running
    the generation process, it checks whether the generated dataset matches the
    expected result after the second API call. The test also asserts the number
    of calls to the API mock matches the expected number.

    Note that the first API call's batch_size = 2, generating 6 responses.
    The second API call's batch_size = 1, generating 3 responses.
    Init the OpenAIDatasetGenerator with `max_api_calls = 3`.
    """
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
        assert are_datasets_identical(
            dataset_generator.generated_dataset, excepted_dataset
        )
        excepted_examples = [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="c"),
            Example(input_col="2", output_col="a"),
            Example(input_col="2", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
        ]
        assert dataset_generator.generated_examples == excepted_examples
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_third_batch(mocked_generate_example):
    """Tests the filter methods in OpenAIDataSetGenerator in the thrird batch.

    This test initializes an OpenAIDataSetGenerator with the same settings as
    the suite's description but limits the number of API calls to 4. After running
    the generation process, it checks whether the generated dataset matches the
    expected result after the second API call. The test also asserts the number
    of calls to the API mock matches the expected number.

    Note that the first API call's batch_size = 2, generating 6 responses.
    The second API call's batch_size = 1, generating 3 responses.
    The third API call's batch_size = 1, generating 3 responses.
    Init the OpenAIDatasetGenerator with `max_api_calls = 4`.
    """
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
        assert are_datasets_identical(
            dataset_generator.generated_dataset, excepted_dataset
        )
        excepted_examples = [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="c"),
            Example(input_col="2", output_col="a"),
            Example(input_col="2", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
        ]
        assert dataset_generator.generated_examples == excepted_examples
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_forth_batch(mocked_generate_example):
    """Tests the filter methods in OpenAIDataSetGenerator in the forth batch.

    This test initializes an OpenAIDataSetGenerator with the same settings as
    the suite's description but limits the number of API calls to 5. After running
    the generation process, it checks whether the generated dataset matches the
    expected result after the second API call. The test also asserts the number
    of calls to the API mock matches the expected number.

    Note that the first API call's batch_size = 2, generating 6 responses.
    The second API call's batch_size = 1, generating 3 responses.
    The third API call's batch_size = 1, generating 3 responses.
    The forth and last API call's batch_size = 1. And generate 3 responses.
    Init the OpenAIDatasetGenerator with `max_api_calls = 5`.
    """
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
        assert are_datasets_identical(
            dataset_generator.generated_dataset, excepted_dataset
        )
        excepted_examples = [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="c"),
            Example(input_col="2", output_col="a"),
            Example(input_col="2", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="4", output_col="c"),
            Example(input_col="4", output_col="c"),
            Example(input_col="5", output_col="a"),
        ]
        assert dataset_generator.generated_examples == excepted_examples
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_unlimited_api_calls(mocked_generate_example):
    """Tests the filter methods in OpenAIDataSetGenerator with unlimited API calls.

    This test initializes an OpenAIDataSetGenerator with the same settings as
    the suite's description but limits the number of API calls to unlimited. After
    running the generation process, it checks whether the generated dataset
    matches the expected result after the second API call. The test also asserts
    the number of calls to the API mock matches the expected number.

    Note that the first API call's batch_size = 2, generating 6 responses.
    The second API call's batch_size = 1, generating 3 responses.
    The third API call's batch_size = 1, generating 3 responses.
    The forth and last API call's batch_size = 1. And generate 3 responses.
    After the forth batch, the generation ends. No need for further API call.
    # Init the OpenAIDatasetGenerator with unlimited `max_api_calls`.
    """
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
        assert are_datasets_identical(
            dataset_generator.generated_dataset, excepted_dataset
        )
        excepted_examples = [
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="a"),
            Example(input_col="1", output_col="c"),
            Example(input_col="2", output_col="a"),
            Example(input_col="2", output_col="b"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="a"),
            Example(input_col="3", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="1", output_col="b"),
            Example(input_col="4", output_col="c"),
            Example(input_col="4", output_col="c"),
            Example(input_col="5", output_col="a"),
        ]
        assert dataset_generator.generated_examples == excepted_examples
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_batch_openai_chat_completion",
    side_effect=mock_batch_openai_response_with_different_completions,
)
def test_generator_with_filter_to_generate_datasetdict(mocked_generate_example):
    """The last test cheks the generation of a DatasetDict.

    Inintialize OpenAIDataSetGenerator with batch_size = 2,
    responses_per_request=3, filter_duplicated_examples=True,
    max_api_calls=7 and expected_num_examples = {
        DatasetSplit.TRAIN: 4,
        DatasetSplit.VAL: 4,
        DatasetSplit.TEST: 2,
    }

    For the first split, the batch_size = 2, calling the API twice,
    generating 6 responses in the first batch. After filtering
    duplicates, the generated dataset for train split is:
    Dataset.from_dict(
    {
        "input_col": ["1", "2"],
        "output_col": ["a", "a"],
    })

    The train split has not meets the expected num_examples, so
    batch_size = (expected_num_examples - len(generated_dataset))
    / responses_per_request = (5 - 2) / 3 = 1.

    The second API call reduces batch_size to 1 and generates 3 more responses.

    After filtering duplicates, the generated_dataset for train split will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3"],
            "output_col": ["a", "a", "a"],
        })

    The third API call again uses batch_size = 1 and generates another 3 responses.
    After filtering duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3"],
            "output_col": ["b", "a", "a"],
        })

    The fourth and final API call also uses batch_size = 1 and generates a final 3
    responses. After filtering duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3", "4", "5"],
            "output_col": ["b", "a", "a", "c", "a"],
        })

    Now the API call counter is 5. There is still 2 API calls left to
    genrate other spilits.

    For the val split, the batch_size = 2, calling the API twice,
    generating 6 responses in the first batch. After filtering
    duplicates, the generated dataset for train split is:
    Dataset.from_dict(
    {
        "input_col": ["1", "2"],
        "output_col": ["a", "a"],
    })

    The val split has not meets the expected num_examples,
    but the max_api_calls is reached, so the generation ends.

    Thus the val split's generated_dataset is:
        Dataset.from_dict(
    {
        "input_col": ["1", "2"],
        "output_col": ["a", "a"],
    })

    And the test split's generated_dataset is empty.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        reset_mock_batch_openai_response_with_different_completions()
        dataset_generator = OpenAIDatasetGenerator(
            api_key,
            filter_duplicated_examples=filter_duplicated_examples,
            cache_root=cache_dir,
            batch_size=batch_size,
            responses_per_request=responses_per_request,
            max_api_calls=7,
        )
        generated_dataset_dict = dataset_generator.generate_dataset_dict(
            prompt_spec,
            expected_num_examples={
                DatasetSplit.TRAIN: 4,
                DatasetSplit.VAL: 4,
                DatasetSplit.TEST: 2,
            },
        )
        assert mocked_generate_example.call_count == 5
        assert dataset_generator.api_call_counter == 7
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
        assert are_datasets_identical(generated_dataset_dict, expected_dataset_dict)
    gc.collect()
