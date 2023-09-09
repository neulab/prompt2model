"""Testing DatasetGenerator through PromptBasedDatasetGenerator."""

import logging
import os
import tempfile
from functools import partial
from unittest.mock import patch

import datasets
import pytest
from datasets import Dataset

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.prompt_based import (
    Example,
    PromptBasedDatasetGenerator,
)
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import api_tools
from test_helpers import (
    MockCompletion,
    UnknownGpt3Exception,
    mock_batch_api_response_identical_completions,
)
from test_helpers.mock_api import MockAPIAgent, MockBatchDifferentCompletions
from test_helpers.test_utils import temp_setattr

logger = logging.getLogger("DatasetGenerator")

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)

MOCK_CLASSIFICATION_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1"}',
)
MOCK_WRONG_KEY_EXAMPLE = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "label": "1"}',
)
MOCK_INVALID_JSON = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "This is a great movie!", "output": "1}',
)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generate_dataset(mocked_generate_example):
    """Test the `generate_dataset_split()` function of `PromptBasedDatasetGenerator`."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    dataset_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=False)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    split = DatasetSplit.TRAIN
    num_examples = 29
    # If num_examples >= max_api_calls, the returned dataset's
    # length will be less than or equal to max_api_calls.
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    # Since each API call would return one completion object with 5 responses
    # and some of the responses are invalid JSON objects, the upper bound of
    # the length of the dataset is num_examples + 5, where 5 is the
    # default number of responses per API call.
    assert len(dataset) < num_examples + 5
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns
    return dataset


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generate_dataset_dict(mocked_generate_example):
    """Test the `generate_dataset_dict()` function of `PromptBasedDatasetGenerator`."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    dataset_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=False)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = {
        DatasetSplit.TRAIN: 50,
        DatasetSplit.VAL: 24,
        DatasetSplit.TEST: 26,
    }
    dataset_dict = dataset_generator.generate_dataset_dict(
        prompt_spec=prompt_spec,
        num_examples=num_examples,
    )

    assert set(dataset_dict.keys()) == {"train", "val", "test"}
    for split, num in num_examples.items():
        # As explained previously, the upper bound of the length of
        # generated dataset is num_examples + 5, where
        # 5 is the default number of responses per API call.
        assert len(dataset_dict[split.value]) < num + 5
    expected_columns = {"input_col", "output_col"}
    for dataset in dataset_dict.values():
        assert set(dataset.column_names) == expected_columns


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generator_without_filter(mocked_generate_example):
    """Unlimited dataset generation using the PromptBasedDatasetGenerator."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    dataset_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=False)
    dataset = dataset_generator.generate_dataset_split(
        MockPromptSpec(TaskType.TEXT_GENERATION), 29, DatasetSplit.TRAIN
    )
    assert len(dataset) == 29
    # The default responses_per_request is 5. So each API call will return
    # 5 responses, i.e. 5 choices in openai.Completion.choices.
    # Each API call will return 5 responses, and each response is a valid JSON.
    # So the unlimited_dataset_generator will call the API 6 times.
    assert dataset_generator.api_call_counter == 6
    # The default batch_size is 5. So generate_batch_completion
    # will be called 2 times with  first batch_size = 5 and second batch_size = 1.
    assert mocked_generate_example.call_count == 2


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generator_without_filter_dict(mocked_generate_example):
    """Test generation of a dataset dict."""
    dataset_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=False)

    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = {
        DatasetSplit.TRAIN: 50,
        DatasetSplit.VAL: 24,
        DatasetSplit.TEST: 26,
    }

    dataset_dict = dataset_generator.generate_dataset_dict(
        prompt_spec=prompt_spec,
        num_examples=num_examples,
    )

    assert set(dataset_dict.keys()) == {"train", "val", "test"}
    for split, num in num_examples.items():
        # As explained previously, the upper bound of the length of
        # generated dataset is num_examples + 5, where
        # 5 is the default number of responses per API call.
        assert len(dataset_dict[split.value]) < num + 5
    expected_columns = {"input_col", "output_col"}
    for dataset in dataset_dict.values():
        assert set(dataset.column_names) == expected_columns

    # Each API call returns five responses. So the dataset_generator will
    # call the API (50 // 5 + 24 // 5 + 1 + 26 // 5 + 1) = 21 times.
    assert dataset_generator.api_call_counter == (50 // 5 + 24 // 5 + 1 + 26 // 5 + 1)
    # The default batch_size is 5. So generate_batch_completion
    # will be called 2 times for 50 examples in the train split,
    # 1 time for 24 examples in the validation split,
    # and 2 times for 26 examples in the test split.
    assert mocked_generate_example.call_count == 2 + 1 + 2

    # Each API call returns 5 responses, and each response is a valid JSON.
    # So the dataset_dict will contain (50, 25, 30) examples.
    assert len(dataset_dict["train"]) == 50
    assert len(dataset_dict["val"]) == 24
    assert len(dataset_dict["test"]) == 26


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generator_max_api_calls(mocked_generate_example):
    """Test generation when num_examples >= max_api_calls."""
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=3, filter_duplicated_examples=False
    )
    dataset = dataset_generator.generate_dataset_split(
        MockPromptSpec(TaskType.TEXT_GENERATION), 29, DatasetSplit.TRAIN
    )
    # The max_api_calls is 3. So the limited_dataset_generator calls the
    # API 3 times. Each API call returns 5 responses. So the
    # limited_dataset_generator will have 3 * 5 = 15 examples.
    assert len(dataset) == 15

    # The default batch_size is 5. So generate_batch_completion
    # will be called only once.
    assert mocked_generate_example.call_count == 1

    # Each API call returns 5 responses, so the limited_dataset_generator
    # will use up all the available API calls.
    assert dataset_generator.api_call_counter == 3

    # Each API call returns 5 responses, and each response is a valid JSON.
    # So the dataset will contain 15 examples.
    assert len(dataset) == 15


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_first_batch(mocked_generate_example):
    """Test PromptBasedDatasetGenerator with filter methods in the first batch."""
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=2,
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
    )

    # Generate the dataset split using the initialized generator.
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples=5,
        split=DatasetSplit.TRAIN,
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
    assert list(generated_dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_second_batch(mocked_generate_example):
    """Test PromptBasedDatasetGenerator with filter methods in the second batch.

    This test verifies the behavior of the PromptBasedDatasetGenerator with filter
    methods in the second batch of API calls. It initializes an
    PromptBasedDatasetGenerator with specific settings, limiting the number of
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
    # Initialize the PromptBasedDatasetGenerator with specific settings.
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=3,
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
    )

    # Generate the dataset split using the initialized generator.
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples=5,
        split=DatasetSplit.TRAIN,
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
    assert list(generated_dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_third_batch(mocked_generate_example):
    """Test PromptBasedDatasetGenerator with filter methods in the third batch.

    This test verifies the behavior of the PromptBasedDatasetGenerator with
    filter methods in the third batch of API calls. It initializes an
    PromptBasedDatasetGenerator with specific settings, limiting the number
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
    # Initialize the PromptBasedDatasetGenerator with specific settings.
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=4,
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
    )

    # Generate the dataset split using the initialized generator.
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples=5,
        split=DatasetSplit.TRAIN,
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
    assert list(generated_dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_forth_batch(mocked_generate_example):
    """Test PromptBasedDatasetGenerator with filter methods in the forth batch."""
    # Initialize the PromptBasedDatasetGenerator with specific settings.
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=5,
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
    )

    # Generate the dataset split using the initialized generator.
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples=5,
        split=DatasetSplit.TRAIN,
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
    assert list(generated_dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions().mock_completions,
)
def test_generator_with_filter_unlimited_api_calls(mocked_generate_example):
    """Test PromptBasedDatasetGenerator with filter methods and unlimited API calls."""
    # Initialize the PromptBasedDatasetGenerator with
    # specific settings and unlimited API calls.
    dataset_generator = PromptBasedDatasetGenerator(
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
    )

    # Generate the dataset split using the initialized generator.
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples=5,
        split=DatasetSplit.TRAIN,
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
    assert list(generated_dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MockBatchDifferentCompletions(length=5).mock_completions,
)
def test_generator_with_filter_to_generate_datasetdict(mocked_generate_example):
    """Test with filter methods to generate a DatasetDict."""
    # Initialize the PromptBasedDatasetGenerator with
    # specific settings and limited API calls.
    dataset_generator = PromptBasedDatasetGenerator(
        filter_duplicated_examples=True,
        max_batch_size=2,
        responses_per_request=3,
        max_api_calls=7,
    )

    # Generate the DatasetDict using the initialized generator.
    generated_dataset_dict = dataset_generator.generate_dataset_dict(
        prompt_spec=MockPromptSpec(TaskType.TEXT_GENERATION),
        num_examples={
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
                    "input_col": ["1", "2", "3", "4"],
                    "output_col": ["b", "a", "a", "c"],
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
    assert list(generated_dataset_dict["train"]) == list(expected_dataset_dict["train"])
    assert list(generated_dataset_dict["val"]) == list(expected_dataset_dict["val"])
    assert list(generated_dataset_dict["test"]) == list(expected_dataset_dict["test"])


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_generator_max_api_calls_dict(mocked_generate_example):
    """Test generation of a dataset dict where we hit max api calls."""
    # Refresh the call_count and create a new limited_dataset_generator.
    dataset_generator = PromptBasedDatasetGenerator(
        filter_duplicated_examples=False,
        max_api_calls=13,
    )

    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = {
        DatasetSplit.TRAIN: 50,
        DatasetSplit.VAL: 24,
        DatasetSplit.TEST: 26,
    }

    dataset_dict = dataset_generator.generate_dataset_dict(
        prompt_spec=prompt_spec,
        num_examples=num_examples,
    )

    # Since the max_api_calls is 13, the limited_dataset_generator cannot
    # generate the whole dataset_dict and will call the API 13 times.
    assert dataset_generator.api_call_counter == 13

    # The train split has 50 examples, so it will call the API 10 times and call
    # generate_batch_completion 2 times.
    # The validation split has 24 examples, but there are only 3 API calls
    # left, so it will call the API 3 times and call
    # generate_batch_completion 1 time.
    # The test split has 26 examples, but there are no more API calls left,
    # so it will not call generate_batch_completion.
    assert mocked_generate_example.call_count == 2 + 1 + 0

    # Each API call returns 5 responses, and each response is a valid JSON.
    # So the generated_dataset_dict will contain (50, 15, 0) examples.
    assert len(dataset_dict["train"]) == 50
    assert len(dataset_dict["val"]) == 15
    assert len(dataset_dict["test"]) == 0


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_WRONG_KEY_EXAMPLE,
)
def test_wrong_key_example(mocked_generate_example):
    """Test PromptBasedDatasetGenerator when the agent returns wrong keys."""
    dataset_generator = PromptBasedDatasetGenerator(
        max_api_calls=3, filter_duplicated_examples=False
    )
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = 1
    split = DatasetSplit.TRAIN
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec, num_examples, split
    )
    assert mocked_generate_example.call_count == 3
    expected_dataset = Dataset.from_dict({"input_col": [], "output_col": []})
    assert list(expected_dataset) == list(generated_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_INVALID_JSON,
)
def test_invalid_json_response(mocked_generate_example):
    """Test when the agent returns invalid JSON responses."""
    # Init the PromptBasedDatasetGenerator with `max_api_calls = 3`.
    dataset_generator = PromptBasedDatasetGenerator(3, filter_duplicated_examples=False)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    num_examples = 1
    split = DatasetSplit.VAL
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 3
    expected_dataset = Dataset.from_dict({"input_col": [], "output_col": []})
    assert list(dataset) == list(expected_dataset)


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=UnknownGpt3Exception(),
)
def test_unexpected_examples_of_gpt(mocked_generate_example):
    """Test PromptBasedDatasetGenerator when the agent returns unexpected examples."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    # Init the PromptBasedDatasetGenerator with `max_api_calls = 3`.
    with pytest.raises(UnknownGpt3Exception):
        dataset_generator = PromptBasedDatasetGenerator(
            max_api_calls=3, filter_duplicated_examples=False
        )
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
        num_examples = 1
        split = DatasetSplit.TEST
        _ = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
    assert mocked_generate_example.call_count == 1


def test_filter_with_duplicate_inputs_unique_outputs():
    """Test filtering with duplicate inputs, unique outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=True)
    generated_examples = [
        Example(input_col="apple", output_col="A"),
        Example(input_col="banana", output_col="B"),
        Example(input_col="apple", output_col="E"),
        Example(input_col="orange", output_col="O"),
        Example(input_col="apple", output_col="D"),
    ]
    filtered_examples = data_generator.apply_multi_vote_filtering(generated_examples)
    expected_examples = [
        Example(input_col="apple", output_col="A"),
        Example(input_col="banana", output_col="B"),
        Example(input_col="orange", output_col="O"),
    ]
    assert sorted(expected_examples) == sorted(filtered_examples)


def test_filter_duplicate_inputs_duplicate_outputs():
    """Test constructing a map with duplicate inputs and duplicate outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=True)
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
    filtered_examples = data_generator.apply_multi_vote_filtering(generated_examples)
    expected_examples = [
        Example(input_col="apple", output_col="A"),
        Example(input_col="banana", output_col="B"),
        Example(input_col="orange", output_col="O"),
    ]
    assert expected_examples == filtered_examples


def test_create_all_examples_dataset_and_generated_dataset_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=True)
    generated_examples = [
        Example(input_col="apple", output_col="A"),
        Example(input_col="banana", output_col="B"),
        Example(input_col="orange", output_col="O"),
    ]
    filtered_examples = data_generator.apply_multi_vote_filtering(generated_examples)
    assert generated_examples == filtered_examples


def test_create_all_examples_dataset_and_generated_dataset_with_empty_examples_list():
    """Test constructing a map with empty inputs and outputs."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=True)
    generated_examples = []
    filtered_examples = data_generator.apply_multi_vote_filtering(generated_examples)
    assert generated_examples == filtered_examples


def test_compute_batch_size_with_limited_max_api_calls():
    """Test the batch size computation with limited max API calls."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(max_api_calls=28)
    data_generator.api_call_counter = 26
    # Default batch size and responses_per_request are both 5.
    # So each batch should contain 25 examples.

    # At least (125 - 110) / 5 = 3 API calls needed to get
    # more than 125 examples.

    batch_size = data_generator.compute_batch_size(
        num_examples=125, generated_dataset_size=110
    )
    assert (
        batch_size
        == data_generator.max_api_calls - data_generator.api_call_counter
        == 28 - 26
    )

    data_generator.api_call_counter = 20
    batch_size = data_generator.compute_batch_size(125, generated_dataset_size=110)
    assert (
        batch_size
        == (125 - 110) / data_generator.responses_per_request
        == (125 - 110) / 5
    )

    data_generator.api_call_counter = 0
    batch_size = data_generator.compute_batch_size(125, generated_dataset_size=50)
    assert batch_size == data_generator.max_batch_size


def test_compute_batch_size_with_unlimited_max_api_calls():
    """Test the batch size computation with unlimited max API calls."""
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator()
    # Default batch size and responses_per_request are both 5.
    # So each batch should contain 25 examples.

    # At least (125 - 110) / 5 = 3 API calls needed to get
    # more than 125 examples.

    batch_size = data_generator.compute_batch_size(125, generated_dataset_size=110)
    assert (
        batch_size
        == (125 - 110) / data_generator.responses_per_request
        == (125 - 110) / 5
    )

    batch_size = data_generator.compute_batch_size(125, generated_dataset_size=50)
    assert batch_size == data_generator.max_batch_size == 5


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

    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    data_generator = PromptBasedDatasetGenerator(filter_duplicated_examples=True)
    generated_examples = []
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        data_generator.extract_and_append_responses(
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
    data_generator.extract_and_append_responses([mock_completion_3], generated_examples)
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
        data_generator.extract_and_append_responses(
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
        data_generator = PromptBasedDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )
        generated_examples = []
        with patch.object(logger, "info") as mock_info, patch.object(
            logger, "warning"
        ) as mock_warning:
            data_generator.extract_and_append_responses(
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
        data_generator.extract_and_append_responses(
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
            data_generator.extract_and_append_responses(
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


def test_initialize_dataset_generator_with_dynamic_temperature():
    """Test the correct initialization of the dynamic temperature strategy."""
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        with pytest.raises(ValueError) as exc_info:
            _ = PromptBasedDatasetGenerator(
                cache_root=cache_dir, initial_temperature=-0.2
            )
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "initial_temperature must be >= 0, but self.initial_temperature=-0.2"
        )
        with pytest.raises(ValueError) as exc_info:
            _ = PromptBasedDatasetGenerator(cache_root=cache_dir, max_temperature=2.3)
            error_info = exc_info.value.args[0]
            assert (
                error_info
                == "max_temperature must be <= 2,0, but self.max_temperature=2.3"
            )

        with pytest.raises(ValueError) as exc_info:
            _ = PromptBasedDatasetGenerator(
                cache_root=cache_dir, max_temperature=1.2, initial_temperature=1.5
            )
            error_info = exc_info.value.args[0]
            assert (
                error_info
                == "self.initial_temperature=1.5 must be <= self.max_temperature=1.2"
            )


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=MOCK_CLASSIFICATION_EXAMPLE,
)
def test_dataset_generator_terminates(mocked_generate_example):
    """Check to make sure that the dataset generator terminates."""
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    dataset_generator = PromptBasedDatasetGenerator(
        initial_temperature=0.3,
        max_temperature=1.4,
        responses_per_request=3,
        max_api_calls=10000,
        requests_per_minute=80,
        filter_duplicated_examples=False,
    )
    generated_dataset = dataset_generator.generate_dataset_split(
        prompt_spec, 100, split=DatasetSplit.TRAIN
    )
    generated_df = generated_dataset.to_pandas()
    assert len(generated_dataset) == 100
    assert list(generated_df.columns) == ["input_col", "output_col"]


def test_generate_dataset_agent_switch():
    """Test if dataset generation can use a user-set API agent."""
    my_agent = MockAPIAgent(
        default_content='{"input": "This is input.", "output": "This is an output."}'
    )
    with temp_setattr(api_tools, "default_api_agent", my_agent):
        prompt_spec = MockPromptSpec(TaskType.CLASSIFICATION)
        dataset_generator = PromptBasedDatasetGenerator(
            initial_temperature=0.3,
            max_temperature=1.4,
            responses_per_request=1,
            max_api_calls=100,
            requests_per_minute=80,
            filter_duplicated_examples=False,
        )
        dataset_generator.generate_dataset_split(
            prompt_spec, 100, split=DatasetSplit.TRAIN
        )
    # 100 outputs, and each batch has 5 outputs so 20 api calls
    assert my_agent.generate_batch_call_counter == 20
