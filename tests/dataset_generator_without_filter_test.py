"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import os
import tempfile
from functools import partial
from unittest.mock import patch

import pytest

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import mock_batch_openai_response_with_identical_completions

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
def test_api_call_counter(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    This function first test the unlimited generation. Then test generation
    when expected_num_examples >= max_api_calls. Thus the API agent will only be
    called max_api_calls times.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    unlimited_dataset_generator = OpenAIDatasetGenerator()
    unlimited_generated_dataset = check_generate_dataset(unlimited_dataset_generator)
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

    # Refresh the call_count and api_call_counter.
    mocked_generate_example.call_count = 0
    unlimited_dataset_generator.api_call_counter = 0

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

    # Refresh the call_count.
    mocked_generate_example.call_count = 0

    limited_dataset_generator = OpenAIDatasetGenerator(max_api_calls=3)
    limited_generated_dataset = check_generate_dataset(limited_dataset_generator)
    # The max_api_calls is 3. So the limited_dataset_generator will call API 3 times.
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

    # Refresh the call_count and create a new limited_dataset_generator.
    mocked_generate_example.call_count = 0
    limited_dataset_generator = OpenAIDatasetGenerator(max_api_calls=13)

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
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    expected_num_examples = 1
    split = DatasetSplit.TRAIN
    dataset = dataset_generator.generate_dataset_split(
        prompt_spec, expected_num_examples, split
    )
    assert mocked_generate_example.call_count == 3
    assert dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
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
    dataset_generator = OpenAIDatasetGenerator(api_key, 3)
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    expected_num_examples = 1
    split = DatasetSplit.VAL
    dataset = dataset_generator.generate_dataset_split(
        prompt_spec, expected_num_examples, split
    )
    assert mocked_generate_example.call_count == 3
    assert dataset["input_col"] == dataset["output_col"] and dataset["input_col"] == []
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
    with pytest.raises(UNKNOWN_GPT3_EXCEPTION):
        dataset_generator = OpenAIDatasetGenerator(max_api_calls=3)
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
    gc.collect()
