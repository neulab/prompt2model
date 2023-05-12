"""Testing DatasetGenerator through SimpleDatasetGenerator."""

import os
import tempfile
from functools import partial
from unittest.mock import patch

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec
from test_helpers import mock_openai_response
from datasets import Dataset

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


def check_generate_dataset(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_split()` function of `OpenAIDatasetGenerator`.

    This function generates a Dataset for a specified split of the data
    (train, validation, or test) using a simple prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example is not empty.
    """
    prompt_spec = MockPromptSpec()
    num_examples = 1
    split = DatasetSplit.TRAIN
    dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)

    # Check that the generated dataset has the expected number of examples.
    assert len(dataset) == num_examples

    # Check that the generated dataset has the expected columns.
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns

    # Check that each example is not empty.
    for example in dataset:
        assert example["output_col"] != "", "Expected example to not be empty"


def check_generate_dataset_dict(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_dataset_dict()` function of `OpenAIDatasetGenerator`.

    This function generates movie comments datasets by creating a specified
    number of examples for each split of the data, which includes
    train, validation, and test. It uses a simple prompt
    specification and saves the generated datasets to a temporary
    directory. Afterward, the function checks whether the dataset
    dictionary contains all the expected keys, each split has the
    anticipated number of examples, every dataset has the anticipated
    columns, each example is not empty, and whether the dataset dictionary
    is saved to the output directory.
    """
    prompt_spec = MockPromptSpec()
    num_examples = {DatasetSplit.TRAIN: 1, DatasetSplit.VAL: 1, DatasetSplit.TEST: 0}
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_dataset_dict(
            prompt_spec=prompt_spec, num_examples=num_examples, output_dir=output_dir
        )

        assert set(dataset_dict.keys()) == {"train", "val", "test"}
        for split, num in num_examples.items():
            assert len(dataset_dict[split.value]) == num, (
                f"Expected {num} examples for {split.value} split, but"
                f" got {len(dataset_dict[split.value])}"
            )
        expected_columns = {"input_col", "output_col"}
        for dataset in dataset_dict.values():
            assert (
                set(dataset.column_names) == expected_columns
            ), f"Expected columns {expected_columns}, but got {dataset.column_names}"
            for example in dataset:
                assert example["output_col"] != "", "Expected example to not be empty"
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
def test_classification_dataset_generation(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    Args:
        mocked_generate_example: The function represents the @patch function.
    """
    api_key = None
    dataset_generator = OpenAIDatasetGenerator(api_key)
    check_generate_dataset_dict(dataset_generator)
    check_generate_dataset(dataset_generator)
    assert mocked_generate_example.call_count == 3


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