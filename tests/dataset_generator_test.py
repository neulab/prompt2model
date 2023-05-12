"""Testing DatasetGenerator classes."""

import os
import tempfile
from functools import partial
from unittest.mock import patch

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.prompt_parser import MockPromptSpec
from test_helpers import mock_openai_response

mock_classification_example = partial(
    mock_openai_response,
    content='{"sample": "This is a great movie!", "annotation": 1}',
)
mock_translation_example = partial(
    mock_openai_response, content='{"sample": "我爱你", "annotation": "I love you."}'
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
    side_effect=mock_translation_example,
)
def test_translation_dataset_generation(mocked_generate_example):
    """Test translation dataset generation using the OpenAIDatasetGenerator.

    Args:
        mocked_generate_example: None used parameter.
        But test_translation_dataset_generation should require one
        positional argument.
    """
    api_key = None
    dataset_generator = OpenAIDatasetGenerator(api_key)
    check_generate_dataset_dict(dataset_generator)
    check_generate_dataset(dataset_generator)


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=mock_classification_example,
)
def test_classification_dataset_generation(mocked_generate_example):
    """Test classification dataset generation using the OpenAIDatasetGenerator.

    Args:
        mocked_generate_example: None used parameter.
        But test_classification_dataset_generation should require one
        positional argument.
    """
    api_key = None
    dataset_generator = OpenAIDatasetGenerator(api_key)
    check_generate_dataset_dict(dataset_generator)
    check_generate_dataset(dataset_generator)
