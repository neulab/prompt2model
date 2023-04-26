"""Testing DatasetGenerator through SimpleDatasetGenerator."""

import json
import os
import tempfile
from functools import partial
from unittest.mock import patch

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.iogenerator import InputOutputGenerator
from prompt2model.dataset_generator.openai import OpenAIDatasetGenerator


class MockCompletion:
    """Mock openai completion object."""

    def __init__(self, content: str):
        """Initialize a new instance of MockCompletion class.

        Args:
            content: The mocked content to be returned, i.e.,
            json.dumps({"comment": "This is a great movie!",
            "label": 1}).
        """
        self.choices = [{"message": {"content": content}}]

    def __repr__(self):
        """Return a string representation of the MockCompletion object.

        Returns:
            _string: A string representation of the object, including its choices.
        """
        _string = f"<MockObject choices={self.choices}>"
        return _string


def mock_example(prompt: str, content: dict) -> MockCompletion:
    """Generate a mock completion object containing a choice with example content.

    This function creates a `MockCompletion` object with a `content` attribute set to
    a JSON string representing an example label and comment. The `MockCompletion`
    object is then returned.

    Args:
        prompt: A mocked prompt that won't be used.
        content: The example content to be returned.

    Returns:
        a `MockCompletion` object.
    """
    _ = prompt
    example_content = json.dumps(content)
    mock_completion = MockCompletion(content=example_content)
    return mock_completion


mock_NLI_example = partial(
    mock_example, content={"input": "This is a great movie!", "output": 1}
)
mock_NLG_example = partial(
    mock_example, content={"input": "我爱你", "output": "I love you."}
)


def check_generate_examples(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_examples()` function of `OpenAIDatasetGenerator`.

    This function generates a Dataset for a specified split of the data
    (train, validation, or test) using a simple prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example is not empty.
    """
    prompt_spec = None
    num_examples = 1
    split = DatasetSplit.TRAIN
    dataset = dataset_generator.generate_examples(prompt_spec, num_examples, split)

    # Check that the generated dataset has the expected number of examples.
    assert len(dataset) == num_examples

    # Check that the generated dataset has the expected columns.
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns

    # Check that each example is not empty.
    for example in dataset:
        assert example["output_col"] != "", "Expected example to not be empty"


def check_generate_datasets(dataset_generator: OpenAIDatasetGenerator):
    """Test the `generate_datasets()` function of `OpenAIDatasetGenerator`.

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
    prompt_spec = None
    num_examples = {DatasetSplit.TRAIN: 1, DatasetSplit.VAL: 1, DatasetSplit.TEST: 0}
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_datasets(
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


def test_NLI_and_NLG_Dataset_Generation():
    """Checks the functionality of InputOutputGenerator.

    The function mocks the generate_example function.
    It checks the generation of datasets and examples.
    """

    @patch(
        "prompt2model.dataset_generator.openai.OpenAIDatasetGenerator.generate_example",
        side_effect=mock_NLI_example,
    )
    def check_NLI(mocked_generate_example=None):
        api_key = None
        dataset_generator = InputOutputGenerator(api_key)
        check_generate_datasets(dataset_generator)
        check_generate_examples(dataset_generator)

    @patch(
        "prompt2model.dataset_generator.openai.OpenAIDatasetGenerator.generate_example",
        side_effect=mock_NLG_example,
    )
    def check_NLG(mocked_generate_example=None):
        api_key = None
        dataset_generator = InputOutputGenerator(api_key)
        check_generate_datasets(dataset_generator)
        check_generate_examples(dataset_generator)

    check_NLG()
    check_NLI()
