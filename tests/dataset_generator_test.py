"""Testing DatasetGenerator through DatasetGenerator."""

import os
import tempfile

import pandas as pd

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.mock import MockDatasetGenerator


def test_generate_datasets():
    """Test the `generate_datasets()` function of a `MockDatasetGenerator` object.

    This function generates datasets by creating a specified
    number of examples foreach split of the data, which includes
    train, validation, and test. It uses "None"as the prompt
    specification and saves the generated datasets to a temporary
    directory. Afterward, the function checks whether the dataset
    dictionary contains all the expected keys, each split has the
    anticipated number of examples, every dataset has the anticipated
    columns, each example has the expected empty values, and
    whether the dataset dictionary is saved to the output directory.
    """
    prompt_spec = None
    num_examples = {DatasetSplit.TRAIN: 10, DatasetSplit.VAL: 5, DatasetSplit.TEST: 3}

    dataset_generator = MockDatasetGenerator()
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_datasets(
            prompt_spec=prompt_spec, num_examples=num_examples, output_dir=output_dir
        )

        assert set(dataset_dict.keys()) == {"train", "val", "test"}
        for split, num in num_examples.items():
            assert (
                len(dataset_dict[split.value]) == num
            ), f"Expected {num} examples for {split.value} split, but \
                got {len(dataset_dict[split.value])}"
        expected_columns = {"input_col", "output_col"}
        for dataset in dataset_dict.values():
            assert (
                set(dataset.column_names) == expected_columns
            ), f"Expected columns {expected_columns}, but got {dataset.column_names}"
            expected_values = pd.Series("", index=expected_columns)
            for example in dataset:
                assert (
                    example == expected_values.to_dict()
                ), f"Expected example {expected_values.to_dict()}, but got {example}"
        assert os.path.isdir(output_dir)
        assert set(os.listdir(output_dir)) == {
            "dataset_dict.json",
            "test",
            "train",
            "val",
        }


def test_generate_examples():
    """Test the `generate_examples()` function of a `MockDatasetGenerator` object.

    This function generates examples for a specified split of the data
    (train, validation, or test) using "None" as the prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example has the expected empty values.
    """
    prompt_spec = None
    num_examples = 10
    split = DatasetSplit.TRAIN

    dataset_generator = MockDatasetGenerator()
    dataset = dataset_generator.generate_examples(prompt_spec, num_examples, split)

    # Check that the generated dataset has the expected number of examples.
    assert len(dataset) == num_examples

    # Check that the generated dataset has the expected columns.
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns

    # Check that each example has the expected empty values.
    expected_values = pd.Series("", index=expected_columns)
    for example in dataset:
        assert example == expected_values.to_dict()
