"""Testing DatasetGenerator through SimpleDatasetGenerator."""

import os
import tempfile

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.simple import SimpleDatasetGenerator


def test_generate_datasets():
    """Test the `generate_datasets()` function of a `SimpleDatasetGenerator` object.

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
    api_key = "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"
    prompt_spec = None
    num_examples = {DatasetSplit.TRAIN: 1, DatasetSplit.VAL: 1, DatasetSplit.TEST: 0}

    dataset_generator = SimpleDatasetGenerator(api_key)
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
            for example in dataset:
                assert example["output_col"] != "", "Expected example to not be empty"
        assert os.path.isdir(output_dir)
        assert set(os.listdir(output_dir)) == {
            "dataset_dict.json",
            "test",
            "train",
            "val",
        }


def test_generate_examples():
    """Test the `generate_examples()` function of a `SimpleDatasetGenerator` object.

    This function generates movie comments examples for a specified split of the data
    (train, validation, or test) using a simple prompt specification
    and saves them to a temporary directory. Then, it checks that the
    generated dataset has the expected number of examples, the expected
    columns, and each example is not empty.
    """
    api_key = "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"
    prompt_spec = None
    num_examples = 1
    split = DatasetSplit.TRAIN

    dataset_generator = SimpleDatasetGenerator(api_key)
    dataset = dataset_generator.generate_examples(prompt_spec, num_examples, split)

    # Check that the generated dataset has the expected number of examples.
    assert len(dataset) == num_examples

    # Check that the generated dataset has the expected columns.
    expected_columns = {"input_col", "output_col"}
    assert set(dataset.column_names) == expected_columns

    # Check that each example is not empty.
    for example in dataset:
        assert example["output_col"] != "", "Expected example to not be empty"
