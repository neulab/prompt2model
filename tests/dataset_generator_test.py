import pandas as pd
import os
import tempfile
from prompt2model.dataset_generator.mock import MockDatasetGenerator
from prompt2model.dataset_generator.base import DatasetSplit


def test_generate_examples():
    prompt_spec = None
    num_examples = 10
    split = DatasetSplit.TRAIN

    dataset_generator = MockDatasetGenerator()
    dataset = dataset_generator.generate_examples(prompt_spec, num_examples, split)

    # Check that the generated dataset has the expected number of examples.
    assert len(dataset) == num_examples

    # Check that the generated dataset has the expected columns.
    assert set(dataset.column_names) == {"input_col", "output_col"}

    # Check that each example has the expected empty values.
    expected_values = pd.Series("", index=dataset.column_names)
    for example in dataset:
        assert example == expected_values.to_dict()


def test_generate_datasets():
    prompt_spec = None
    num_examples = {DatasetSplit.TRAIN: 10, DatasetSplit.VAL: 5, DatasetSplit.TEST: 3}

    dataset_generator = MockDatasetGenerator()
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = os.path.join(tmpdirname, "output")
        dataset_dict = dataset_generator.generate_datasets(prompt_spec, num_examples, output_dir)

        # Check that the dataset dict has the expected keys.
        assert set(dataset_dict.keys()) == {"train", "val", "test"}

        # Check that the generated datasets have the expected number of examples.
        for split, num in num_examples.items():
            assert len(dataset_dict[split.value]) == num

        # Check that the generated datasets have the expected columns.
        expected_columns = {"input_col", "output_col"}
        for dataset in dataset_dict.values():
            assert set(dataset.column_names) == expected_columns

        # Check that the generated datasets have the expected empty values.
        expected_values = pd.Series("", index=expected_columns)
        for dataset in dataset_dict.values():
            for example in dataset:
                assert example == expected_values.to_dict()

        # Check that the dataset dict is saved to the output directory.
        assert os.path.isdir(output_dir)
        assert set(os.listdir(output_dir)) == {'dataset_dict.json', 'test', 'train', 'val'}