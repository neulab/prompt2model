"""Testing T5Processor."""

import datasets

from prompt2model.dataset_processor.base import GPTProcessor, T5Processor


def test_T5_dataset_processor():
    """Test the `process_dataset_dict()` function of `T5Processor`.

    It sets up dataset dicts containing input and output columns and an
    instruction to convert to text2text fashion. It then initializes the
    `T5Processor` and calls `process_dataset_dict` on the instruction
    and dataset dicts. Finally, it checks that the modified dataset dicts
    have the expected content.
    """
    # Setup
    dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"input_col": ["foo", "bar"], "output_col": ["baz", "qux"]}
                )
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"input_col": ["spam", "eggs"], "output_col": ["ham", "sau"]}
                )
            }
        ),
    ]
    instruction = "convert to text2text"

    # Initialize the `T5Processor` and call `process_dataset_dict`
    processor = T5Processor()
    modified_dataset_dicts = processor.process_dataset_dict(instruction, dataset_dicts)

    # Check that the modified dataset dicts have the expected content
    expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": [
                            "<task 0> convert to text2text foo",
                            "<task 0> convert to text2text bar",
                        ],
                        "output_col": ["baz", "qux"],
                    }
                )
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": [
                            "<task 1> convert to text2text spam",
                            "<task 1> convert to text2text eggs",
                        ],
                        "output_col": ["ham", "sau"],
                    }
                )
            }
        ),
    ]
    assert (
        modified_dataset_dicts[0]["train"][0] == expected_dataset_dicts[0]["train"][0]
    )
    assert (
        modified_dataset_dicts[0]["train"][1] == expected_dataset_dicts[0]["train"][1]
    )
    assert (
        modified_dataset_dicts[1]["train"][0] == expected_dataset_dicts[1]["train"][0]
    )
    assert (
        modified_dataset_dicts[1]["train"][1] == expected_dataset_dicts[1]["train"][1]
    )


def test_GPT_dataset_processor():
    """Test the `process_dataset_dict()` function of `GPTProcessor`.

    It sets up dataset dicts containing input and output columns and an
    instruction to convert to text2text fashion. It then initializes the
    `GPTProcessor` and calls `process_dataset_dict` on the instruction
    and dataset dicts. Finally, it checks that the modified dataset dicts
    have the expected content.
    """
    # Setup
    dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"input_col": ["foo", "bar"], "output_col": ["baz", "qux"]}
                )
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"input_col": ["spam", "eggs"], "output_col": ["ham", "sau"]}
                )
            }
        ),
    ]
    instruction = "convert to text2text"

    # Initialize the `GPTProcessor` and call `process_dataset_dict`
    processor = GPTProcessor()
    modified_dataset_dicts = processor.process_dataset_dict(instruction, dataset_dicts)

    # Check that the modified dataset dicts have the expected content
    expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": [
                            "<task 0> convert to text2text Example: foo Label: baz",
                            "<task 0> convert to text2text Example: bar Label: qux",
                        ],
                        "output_col": ["baz", "qux"],
                    }
                )
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": [
                            "<task 1> convert to text2text Example: spam Label: ham",
                            "<task 1> convert to text2text Example: eggs Label: sau",
                        ],
                        "output_col": ["ham", "sau"],
                    }
                )
            }
        ),
    ]
    assert (
        modified_dataset_dicts[0]["train"][0] == expected_dataset_dicts[0]["train"][0]
    )
    assert (
        modified_dataset_dicts[0]["train"][1] == expected_dataset_dicts[0]["train"][1]
    )
    assert (
        modified_dataset_dicts[1]["train"][0] == expected_dataset_dicts[1]["train"][0]
    )
    assert (
        modified_dataset_dicts[1]["train"][1] == expected_dataset_dicts[1]["train"][1]
    )
