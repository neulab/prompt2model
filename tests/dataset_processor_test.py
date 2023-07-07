"""Testing TextualizeProcessor."""

import datasets
import pytest

from prompt2model.dataset_processor.textualize import TextualizeProcessor

DATASET_DICTS = [
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"input_col": ["foo", "bar"], "output_col": ["baz", "qux"]}
            ),
            "test": datasets.Dataset.from_dict(
                {"input_col": ["foo", "bar"], "output_col": ["baz", "qux"]}
            ),
        }
    ),
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"input_col": ["spam", "eggs"], "output_col": ["ham", "sau"]}
            ),
            "val": datasets.Dataset.from_dict(
                {"input_col": ["spam", "eggs"], "output_col": ["ham", "sau"]}
            ),
        }
    ),
]

# Our support spilts are `train, val, test`.
UNEXPECTED_DATASET_DICTS_WITH_WRONG_SPLIT = [
    datasets.DatasetDict(
        {
            "full": datasets.Dataset.from_dict(
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

# Our support columns are `input_col, output_col`.
UNEXPECTED_DATASET_DICTS_WITH_WRONG_COLUMNS = [
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
                {"input_col": ["spam", "eggs"], "output": ["ham", "sau"]}
            )
        }
    ),
]

INSTRUCTION = "convert to text2text"


def test_dataset_processor_t5_style():
    """Test the `process_dataset_dict` function of T5-type `TextualizeProcessor`."""
    t5_processor = TextualizeProcessor(has_encoder=True)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    t5_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo",
                            "<task 0> convert to text2text Example: bar",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz", "qux"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo",
                            "<task 0> convert to text2text Example: bar",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz", "qux"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam",
                            "<task 1> convert to text2text Example: eggs",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham", "sau"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam",
                            "<task 1> convert to text2text Example: eggs",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham", "sau"],
                    }
                ),
            }
        ),
    ]
    for index in range(len(t5_modified_dataset_dicts)):
        dataset_dict = t5_modified_dataset_dicts[index]
        dataset_splits = list(dataset_dict.keys())
        for dataset_split in dataset_splits:
            assert (
                dataset_dict[dataset_split]["model_input"]
                == t5_expected_dataset_dicts[index][dataset_split]["model_input"]
            )


def test_dataset_processor_decoder_only_style():
    """Test the `process_dataset_dict` function of a GPT-type `TextualizeProcessor`."""
    gpt_processor = TextualizeProcessor(has_encoder=False)
    gpt_modified_dataset_dicts = gpt_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )

    # Check that the modified dataset dicts have the expected content
    gpt_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo Label: baz",
                            "<task 0> convert to text2text Example: bar Label: qux",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz", "qux"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo Label: ",
                            "<task 0> convert to text2text Example: bar Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz", "qux"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ham",
                            "<task 1> convert to text2text Example: eggs Label: sau",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham", "sau"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ",
                            "<task 1> convert to text2text Example: eggs Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham", "sau"],
                    }
                ),
            }
        ),
    ]
    for index in range(len(gpt_modified_dataset_dicts)):
        dataset_dict = gpt_modified_dataset_dicts[index]
        dataset_splits = list(dataset_dict.keys())
        for dataset_split in dataset_splits:
            assert (
                dataset_dict[dataset_split]["model_input"]
                == gpt_expected_dataset_dicts[index][dataset_split]["model_input"]
            )


def test_unexpected_dataset_split():
    """Test the error handler for unexpercted dataset split."""
    with pytest.raises(AssertionError):
        gpt_processor = TextualizeProcessor(has_encoder=False)
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_SPLIT
        )


def test_unexpected_columns():
    """Test the error handler for unexpercted dataset columns."""
    with pytest.raises(AssertionError):
        gpt_processor = TextualizeProcessor(has_encoder=False)
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_COLUMNS
        )
