"""Testing TextualizeProcessor."""

import datasets
import pytest
from transformers import T5Tokenizer

from prompt2model.dataset_processor.textualize import TextualizeProcessor
from test_helpers import create_gpt2_model_and_tokenizer

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
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_processor = TextualizeProcessor(
        has_encoder=True, eos_token=t5_tokenizer.eos_token
    )
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    t5_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo Label: ",
                            "<task 0> convert to text2text Example: bar Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz</s>", "qux</s>"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo Label: ",
                            "<task 0> convert to text2text Example: bar Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz</s>", "qux</s>"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ",
                            "<task 1> convert to text2text Example: eggs Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham</s>", "sau</s>"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ",
                            "<task 1> convert to text2text Example: eggs Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham</s>", "sau</s>"],
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
    _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
    gpt_processor = TextualizeProcessor(
        has_encoder=False, eos_token=gpt2_tokenizer.eos_token
    )
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
                            "<task 0> convert to text2text Example: foo Label: baz<|endoftext|>",  # noqa: E501
                            "<task 0> convert to text2text Example: bar Label: qux<|endoftext|>",  # noqa: E501
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz<|endoftext|>", "qux<|endoftext|>"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0> convert to text2text Example: foo Label: ",
                            "<task 0> convert to text2text Example: bar Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["baz<|endoftext|>", "qux<|endoftext|>"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ham<|endoftext|>",  # noqa: E501
                            "<task 1> convert to text2text Example: eggs Label: sau<|endoftext|>",  # noqa: E501
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham<|endoftext|>", "sau<|endoftext|>"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1> convert to text2text Example: spam Label: ",
                            "<task 1> convert to text2text Example: eggs Label: ",
                        ],
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham<|endoftext|>", "sau<|endoftext|>"],
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
        _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
        gpt_processor = TextualizeProcessor(
            has_encoder=False, eos_token=gpt2_tokenizer.eos_token
        )
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_SPLIT
        )


def test_unexpected_columns():
    """Test the error handler for unexpercted dataset columns."""
    with pytest.raises(AssertionError):
        _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
        gpt_processor = TextualizeProcessor(
            has_encoder=False, eos_token=gpt2_tokenizer.eos_token
        )
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_COLUMNS
        )
