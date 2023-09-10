"""Testing TextualizeProcessor."""

import gc
import logging
from copy import deepcopy
from unittest.mock import patch

import datasets
import pytest

from prompt2model.dataset_processor.textualize import TextualizeProcessor
from test_helpers import create_gpt2_model_and_tokenizer, create_t5_model_and_tokenizer

logger = logging.getLogger("DatasetProcessor")

DATASET_DICTS = [
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "input_col": ["foo", "bar"],
                    "output_col": ["baz", "qux"],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "input_col": ["foo", "bar"],
                    "output_col": ["baz", "qux"],
                }
            ),
        }
    ),
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "input_col": ["spam", "eggs"],
                    "output_col": ["ham", "sau"],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "input_col": ["spam", "eggs"],
                    "output_col": ["ham", "sau"],
                }
            ),
        }
    ),
]


INSTRUCTION = "convert to text2text"

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


def test_the_logging_for_provide_unnecessary_eos_token_for_t5():
    """Test the logger.info for unnecessary eos token for T5 model is logged."""
    _, t5_tokenizer = create_t5_model_and_tokenizer()

    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        _ = TextualizeProcessor(has_encoder=True, eos_token=t5_tokenizer.eos_token)
        mock_info.assert_called_once_with(
            "The T5 tokenizer automatically adds eos token in the end of sequence when tokenizing. So the eos_token of encoder-decoder model tokenizer is unnecessary."  # noqa E501
        )
        mock_warning.assert_not_called()
    gc.collect()


def test_the_logging_for_eos_token_required_for_gpt():
    """Test the logger.warning for requiring eos token for GPT model is logged."""
    with patch.object(logger, "info") as mock_info, patch.object(
        logger, "warning"
    ) as mock_warning:
        _ = TextualizeProcessor(has_encoder=False)
        mock_info.assert_not_called()
        mock_warning.assert_called_once_with(
            "The autoregressive model tokenizer does not automatically add eos token in the end of the sequence. So the `eos_token` of the autoregressive model is required."  # noqa E501
        )
    gc.collect()


def test_dataset_processor_t5_style():
    """Test the `process_dataset_dict` function of T5-type `TextualizeProcessor`."""
    t5_processor = TextualizeProcessor(has_encoder=True)
    raw_dataset_dicts = deepcopy(DATASET_DICTS)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    # Ensure the dataset_dicts themselves are the same after processing.
    for raw, origin in zip(raw_dataset_dicts, DATASET_DICTS):
        assert list(raw["train"]) == list(origin["train"])
        if "val" in raw:
            assert list(raw["val"]) == list(origin["val"])
        if "test" in raw:
            assert list(raw["test"]) == list(origin["test"])
    t5_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                            "<task 0>convert to text2text\nExample:\nbar\nLabel:\n",
                        ],
                        "model_output": ["baz", "qux"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                            "<task 0>convert to text2text\nExample:\nbar\nLabel:\n",
                        ],
                        "model_output": ["baz", "qux"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1>convert to text2text\nExample:\nspam\nLabel:\n",
                            "<task 1>convert to text2text\nExample:\neggs\nLabel:\n",
                        ],
                        "model_output": ["ham", "sau"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1>convert to text2text\nExample:\nspam\nLabel:\n",
                            "<task 1>convert to text2text\nExample:\neggs\nLabel:\n",
                        ],
                        "model_output": ["ham", "sau"],
                    }
                ),
            }
        ),
    ]
    for exp, act in zip(t5_expected_dataset_dicts, t5_modified_dataset_dicts):
        assert list(exp["train"]) == list(act["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(act["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(act["test"])
    gc.collect()


def test_dataset_processor_with_numerical_column():
    """Test process_dataset_dict with numerical column values."""
    t5_processor = TextualizeProcessor(has_encoder=True)
    raw_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": ["foo", "bar"],
                        "output_col": ["baz", "qux"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "input_col": ["spam", "eggs"],
                        "output_col": ["ham", "sau"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "input_col": ["foo", "bar"],
                        "output_col": [0, 1],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "input_col": ["spam", "eggs"],
                        "output_col": [1, 2],
                    }
                ),
            }
        ),
    ]
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, raw_dataset_dicts
    )
    expected_dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                        "<task 0>convert to text2text\nExample:\nbar\nLabel:\n",
                        "<task 1>convert to text2text\nExample:\nfoo\nLabel:\n",
                        "<task 1>convert to text2text\nExample:\nbar\nLabel:\n",
                    ],
                    "model_output": ["baz", "qux", "0", "1"],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        "<task 0>convert to text2text\nExample:\nspam\nLabel:\n",
                        "<task 0>convert to text2text\nExample:\neggs\nLabel:\n",
                        "<task 1>convert to text2text\nExample:\nspam\nLabel:\n",
                        "<task 1>convert to text2text\nExample:\neggs\nLabel:\n",
                    ],
                    "model_output": ["ham", "sau", "1", "2"],
                }
            ),
        }
    )
    training_datasets = []
    test_datasets = []
    for modified_dataset_dict in t5_modified_dataset_dicts:
        training_datasets.append(modified_dataset_dict["train"])
        test_datasets.append(modified_dataset_dict["test"])

    concatenated_training_dataset = datasets.concatenate_datasets(training_datasets)
    concatenated_test_dataset = datasets.concatenate_datasets(test_datasets)
    actual_dataset_dict = datasets.DatasetDict(
        {"train": concatenated_training_dataset, "test": concatenated_test_dataset}
    )
    assert list(expected_dataset_dict["train"]) == list(actual_dataset_dict["train"])
    assert list(expected_dataset_dict["test"]) == list(actual_dataset_dict["test"])


def test_dataset_processor_decoder_only_style():
    """Test the `process_dataset_dict` function of a GPT-type `TextualizeProcessor`."""
    _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
    gpt_processor = TextualizeProcessor(
        has_encoder=False, eos_token=gpt2_tokenizer.eos_token
    )
    raw_dataset_dicts = deepcopy(DATASET_DICTS)
    gpt_modified_dataset_dicts = gpt_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    # Ensure the dataset_dicts themselves are the same after processing.
    for raw, origin in zip(raw_dataset_dicts, DATASET_DICTS):
        assert list(raw["train"]) == list(origin["train"])
        if "val" in raw:
            assert list(raw["val"]) == list(origin["val"])
        if "test" in raw:
            assert list(raw["test"]) == list(origin["test"])
    # Check that the modified dataset dicts have the expected content
    gpt_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\nbaz<|endoftext|>",  # noqa: E501
                            "<task 0>convert to text2text\nExample:\nbar\nLabel:\nqux<|endoftext|>",  # noqa: E501
                        ],
                        "model_output": ["baz<|endoftext|>", "qux<|endoftext|>"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                            "<task 0>convert to text2text\nExample:\nbar\nLabel:\n",
                        ],
                        "model_output": ["baz", "qux"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1>convert to text2text\nExample:\nspam\nLabel:\nham<|endoftext|>",  # noqa: E501
                            "<task 1>convert to text2text\nExample:\neggs\nLabel:\nsau<|endoftext|>",  # noqa: E501
                        ],
                        "model_output": ["ham<|endoftext|>", "sau<|endoftext|>"],
                    }
                ),
                "val": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 1>convert to text2text\nExample:\nspam\nLabel:\n",
                            "<task 1>convert to text2text\nExample:\neggs\nLabel:\n",
                        ],
                        "model_output": ["ham", "sau"],
                    }
                ),
            }
        ),
    ]
    for exp, modified in zip(gpt_expected_dataset_dicts, gpt_modified_dataset_dicts):
        assert list(exp["train"]) == list(modified["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(modified["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(modified["test"])


def test_unexpected_dataset_split():
    """Test the error handler for unexpercted dataset split."""
    with pytest.raises(ValueError) as exc_info:
        _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
        gpt_processor = TextualizeProcessor(
            has_encoder=False, eos_token=gpt2_tokenizer.eos_token
        )
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_SPLIT
        )
        assert str(exc_info.value) == ("Datset split must be in train/val/test.")
    gc.collect()


def test_unexpected_columns():
    """Test the error handler for unexpercted dataset columns."""
    with pytest.raises(ValueError) as exc_info:
        _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
        gpt_processor = TextualizeProcessor(
            has_encoder=False, eos_token=gpt2_tokenizer.eos_token
        )
        _ = gpt_processor.process_dataset_dict(
            INSTRUCTION, UNEXPECTED_DATASET_DICTS_WITH_WRONG_COLUMNS
        )
        assert str(exc_info.value) == (
            "Example dictionary must have 'input_col' and 'output_col' keys."
        )
    gc.collect()


DATASET_DICTS_WITH_EMPTY_COLUMNS = [
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "input_col": ["foo", "", "test"],
                    "output_col": ["", "qux", "key"],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "input_col": ["foo", ""],
                    "output_col": ["baz", "qux"],
                }
            ),
        }
    ),
    datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "input_col": ["", ""],
                    "output_col": ["ham", "sau"],
                }
            ),
        }
    ),
]


def test_empty_filter_t5_type():
    """Test that examples with empty input_col or output_col are discarded."""
    t5_processor = TextualizeProcessor(has_encoder=True)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS_WITH_EMPTY_COLUMNS
    )
    t5_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\ntest\nLabel:\n",
                        ],
                        "model_output": ["key"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                        ],
                        "model_output": [
                            "baz",
                        ],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [],
                        "model_output": [],
                    }
                ),
            }
        ),
    ]
    for exp, modified in zip(t5_expected_dataset_dicts, t5_modified_dataset_dicts):
        assert list(exp["train"]) == list(modified["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(modified["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(modified["test"])


def test_empty_filter_decoder_only_style():
    """Test the `process_dataset_dict` function of a GPT-type `TextualizeProcessor`."""
    _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
    gpt_processor = TextualizeProcessor(
        has_encoder=False, eos_token=gpt2_tokenizer.eos_token
    )
    gpt_modified_dataset_dicts = gpt_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS_WITH_EMPTY_COLUMNS
    )

    # Check that the modified dataset dicts have the expected content
    gpt_expected_dataset_dicts = [
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\ntest\nLabel:\nkey<|endoftext|>",  # noqa: E501
                        ],
                        "model_output": ["key<|endoftext|>"],
                    }
                ),
                "test": datasets.Dataset.from_dict(
                    {
                        "model_input": [
                            "<task 0>convert to text2text\nExample:\nfoo\nLabel:\n",
                        ],
                        "model_output": ["baz"],
                    }
                ),
            }
        ),
        datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {
                        "model_input": [],
                        "model_output": [],
                    }
                ),
            }
        ),
    ]
    for exp, modified in zip(gpt_expected_dataset_dicts, gpt_modified_dataset_dicts):
        assert list(exp["train"]) == list(modified["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(modified["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(modified["test"])
    gc.collect()


GENERATED_DATASET = datasets.Dataset.from_dict(
    {
        "input_col": list(range(10000)),
        "output_col": list(range(10000, 20000)),
    }
)

RETRIEVED_TRAIN_DATASET = datasets.Dataset.from_dict(
    {
        "input_col": list(range(20000, 30000)),
        "output_col": list(range(30000, 40000)),
    }
)

DATASET_LIST = [GENERATED_DATASET, RETRIEVED_TRAIN_DATASET]


def test_raise_value_error_of_process_dataset_lists():
    """Test that the ValueError is correctly raised."""
    _, gpt2_tokenizer = create_gpt2_model_and_tokenizer()
    gpt_processor = TextualizeProcessor(
        has_encoder=False, eos_token=gpt2_tokenizer.eos_token
    )
    with pytest.raises(ValueError) as exc_info:
        gpt_processor.process_dataset_lists(INSTRUCTION, DATASET_LIST, 0.8, 0.2)
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "train_proportion 0.8 + val_proportion 0.2 must be less than 1."
        )

    t5_processor = TextualizeProcessor(has_encoder=True)
    with pytest.raises(ValueError) as exc_info:
        t5_processor.process_dataset_lists(INSTRUCTION, DATASET_LIST, 0.8, 0.2)
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "train_proportion 0.8 + val_proportion 0.2 must be less than 1."
        )


def test_process_dataset_lists():
    """Test the `process_dataset_lists` function."""
    processor = TextualizeProcessor(has_encoder=True)
    modified_dataset_dicts = processor.process_dataset_lists(
        INSTRUCTION, DATASET_LIST, 0.6, 0.2
    )
    expected_modified_generated_dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(6000)
                    ],
                    "model_output": [f"{output}" for output in range(10000, 16000)],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(6000, 8000)
                    ],
                    "model_output": [f"{output}" for output in range(16000, 18000)],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(8000, 10000)
                    ],
                    "model_output": [f"{output}" for output in range(18000, 20000)],
                }
            ),
        }
    )
    expected_modified_retrieved_dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(20000, 26000)
                    ],
                    "model_output": [f"{output}" for output in range(30000, 36000)],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(26000, 28000)
                    ],
                    "model_output": [f"{output}" for output in range(36000, 38000)],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(28000, 30000)
                    ],
                    "model_output": [f"{output}" for output in range(38000, 40000)],
                }
            ),
        }
    )
    for exp, modified in zip(
        [
            expected_modified_generated_dataset_dict,
            expected_modified_retrieved_dataset_dict,
        ],
        modified_dataset_dicts,
    ):
        assert list(exp["train"]) == list(modified["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(modified["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(modified["test"])


def test_process_dataset_lists_with_maximum_example_num():
    """Test the maximum_example_num parameter."""
    processor = TextualizeProcessor(has_encoder=True)
    modified_dataset_dicts = processor.process_dataset_lists(
        INSTRUCTION, DATASET_LIST, 0.6, 0.2, {"train": 3000, "val": 500, "test": 1000}
    )
    # Before applying the maximum_example_num, train_num = 6000,
    # val_num = 2000, test_num = 2000.
    # After applying the maximum_example_num, train_num = 3000,
    # val_num = 2000, test_num = 2000.
    expected_modified_generated_dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(3000)
                    ],
                    "model_output": [f"{output}" for output in range(10000, 13000)],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(3000, 3500)
                    ],
                    "model_output": [f"{output}" for output in range(13000, 13500)],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 0>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(3500, 4500)
                    ],
                    "model_output": [f"{output}" for output in range(13500, 14500)],
                }
            ),
        }
    )
    expected_modified_retrieved_dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(20000, 23000)
                    ],
                    "model_output": [f"{output}" for output in range(30000, 33000)],
                }
            ),
            "val": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(23000, 23500)
                    ],
                    "model_output": [f"{output}" for output in range(33000, 33500)],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "model_input": [
                        f"<task 1>convert to text2text\nExample:\n{input}\nLabel:\n"
                        for input in range(23500, 24500)
                    ],
                    "model_output": [f"{output}" for output in range(33500, 34500)],
                }
            ),
        }
    )
    for exp, modified in zip(
        [
            expected_modified_generated_dataset_dict,
            expected_modified_retrieved_dataset_dict,
        ],
        modified_dataset_dicts,
    ):
        assert list(exp["train"]) == list(modified["train"])
        if "val" in exp:
            assert list(exp["val"]) == list(modified["val"])
        if "test" in exp:
            assert list(exp["test"]) == list(modified["test"])
