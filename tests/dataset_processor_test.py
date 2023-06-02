"""Testing TextualizeProcessor."""

import datasets

from prompt2model.dataset_processor.textualize import TextualizeProcessor

DATASET_DICTS = [
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

INSTRUCTION = "convert to text2text"


def test_dataset_processor_t5_style():
    """Test the `process_dataset_dict` function of T5-type `TextualizeProcessor`."""
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
                )
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
                )
            }
        ),
    ]
    for index in range(len(gpt_expected_dataset_dicts)):
        assert (
            gpt_modified_dataset_dicts[index]["train"]["model_input"]
            == gpt_expected_dataset_dicts[index]["train"]["model_input"]
        )

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
                )
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
                )
            }
        ),
    ]
    for index in range(len(t5_modified_dataset_dicts)):
        assert (
            t5_modified_dataset_dicts[index]["train"]["model_input"]
            == t5_expected_dataset_dicts[index]["train"]["model_input"]
        )


def test_dataset_processor_decoder_only_style():
    """Test the `process_dataset_dict` function of a GPT-type `TextualizeProcessor`."""
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
                )
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
                )
            }
        ),
    ]
    for index in range(len(t5_modified_dataset_dicts)):
        assert (
            t5_modified_dataset_dicts[index]["train"]["model_input"]
            == t5_expected_dataset_dicts[index]["train"]["model_input"]
        )