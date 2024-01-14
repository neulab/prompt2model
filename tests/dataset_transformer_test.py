"""Tests for dataset_transformer."""

from functools import partial
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from prompt2model.dataset_transformer import PromptBasedDatasetTransformer
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import MockCompletion, mock_batch_api_response_identical_completions

PLAN_RESPONSE = MockCompletion(
    """plan:
1. Combine "context" and "question" into "input".
2. Combine "answer" into "output"."""
)

TRANSFORMED_DATA = partial(
    mock_batch_api_response_identical_completions,
    content='{"input": "context: Albany, the state capital, is the sixth-largest city in the State of New York.\nquestion: What is the capital of New York?", "output": "Albany"}',  # noqa E501
)


def test_canonicalize_dataset_using_samples():
    """Test canonicalize_dataset_using_samples."""
    dataset_transformer = PromptBasedDatasetTransformer()
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    dataset_dict = dataset_transformer.make_dataset_from_samples(inputs, outputs)
    assert dataset_dict["train"].column_names == ["input_col", "output_col"]
    assert dataset_dict["train"].shape == (2, 2)
    assert dataset_dict["train"][0]["input_col"] == "input1"
    assert dataset_dict["train"][0]["output_col"] == "output1"
    assert dataset_dict["train"][1]["input_col"] == "input2"
    assert dataset_dict["train"][1]["output_col"] == "output2"


@patch(
    "prompt2model.utils.APIAgent.generate_batch_completion",
    side_effect=TRANSFORMED_DATA,
)
@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[PLAN_RESPONSE],
)
def test_transform_data(mock_batch_completion, mock_one_completion):
    """Test transform_data."""
    dataset_transformer = PromptBasedDatasetTransformer()
    prompt_spec = MockPromptSpec(
        TaskType.TEXT_GENERATION,
        instruction="instruction",
        examples="example1\nexample2",
    )
    mock_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of New York?"],
            "context": [
                "Albany, the state capital, is the sixth-largest city in the State of New York."  # noqa E501
            ],
            "answer": ["Albany"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    dataset_dict = dataset_transformer.transform_data(
        prompt_spec=prompt_spec,
        dataset=dataset["train"],
        num_points_to_transform=1,
    )
    assert dataset_dict["train"].column_names == ["input_col", "output_col"]
    assert dataset_dict["train"].shape == (1, 2)
    assert (
        dataset_dict["train"][0]["input_col"]
        == "context: Albany, the state capital, is the sixth-largest city in the State of New York.\nquestion: What is the capital of New York?"  # noqa E501
    )
    assert dataset_dict["train"][0]["output_col"] == "Albany"
