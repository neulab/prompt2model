"""Testing a dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import os
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from prompt2model.dataset_retriever import DatasetInfo, DescriptionDatasetRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import MockCompletion, create_test_search_index

# The following variables are specifically for the
# four automatic column selection tests.
DATASET_NAME = "squad"
DATASET_COLUMNS = "id, title, context, question, answers"
DATASET_DESCRIPTION = "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable."  # noqa: E501
EXAMPLE_ROWS = {
    "id": "5733be284776f41900661182",
    "title": "University_of_Notre_Dame",
    "context": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',  # noqa: E501
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",  # noqa: E501
    "answers.text": "Saint Bernadette Soubirous",
    "answers.answer_start": 515,
}

INSTRUCTION = "Your task is to generate a relevant answer to a given question. You will be provided with context for each question"  # noqa: E501
GPT3_RESPONSE_CORRECT = MockCompletion(
    """{\n        \"input\": [\"context\", \"question\"],\n        \"output\": [\"answers\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_WITHOUT_REQUIRED_COLS = MockCompletion(
    """{\n       \"output\": [\"answers\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_WITH_UNKNOWN_COLS = MockCompletion(
    """{\n   \"input\": [\"comprehension\", \"question\"],\n     \"output\": [\"answers\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)
GPT3_RESPONSE_WITH_MORE_THAN_ONE_OUTPUT = MockCompletion(
    """{\n   \"input\": [\"context\", \"question\"],\n     \"output\": [\"answers\", \"title\"],\n        \"irrelevant\": [\"id\", \"title\"],\n        \"ambiguous\": []\n}"""  # noqa: E501
)

TINY_DUAL_ENCODER_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_initialize_dataset_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        # This tiny dataset search index contains 3 datasets.
        assert len(retriever.dataset_infos) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.TemporaryDirectory() as tempdir:
        temporary_file = os.path.join(tempdir, "search_index.pkl")
        retriever = DescriptionDatasetRetriever(
            search_index_path=temporary_file,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        retriever.initialize_search_index()
        with open(temporary_file, "rb") as f:
            model_vectors, _ = pickle.load(f)
        assert model_vectors.shape == (3, 128)


def test_canonicalize_dataset_using_columns():
    """Test canonicalizing a dataset with specified column names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
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
        dataset_splits = DatasetDict(
            {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
        )
        canonicalized_dataset = retriever.canonicalize_dataset_using_columns(
            dataset_splits, ["question", "context"], "answer"
        )
        splits = ["train", "val", "test"]
        assert canonicalized_dataset.keys() == set(splits)

        for split in splits:
            assert len(canonicalized_dataset[split]) == 1
            row = canonicalized_dataset[split][0]
            assert (
                row["input_col"]
                == """question: What is the capital of New York?
context: Albany, the state capital, is the sixth-largest city in the State of New York."""  # noqa E501
            )
            assert row["output_col"] == "Albany"


def mock_choose_dataset(self, top_datasets: list[DatasetInfo]) -> str | None:
    """Mock the choose_dataset by always choosing the first choice."""
    # Given the max_search_depth of 3 used below, we should choose among 3 datasets.
    assert len(top_datasets) == 3
    return top_datasets[0].name


def mock_canonicalize_dataset(self, dataset_name: str, prompt_spec) -> DatasetDict:
    """Mock the canonicalize_dataset function by constructing a mock dataset."""
    # Given the dataset of
    # [ [0.9, 0, 0],
    #   [0, 0.9, 0],
    #   [0, 0, 0.9] ]
    # and the query
    # [1, 0, 0]
    # we should return the first dataset, which in our test index is search_qa.
    assert dataset_name == "search_qa"
    mock_dataset = Dataset.from_dict(
        {
            "input_col": [
                "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            ],
            "output_col": ["mammals"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset_splits = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    return dataset_splits


@patch(
    "prompt2model.dataset_retriever.description_dataset_retriever.encode_text",
    return_value=np.array([[1, 0, 0]]),
)
@patch.object(
    DescriptionDatasetRetriever,
    "choose_dataset_by_cli",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset_by_cli",
    new=mock_canonicalize_dataset,
)
def test_retrieve_dataset_dict_when_search_index_exists(encode_text):
    """Test retrieve dataset without an existing search index."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        assert [info.name for info in retriever.dataset_infos] == [
            "search_qa",
            "squad",
            "trivia_qa",
        ]
        create_test_search_index(f.name)

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        retrieved_dataset = retriever.retrieve_dataset_dict(mock_prompt)
        assert encode_text.call_count == 1
        for split_name in ["train", "val", "test"]:
            assert split_name in retrieved_dataset
            split = retrieved_dataset[split_name]
            assert len(split) == 1
            assert (
                split[0]["input_col"]
                == "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            )
            assert split[0]["output_col"] == "mammals"


@patch(
    "prompt2model.dataset_retriever.description_dataset_retriever.encode_text",
    return_value=np.array([[1, 0, 0]]),
)
@patch.object(
    DescriptionDatasetRetriever,
    "choose_dataset_by_cli",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset_by_cli",
    new=mock_canonicalize_dataset,
)
def test_retrieve_dataset_dict_without_existing_search_index(encode_text):
    """Test retrieve dataset without an existing search index."""
    with tempfile.TemporaryDirectory() as tempdir:
        temporary_file = os.path.join(tempdir, "search_index.pkl")
        retriever = DescriptionDatasetRetriever(
            search_index_path=temporary_file,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_DUAL_ENCODER_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        create_test_search_index(temporary_file)
        assert [info.name for info in retriever.dataset_infos] == [
            "search_qa",
            "squad",
            "trivia_qa",
        ]
        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        retrieved_dataset = retriever.retrieve_dataset_dict(mock_prompt)
        assert encode_text.call_count == 2
        for split_name in ["train", "val", "test"]:
            assert split_name in retrieved_dataset
            split = retrieved_dataset[split_name]
            assert len(split) == 1
            assert (
                split[0]["input_col"]
                == "question: What class of animals are pandas.\ncontext: Pandas are mammals."  # noqa E501
            )
            assert split[0]["output_col"] == "mammals"


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_CORRECT],
)
def test_automatic_column_selection_correct(mocked_parsing_method):
    """Check that normal automatic column selection runs fine."""
    expected_input_columns = ["context", "question"]
    expected_output_column = "answers"
    (
        input_columns,
        output_column,
    ) = DescriptionDatasetRetriever.automatic_column_selection(
        INSTRUCTION, DATASET_NAME, DATASET_DESCRIPTION, DATASET_COLUMNS, EXAMPLE_ROWS
    )

    assert type(input_columns) == list
    assert input_columns == expected_input_columns

    assert type(output_column) == str
    assert output_column == expected_output_column


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITH_UNKNOWN_COLS],
)
def test_automatic_column_selection_unknown_cols(mocked_parsing_method):
    """Check error thrown if there are unknown cols returned in input/output."""
    with pytest.raises(RuntimeError) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            DATASET_NAME,
            DATASET_DESCRIPTION,
            DATASET_COLUMNS,
            EXAMPLE_ROWS,
        )
        error_info = exc_info.value.args[0]
        assert error_info == "Incorrect columns being parsed"


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITHOUT_REQUIRED_COLS],
)
def test_automatic_column_selection_without_required_cols(mocked_parsing_method):
    """Check that if input/output columns are missing, an error is thrown."""
    with pytest.raises(StopIteration) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            DATASET_NAME,
            DATASET_DESCRIPTION,
            DATASET_COLUMNS,
            EXAMPLE_ROWS,
        )
        error_info = exc_info.value.args[0]
        assert error_info == "Maximum number of API calls reached."


@patch(
    "prompt2model.utils.APIAgent.generate_one_completion",
    side_effect=[GPT3_RESPONSE_WITH_MORE_THAN_ONE_OUTPUT],
)
def test_automatic_column_selection_incorrect_number_of_output_cols(
    mocked_parsing_method,
):
    """Check that if number of input/output columns are wrong, an error is thrown."""
    with pytest.raises(RuntimeError) as exc_info:
        _ = DescriptionDatasetRetriever.automatic_column_selection(
            INSTRUCTION,
            DATASET_NAME,
            DATASET_DESCRIPTION,
            DATASET_COLUMNS,
            EXAMPLE_ROWS,
        )
        error_info = exc_info.value.args[0]
        assert (
            error_info
            == "Input columns length was less than 1 or output column length was not 1"
        )
