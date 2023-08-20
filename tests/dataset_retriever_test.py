"""Testing a dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import numpy as np
import pickle
import tempfile
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from prompt2model.dataset_retriever import DescriptionDatasetRetriever, DatasetInfo
from test_helpers import create_test_search_index, create_test_search_index_class_method

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_initialize_dataset_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        # This tiny dataset search index contains 3 datasets.
        assert len(retriever.dataset_infos) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            dataset_info_file="test_helpers/dataset_index_tiny.json",
        )
        model_vectors = retriever.encode_dataset_descriptions(f.name)
        assert model_vectors.shape == (3, 128)


def test_canonicalize_dataset_using_columns():
    """Test canonicalizing a dataset with specified column names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionDatasetRetriever(
            search_index_path=f.name,
            first_stage_search_depth=3,
            max_search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
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

def create_test_search_index(index_file_name):
    """Utility function to create a test search index.

    This search index represents 3 models, each represented with a hand-written vector.
    Given a query of [1, 0, 0], the 1st model will be the most similar.
    """
    mock_model_encodings = np.array([[0.6, 0, 0], [0, 0.7, 0], [0, 0, 0.8]])
    mock_lookup_indices = [0, 1, 2]
    with open(index_file_name, "wb") as f:
        pickle.dump((mock_model_encodings, mock_lookup_indices), f)

def mock_choose_dataset(self, top_datasets: list[DatasetInfo]) -> str | None:
    return top_datasets[0].name    

def mock_canonicalize_dataset(self, dataset_name: str) -> DatasetDict:
    assert dataset_name == "TEST"
    mock_dataset = Dataset.from_dict(
        {
            "premise": ["Pandas are bears."],
            "hypothesis": [ "Pandas are mammals."],
            "answer": ["entailment"],
        }
    )
    # Create a mock DatasetDict consisting of the same example in each split.
    dataset_splits = DatasetDict(
        {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
    )
    return dataset_splits

@patch.object(
    DescriptionDatasetRetriever,
    "encode_dataset_descriptions",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "choose_dataset",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "choose_dataset",
    new=mock_choose_dataset,
)
@patch.object(
    DescriptionDatasetRetriever,
    "canonicalize_dataset",
    new=mock_canonicalize_dataset,
)
def test_retrieve_dataset_dict_without_existing_search_index(mock_choose_dataset):