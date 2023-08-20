"""Testing a dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import tempfile

from datasets import Dataset, DatasetDict

from prompt2model.dataset_retriever import DescriptionDatasetRetriever

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
