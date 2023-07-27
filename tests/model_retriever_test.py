"""Testing description-based model retriever."""

import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from prompt2model.model_retriever import DescriptionModelRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_initialize_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            search_depth=3,
            model_name=TINY_MODEL_NAME,
            model_descriptions_index="huggingface_models/model_info_tiny/",
        )
        assert len(retriever.model_names) == len(retriever.model_descriptions)
        # This tiny directory of HuggingFace models contains 3 models.
        assert len(retriever.model_descriptions) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            search_depth=5,
            model_name=TINY_MODEL_NAME,
            model_descriptions_index="huggingface_models/model_info_tiny/",
        )
        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        model_vectors = retriever.encode_model_descriptions(mock_prompt)
        assert model_vectors.shape == (3, 128)


def create_test_search_index(index_file_name):
    """Utility function to create a test search index.

    This search index represents 3 models, each represented with a hand-written vector.
    Given a query of [0, 0, 1], the 3rd model will be the most similar.
    """
    mock_model_encodings = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]])
    mock_lookup_indices = [0, 1, 2]
    with open(index_file_name, "wb") as f:
        pickle.dump((mock_model_encodings, mock_lookup_indices), f)


@patch(
    "prompt2model.model_retriever.description_based_retriever.encode_text",
    return_value=np.array([[0, 0, 1]]),
)
def test_retrieve_model_from_query_when_similarity_threshold_is_met(mock_encode_text):
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            search_depth=3,
            model_name=TINY_MODEL_NAME,
            model_descriptions_index="huggingface_models/model_info_tiny/",
            use_HyDE=False,
        )
        indexed_models = retriever.model_names
        create_test_search_index(f.name)

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        top_model_names = retriever.retrieve(mock_prompt, similarity_threshold=0.5)
        assert mock_encode_text.call_count == 1
        # The 3rd item in the index is the closest to the query.
        assert top_model_names[0] == indexed_models[2]
        # The other two models should be returned later in the search results, but in
        # no particular order.}")
        assert indexed_models[0] in top_model_names[1:]
        assert indexed_models[1] in top_model_names[1:]
