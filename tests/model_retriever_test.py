"""Testing description-based model retriever."""

from __future__ import annotations  # noqa FI58

import os
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
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
        )
        # This tiny directory of HuggingFace models contains 3 models.
        assert len(retriever.model_infos) == 3


def test_encode_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            first_stage_depth=3,
            search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
            device=torch.device("cpu"),
            use_bm25=False,
        )
        model_vectors = retriever.encode_model_descriptions(f.name)
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
def test_retrieve_model_from_query_dual_encoder(mock_encode_text):
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            first_stage_depth=3,
            search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
            use_bm25=False,
            use_HyDE=False,
        )
        indexed_models = retriever.model_infos
        create_test_search_index(f.name)

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        top_model_names = retriever.retrieve(mock_prompt)
        assert mock_encode_text.call_count == 1
        # The 3rd item in the index is the closest to the query.
        assert top_model_names[0] == indexed_models[2].name
        # The other two models should be returned later in the search results, but in
        # no particular order.}")
        assert indexed_models[0].name in top_model_names[1:]
        assert indexed_models[1].name in top_model_names[1:]


MOCK_HYPOTHETICAL_DOCUMENT = "This is a hypothetical model description."


def mock_encode_text_for_hyde(
    model_name_or_path: str,
    text_to_encode: list[str] | str | None = None,
    device: torch.device = torch.device("cpu"),
):
    """Mock encode_text to support the mocked hypothetical document generated."""
    _ = model_name_or_path, device  # suppress unused variable warnings
    if text_to_encode == MOCK_HYPOTHETICAL_DOCUMENT:
        return np.array([[0, 1, 0]])
    else:
        return np.array([[0, 0, 0.1]])


@patch(
    "prompt2model.model_retriever.description_based_retriever.encode_text",
    side_effect=mock_encode_text_for_hyde,
)
@patch(
    "prompt2model.model_retriever.description_based_retriever"
    + ".generate_hypothetical_model_description",
    return_value=MOCK_HYPOTHETICAL_DOCUMENT,
)
def test_retrieve_model_with_hyde_dual_encoder(
    mock_generate_hypothetical_doc, mock_encode_text
):
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            first_stage_depth=3,
            search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
            use_bm25=False,
            use_HyDE=True,
        )
        indexed_models = retriever.model_infos
        create_test_search_index(f.name)

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        top_model_names = retriever.retrieve(mock_prompt)
        assert mock_generate_hypothetical_doc.call_count == 1
        assert mock_encode_text.call_count == 1
        # With HyDE, the hypothetical document encoding should match the 2rd document
        # in our index. Without HyDE, the mocked query encoding would actually be
        # closest to the 3rd document in our index.
        assert top_model_names[0] == indexed_models[1].name
        # The other two models should be returned later in the search results, but in
        # no particular order.
        assert indexed_models[0].name in top_model_names[1:]
        assert indexed_models[2].name in top_model_names[1:]


def create_test_search_index_class_method(self, index_file_name):
    """Utility function to create a test search index as a simulated class method."""
    _ = self
    create_test_search_index(index_file_name)


@patch.object(
    DescriptionModelRetriever,
    "encode_model_descriptions",
    new=create_test_search_index_class_method,
)
@patch(
    "prompt2model.model_retriever.description_based_retriever.encode_text",
    return_value=np.array([[0, 0, 1]]),
)
def test_retrieve_model_when_no_search_index_is_found(mock_encode_text):
    """Test model retrieval when there's no search index found."""
    with tempfile.TemporaryDirectory() as tempdir:
        temporary_file = os.path.join(tempdir, "search_index.pkl")
        retriever = DescriptionModelRetriever(
            search_index_path=temporary_file,
            first_stage_depth=3,
            search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
            use_bm25=False,
        )
        indexed_models = retriever.model_infos

        mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
        top_model_names = retriever.retrieve(mock_prompt)
        assert mock_encode_text.call_count == 1
        # The 3rd item in the index is the closest to the query.
        assert top_model_names[0] == indexed_models[2].name
        # The other two models should be returned later in the search results, but in
        # no particular order.}")
        assert indexed_models[0].name in top_model_names[1:]
        assert indexed_models[1].name in top_model_names[1:]
