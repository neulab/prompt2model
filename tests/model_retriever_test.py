"""Testing description-based model retriever."""

from __future__ import annotations  # noqa FI58

import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from prompt2model.model_retriever import DescriptionModelRetriever
from prompt2model.model_retriever.generate_hypothetical_document import (
    generate_hypothetical_model_description,
)
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import api_tools
from test_helpers import create_test_search_index
from test_helpers.mock_api import MockAPIAgent
from test_helpers.test_utils import temp_setattr

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_initialize_model_retriever():
    """Test loading a small Tevatron model."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        retriever = DescriptionModelRetriever(
            search_index_path=f.name,
            search_depth=3,
            encoder_model_name=TINY_MODEL_NAME,
            model_descriptions_index_path="test_helpers/model_info_tiny/",
            use_bm25=False,
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


@patch.object(
    DescriptionModelRetriever,
    "encode_model_descriptions",
    new=lambda self, index_file_name: create_test_search_index(index_file_name),
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
            search_depth=3,
            first_stage_depth=3,
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


def test_construct_bm25_index_when_no_index_exists():
    """Test that construct_bm25_index creates a BM25 search index on disk."""
    retriever = DescriptionModelRetriever(
        first_stage_depth=3,
        search_depth=3,
        model_descriptions_index_path="test_helpers/model_info_tiny/",
        use_bm25=True,
        bm25_index_name="missing-index",
        use_HyDE=False,
    )
    assert retriever.bm25_index_exists() is False
    retriever.construct_bm25_index(retriever.model_infos)
    assert retriever.bm25_index_exists() is True
    # Clear search index from disk.
    shutil.rmtree(retriever.search_index_path)


def test_retrieve_bm25_when_index_exists():
    """Test model retrieval with BM25 after manually constructing a search index."""
    retriever = DescriptionModelRetriever(
        first_stage_depth=3,
        search_depth=3,
        model_descriptions_index_path="test_helpers/model_info_tiny/",
        use_bm25=True,
        bm25_index_name="missing-index-2",
        use_HyDE=False,
    )
    retriever.construct_bm25_index(retriever.model_infos)
    assert retriever.bm25_index_exists() is True

    mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
    mock_prompt._instruction = "text generator"
    # Retrieve models after constructing the search index.
    top_model_names = retriever.retrieve(mock_prompt)

    # This query only has term overlap with one model description - t5-base,
    # who's model description is "text to text generator".
    # Therefore that should be the only model we return.
    assert len(top_model_names) == 1
    assert top_model_names[0] == "t5-base"
    # Clear search index from disk.
    shutil.rmtree(retriever.search_index_path)


def test_retrieve_bm25_when_no_index_exists():
    """Test model retrieval with BM25 without a pre-existing search index."""
    retriever = DescriptionModelRetriever(
        first_stage_depth=3,
        search_depth=3,
        model_descriptions_index_path="test_helpers/model_info_tiny/",
        use_bm25=True,
        bm25_index_name="missing-index-3",
        use_HyDE=False,
    )
    assert retriever.bm25_index_exists() is False

    mock_prompt = MockPromptSpec(task_type=TaskType.TEXT_GENERATION)
    mock_prompt._instruction = "text generator"
    # Retrieve models without constructing the search index beforehand.
    top_model_names = retriever.retrieve(mock_prompt)
    # The index will be constructed after calling retriever.retrieve, even if
    # no search index existed before.
    assert retriever.bm25_index_exists() is True

    # This query only has term overlap with one model description - t5-base,
    # who's model description is "text to text generator".
    # Therefore that should be the only model we return.
    assert len(top_model_names) == 1
    assert top_model_names[0] == "t5-base"
    # Clear search index from disk.
    shutil.rmtree(retriever.search_index_path)


def test_generate_hypothetical_document_agent_switch():
    """Test if generate_hypothetical_document can use a user-set API agent."""
    my_agent = MockAPIAgent(default_content="test response")
    with temp_setattr(api_tools, "default_api_agent", my_agent):
        prompt_spec = MockPromptSpec(TaskType.CLASSIFICATION)
        generate_hypothetical_model_description(prompt_spec, max_api_calls=3)
    assert my_agent.generate_one_call_counter == 1
