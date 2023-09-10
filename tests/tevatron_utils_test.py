"""Testing DatasetGenerator through PromptBasedDatasetGenerator."""

import gc
import json
import os
import pickle
import tempfile

import numpy as np
import pytest
from tevatron.modeling import DenseModelForInference
from transformers import PreTrainedTokenizerBase

from prompt2model.utils.tevatron_utils import encode_text, retrieve_objects
from prompt2model.utils.tevatron_utils.encode import load_tevatron_model

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_load_tevatron_model():
    """Test loading a small Tevatron model."""
    model, tokenizer = load_tevatron_model(TINY_MODEL_NAME)
    assert isinstance(model, DenseModelForInference)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    gc.collect()


def test_encode_text_from_string():
    """Test encoding text from a string into a vector."""
    text = "This is an example sentence"
    encoded = encode_text(TINY_MODEL_NAME, text_to_encode=text)
    assert encoded.shape == (1, 128)
    gc.collect()


def test_encode_text_from_file():
    """Test encoding text from a file into a vector."""
    text_rows = [
        {"text_id": 0, "text": "This is an example sentence"},
        {"text_id": 1, "text": "This is another example sentence"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(text_rows, f)
        f.seek(0)
        encoded = encode_text(TINY_MODEL_NAME, file_to_encode=f.name)
        assert encoded.shape == (2, 128)
    gc.collect()


def test_encode_text_from_file_store_to_file():
    """Test encoding text from a file into a vector, then stored to file."""
    text_rows = [
        {"text_id": 0, "text": "This is an example sentence"},
        {"text_id": 1, "text": "This is another example sentence"},
    ]
    with tempfile.TemporaryDirectory() as tempdir:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(text_rows, f)
            f.seek(0)
            encoding_file_path = os.path.join(tempdir, "encoding.pkl")
            encoded = encode_text(
                TINY_MODEL_NAME, file_to_encode=f.name, encoding_file=encoding_file_path
            )
            assert encoded.shape == (2, 128)
            encoded_vectors, encoded_indices = pickle.load(
                open(encoding_file_path, "rb")
            )
            assert (encoded == encoded_vectors).all()
            assert encoded_indices == [0, 1]
    gc.collect()


def test_encode_text_error_from_no_string_or_file():
    """Test that either a string or a file must be passed to encode."""
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME)
    gc.collect()


def test_encode_text_error_from_both_string_and_file():
    """Test that either a string or a file, but not both, must be passed to encode."""
    text = "This is an example sentence"
    file = "/tmp/test.txt"
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME, file_to_encode=file, text_to_encode=text)
    gc.collect()


def test_retrieve_objects():
    """Test retrieval against a list of vectors."""
    mock_query_vector = np.array([[0.0, 0.0, 1.0, 0.0]])
    # The query vector matches the third row in the search collection.
    mock_search_collection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    document_names = ["a", "b", "c", "d"]
    mock_vector_indices = [0, 1, 2, 3]
    with tempfile.TemporaryDirectory() as tmpdir:
        search_index_pickle = os.path.join(tmpdir, "search_index.pkl")
        pickle.dump(
            (mock_search_collection, mock_vector_indices),
            open(search_index_pickle, "wb"),
        )
        results = retrieve_objects(
            mock_query_vector, search_index_pickle, document_names, depth=3
        )
        assert (
            len(results) == 3
        ), "The number of results should match the provided depth."

        # Verify that the index of the first retrieved document matches the document
        # that we known matches the query vector.
        first_retrieved_document, _ = results[0]
        assert first_retrieved_document == "c"

        # Verify that the first retrieved document has the greatest retrieval score.
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        assert sorted_results[0][0] == first_retrieved_document
    gc.collect()
