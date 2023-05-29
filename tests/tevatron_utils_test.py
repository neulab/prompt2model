"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import json
import os
import pickle
import tempfile

import pytest
from tevatron.modeling import DenseModelForInference
from transformers import PreTrainedTokenizerBase

from prompt2model.utils.tevatron_utils import encode_text
from prompt2model.utils.tevatron_utils.encode import load_tevatron_model

TINY_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def test_load_tevatron_model():
    """Test loading a small Tevatron model."""
    model, tokenizer = load_tevatron_model(TINY_MODEL_NAME)
    assert isinstance(model, DenseModelForInference)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_encode_text_from_string():
    """Test encoding text from a string into a vector."""
    text = "This is an example sentence"
    encoded = encode_text(TINY_MODEL_NAME, text_to_encode=text)
    assert encoded.shape == (1, 128)


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


def test_encode_text_error_from_no_string_or_file():
    """Test that either a string or a file must be passed to encode."""
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME)


def test_encode_text_error_from_both_string_and_file():
    """Test that either a string or a file, but not both, must be passed to encode."""
    text = "This is an example sentence"
    file = "/tmp/test.tx`   t"
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME, file_to_encode=file, text_to_encode=text)
