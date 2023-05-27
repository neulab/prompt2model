"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import os
import tempfile
from functools import partial
from unittest.mock import patch

import pytest
from tevatron.modeling import DenseModelForInference
from transformers import PreTrainedTokenizerBase

from prompt2model.utils.tevatron_utils import (  # noqa: F401
    encode_search_corpus, encode_text, retrieve_objects)
from prompt2model.utils.tevatron_utils.encode import load_tevatron_model
from prompt2model.prompt_parser import MockPromptSpec


TINY_MODEL_NAME = "sshleifer/tiny-distilbert-base-cased"

def test_load_tevatron_model():
    model, tokenizer = load_tevatron_model(TINY_MODEL_NAME)
    assert isinstance(model, DenseModelForInference)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

def test_encode_text_from_string():
    text = "This is an example sentence"
    encoded = encode_text(TINY_MODEL_NAME, text_to_encode=text)
    assert encoded.shape == (1, 768)

def test_encode_text_from_file():
    text = "This is an example sentence"
    with tempfile.TemporaryFile() as f:
        f.write(text)
        f.seek(0)
        encoded = encode_text(TINY_MODEL_NAME, file_to_encode=f)
        assert encoded.shape == (1, 768)

def test_encode_text_error_from_no_string_or_file():
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME)

def test_encode_text_error_from_both_string_and_file():
    text = "This is an example sentence"
    file = "/tmp/test.txt"
    with pytest.raises(ValueError):
        _ = encode_text(TINY_MODEL_NAME, file_to_encode=file, text_to_encode=text)
