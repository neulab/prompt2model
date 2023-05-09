"""Import DatasetProcessor classes."""
from prompt2model.dataset_processor.base import BaseProcessor
from prompt2model.dataset_processor.mock import MockProcessor

__all__ = ("BaseProcessor", "MockProcessor")
