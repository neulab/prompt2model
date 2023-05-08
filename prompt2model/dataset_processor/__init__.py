"""Import DatasetProcessor classes."""
from prompt2model.dataset_processor.base import BaseProcesser
from prompt2model.dataset_processor.mock import MockProcessor

__all__ = ("BaseProcesser", "MockProcessor")
