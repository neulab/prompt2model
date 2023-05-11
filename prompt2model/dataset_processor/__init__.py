"""Import DatasetRetriever classes."""
from prompt2model.dataset_processor.base import DatasetProcessor
from prompt2model.dataset_processor.mock import MockProcessor

__all__ = ("DatasetProcessor", "MockProcessor")
