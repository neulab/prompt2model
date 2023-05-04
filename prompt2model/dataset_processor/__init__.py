"""Import DatasetRetriever classes."""
from prompt2model.dataset_processor.base import BasePrcesser
from prompt2model.dataset_processor.mock import MockProcessor

__all__ = ("BasePrcesser", "MockProcessor")
