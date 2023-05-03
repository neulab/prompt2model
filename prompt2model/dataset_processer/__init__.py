"""Import DatasetRetriever classes."""
from prompt2model.dataset_processer.base import BasePrcesser
from prompt2model.dataset_processer.mock import MockPrcesser

__all__ = ("BasePrcesser", "MockPrcesser")
