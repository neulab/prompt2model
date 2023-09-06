"""An interface for model selection."""
from __future__ import annotations

from abc import ABC, abstractmethod

from prompt2model.prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class ModelRetriever(ABC):
    """Retrieve several models from HuggingFace."""

    @abstractmethod
    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> list[str]:
        """Retrieve relevant models from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A list of relevant models' HuggingFace names.
        """
