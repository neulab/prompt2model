"""An interface for model selection."""
from __future__ import annotations

from prompt2model.model_retriever import ModelRetriever
from prompt2model.prompt_parser import PromptSpec


class MockModelRetriever(ModelRetriever):
    """Select a fixed model from among a set of hyperparameter choices."""

    def __init__(self, fixed_model_name: str):
        """Initialize a dummy retriever that returns a fixed model name."""
        self.fixed_model_name = fixed_model_name

    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> list[str]:
        """Select an arbitrary, fixed model from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A relevant model's HuggingFace name.
        """
        return [self.fixed_model_name]
