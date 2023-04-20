"""An interface for model selection."""

from prompt2model.model_retriever import ModelRetriever
from prompt2model.prompt_parser import PromptSpec


class MockModelRetriever(ModelRetriever):
    """Select a fixed model from among a set of hyperparameter choices."""

    def __init__(self, fixed_model_id: str):
        """Initialize a dummy retriever that returns a fixed model ID."""
        self.fixed_model_id = fixed_model_id

    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> str:
        """Select an arbitrary, fixed model from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A relevant model's HuggingFace ID string.
        """
        return self.fixed_model_id
