"""An interface for model selection."""

from prompt2model.model_retriever import ModelRetriever
from prompt2model.prompt_parser import PromptSpec


class MockModelRetriever(ModelRetriever):
    """Select a fixed model from among a set of hyperparameter choices."""

    def __init__(self, pretrained_model_name: str):
        """Initialize a dummy retriever that returns a fixed model name.

        Args:
            pretrained_model_name: A pretrained HuggingFace model name.
        """
        self.pretrained_model_name = pretrained_model_name

    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> str:
        """Select an arbitrary, fixed model from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A relevant model's HuggingFace model name.
        """
        return self.pretrained_model_name
