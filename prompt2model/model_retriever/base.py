"""An interface for model selection."""

from abc import ABC, abstractmethod

from prompt2model.prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class ModelRetriever(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> str:
        """Retrieve relevant models from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A relevant model's name on HuggingFace.
        """
