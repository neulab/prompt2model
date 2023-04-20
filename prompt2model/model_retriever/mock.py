"""An interface for model selection."""

from transformers import AutoModel, PreTrainedModel

from prompt2model.model_retriever import ModelRetriever
from prompt2model.prompt_parser.base import PromptSpec


class MockModelRetriever(ModelRetriever):
    """Select a fixed model from among a set of hyperparameter choices."""

    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> PreTrainedModel:
        """Select an arbitrary, fixed model from HuggingFace.

        Args:
            prompt: A prompt to use to select relevant models.

        Return:
            A relevant model (here, we return an arbitrary, fixed model).
        """
        example_model = AutoModel.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        return example_model
