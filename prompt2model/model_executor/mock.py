"""A dummy class to generate model outputs (for testing purposes)."""

import datasets
import transformers

from prompt2model.model_executor import ModelExecutor, ModelOutput


class MockModelExecutor(ModelExecutor):
    """An interface for automatic model evaluation."""

    def make_predictions(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        test_set: datasets.Dataset,
        input_column: str,
    ) -> list[ModelOutput]:
        """Mock the execution of a model on a test set.

        Args:
            model: A model (not actually evaluated here).
            tokenizer: The model's associated tokenizer (not used here).
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.

        Returns:
            An object containing model outputs.
        """
        predictions = []
        for _ in range(len(test_set[input_column])):
            model_output = ModelOutput(
                prediction="", confidence=None, auxiliary_info={}
            )
            predictions.append(model_output)
        return predictions
