"""A dummy class to generate model outputs (for testing purposes)."""
from __future__ import annotations

import datasets

from prompt2model.model_executor import ModelExecutor, ModelOutput


class MockModelExecutor(ModelExecutor):
    """An interface for automatic model evaluation."""

    def make_prediction(
        self,
        test_set: datasets.Dataset,
        input_column: str,
    ) -> list[ModelOutput]:
        """Mock the execution of a model on a test set.

        Args:
            test_set: The dataset to make predictions on.
            input_column: The dataset column to use as input to the model.

        Returns:
            An object containing model outputs.
        """
        predictions = []
        for _ in test_set[input_column]:
            model_output = ModelOutput(prediction="", auxiliary_info={})
            predictions.append(model_output)
        return predictions

    def make_single_prediction(self, model_input: str) -> ModelOutput:
        """Mock evaluation on one example.

        Args:
            model_input: The input string to the model.

        Returns:
            A single model output, useful for exposing a model to a user interface.
        """
        _ = model_input
        model_output = ModelOutput(prediction="", auxiliary_info={})
        return model_output
