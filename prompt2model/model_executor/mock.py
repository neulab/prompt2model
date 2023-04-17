"""A dummy class to generate model outputs (for testing purposes)."""

import datasets
import transformers

from prompt2model.model_executor import ModelExecutor, ModelOutputs


class MockModelExecutor(ModelExecutor):
    """An interface for automatic model evaluation."""

    def make_predictions(
        self,
        model: transformers.PreTrainedModel,
        test_set: datasets.Dataset,
    ) -> ModelOutputs:
        """Mock the execution of a model on a test set.

        Args:
            model: A model (not actually evaluated here).
            test_set: The dataset to make predictions on.

        Returns:
            An object containing model outputs.
        """
        predictions = [""] * len(test_set)
        return ModelOutputs(predictions=predictions)
