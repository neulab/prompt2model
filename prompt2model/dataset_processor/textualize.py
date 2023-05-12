"""A dataset processor to convert datasets into Text2Text fashion."""

from prompt2model.dataset_processor.base import BaseProcessor


class TextualizeProcessor(BaseProcessor):
    """A class for post-processing datasets, convert them into Text2Text fashion."""

    @staticmethod
    def post_process_example(
        example: dict, instruction: str, task_id: int, has_encoder: bool
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: The dataset index in dataset_dicts, used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.

        Returns:
            A dictionary with `model_input` as the input to models.
        """
        assert (
            "input_col" in example and "output_col" in example
        ), "Example dictionary must have 'input_col' and 'output_col' keys"
        if has_encoder:
            model_input = (
                f"<task {task_id}> {instruction} Example: {example['input_col']}"
            )
        else:
            model_input = (
                f"<task {task_id}> {instruction} Example: {example['input_col']}"
                + f" Label: {example['output_col']}"
            )
        example["model_input"] = model_input
        return example
