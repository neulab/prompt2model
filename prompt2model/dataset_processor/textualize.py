"""A dataset processor to convert datasets into Text2Text fashion."""

from prompt2model.dataset_processor.base import BaseProcessor


class TextualizeProcessor(BaseProcessor):
    """A class for post-processing datasets, convert them into Text2Text fashion."""

    @staticmethod
    def post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this example
                comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.

        Returns:
            A dictionary with `model_input` as the input to models.
        """
        assert (
            "input_col" in example and "output_col" in example
        ), "Example dictionary must have 'input_col' and 'output_col' keys"
        assert dataset_split in (
            "train",
            "val",
            "test",
        ), "Split must be one of train/val/test"
        if has_encoder:
            model_input = (
                f"<task {task_id}> {instruction} Example: {example['input_col']}"
            )
        else:
            if dataset_split == "train":
                model_input = (
                    f"<task {task_id}> {instruction} Example: {example['input_col']}"
                    + f" Label: {example['output_col']}"
                )
            else:
                model_input = (
                    f"<task {task_id}> {instruction} Example: {example['input_col']}"
                    + " Label: "
                )
        example["model_input"] = model_input
        return example
