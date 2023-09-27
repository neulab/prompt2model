"""A class for generating mock inputs (for testing purposes)."""

from prompt2model.output_annotator.base import OutputAnnotator
from prompt2model.prompt_parser import PromptSpec


class MockOutputAnnotator(OutputAnnotator):
    """A class for annotating outputs for each given input (for testing purposes)."""

    def annotate_outputs(
        self,
        input_strings: list[str],
        num_candidate_outputs: int,
        prompt_spec: PromptSpec,
    ) -> dict[str, list[str]]:
        """Generate candidate outputs for each given input.

        Args:
            input_strings: A list of input strings from InputGenerator.
            num_candidate_outputs: Number of candidate outputs for
                each input in input_strings.
            prompt_spec: A parsed prompt spec.

        Returns:
            A dictionary mapping input strings to a list of candidate outputs.
        """
        _ = prompt_spec
        return {
            input: ["This is a mock output."] * num_candidate_outputs
            for input in input_strings
        }
