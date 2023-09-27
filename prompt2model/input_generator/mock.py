"""A class for generating mock inputs (for testing purposes)."""

from prompt2model.input_generator.base import InputGenerator
from prompt2model.prompt_parser import PromptSpec


class MockInputGenerator(InputGenerator):
    """A class for generating empty datasets (for testing purposes)."""

    def generate_inputs(self, num_examples: int, prompt_spec: PromptSpec) -> list[str]:
        """Generate mock inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
        _ = prompt_spec
        return ["The mock input"] * num_examples
