"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

from prompt2model.prompt_parser import PromptSpec


class InputGenerator(ABC):
    """A class for generating inputs of examples from a prompt specification."""

    @abstractmethod
    def generate_inputs(self, num_examples: int, prompt_spec: PromptSpec) -> list[str]:
        """Generate new inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
