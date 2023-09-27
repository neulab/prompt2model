"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

from prompt2model.prompt_parser import PromptSpec


class OutputAnnotator(ABC):
    """A class for annotating outputs for each given input."""

    @abstractmethod
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
