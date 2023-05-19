"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from enum import Enum


class TaskType(Enum):
    """High-level taxonomy of possible NLP model outputs."""

    TEXT_GENERATION = 1
    CLASSIFICATION = 2
    SEQUENCE_TAGGING = 3
    SPAN_EXTRACTION = 4


class PromptSpec(ABC):
    """Parse and store structured information about the prompt."""

    task_type: TaskType
    instruction: str | None
    examples: str | None
    prompt_template: str | None

    @abstractmethod
    def parse_from_prompt(self, prompt: str) -> None:
        """Populate this class by parsing a prompt."""

    @property
    def get_instruction(self) -> str:
        """Return the natural language instruction parsed from the prompt."""
        assert self.instruction is not None
        return self.instruction

    @property
    def get_examples(self) -> str | None:
        """Return the natural language examples parsed from the prompt."""
        return self.examples or ""
