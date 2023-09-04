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
    _instruction: str | None
    _examples: str | None

    @abstractmethod
    def parse_from_prompt(self, prompt: str) -> None:
        """Populate this class by parsing a prompt."""

    @property
    def instruction(self) -> str:
        """Return the natural language instruction parsed from the prompt."""
        if self._instruction is None:
            raise ValueError("Instruction hasn't been parsed from the prompt.")
        return self._instruction

    @property
    def examples(self) -> str:
        """Return the natural language examples parsed from the prompt."""
        return self._examples or ""
