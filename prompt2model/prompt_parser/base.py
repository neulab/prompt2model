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


# pylint: disable=too-few-public-methods
class PromptSpec(ABC):
    """Parse and store structured information about the prompt."""

    @abstractmethod
    def parse_from_prompt(self, prompt: str) -> None:
        """Populate this class by parsing a prompt."""

    @abstractmethod
    def get_instruction(self) -> str:
        """Return the natural language instruction givenb y the prompt."""
