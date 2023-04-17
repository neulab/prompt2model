"""An interface for prompt parsing."""

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


class DefaultSpec(PromptSpec):
    """Use explicitly-set default settings."""

    def __init__(self, task_type: TaskType):
        """By default, assume that every task is a text generation task."""
        self.task_type = task_type

    def parse_from_prompt(self, prompt: str) -> None:
        """Don't parse anything."""
