"""An interface for prompt parsing.
"""

from abc import ABC, abstractmethod
from enum import Enum


class TaskType(Enum):
    """High-level taxonomy of possible NLP model outputs"""

    TEXT_GENERATION = 1
    CLASSIFICATION = 2
    SEQUENCE_TAGGING = 3
    SPAN_EXTRACTION = 4


class PromptSpec(ABC):
    """Parse and store structured information about the prompt."""

    @abstractmethod
    def parse_prompt(self, prompt: str) -> None:
        """Parse the prompt and store the structured information."""


class AllGenerationSpec:
    """Parse and store structured information about the prompt."""

    def __init__(self):
        """
        Initialize with default prompt values. For example, we assume by
        default that every task is a text generation task.
        """
        self.task_type: TaskType = TaskType.TEXT_GENERATION

    def parse_prompt(self, prompt: str) -> None:
        """Assume that every task is text generation."""
        _ = prompt
        self.task_type = TaskType.TEXT_GENERATION
