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

<<<<<<< HEAD:prompt2model/prompt_parser.py
=======
    @abstractmethod
    def parse_prompt(self, prompt: str) -> None:
        """Parse the prompt and store the structured information."""


class AllGenerationSpec:
    """Parse and store structured information about the prompt."""

>>>>>>> 57fdb40f84636aea3428db9df6b2fa2cebd57dcc:prompt2model/prompt_parser/base.py
    def __init__(self):
        """By default, assume that every task is a text generation task."""
        self.task_type: TaskType = TaskType.TEXT_GENERATION

    def parse_prompt(self, prompt: str) -> None:
        """Assume that every task is text generation."""
        _ = prompt
        self.task_type = TaskType.TEXT_GENERATION
