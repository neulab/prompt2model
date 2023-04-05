"""An interface for prompt parsing.
"""

from enum import Enum
from typing import Dict, List

# Input:
#    prompt string
#
# Output:
#    PromptSpec


class TaskType(Enum):
    TEXT_GENERATION = 1
    CLASSIFICATION = 2
    SEQUENCE_TAGGING = 3
    SPAN_EXTRACTION = 4


class PromptSpec:
    def __init__(self):
        self.task_type: TaskType = TaskType.TEXT_GENERATION

    def parse_prompt(self, prompt: str) -> None:
        # raise NotImplementedError
        self.task_type = TaskType.TEXT_GENERATION
