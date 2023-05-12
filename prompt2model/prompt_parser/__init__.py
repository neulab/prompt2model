"""Import PromptSpec classes."""
from prompt2model.prompt_parser.base import DefaultSpec  # noqa: F401
from prompt2model.prompt_parser.base import PromptSpec, TaskType
from prompt2model.prompt_parser.instr_parser import OpenAIInstructionParser
from prompt2model.prompt_parser.mock import MockPromptSpec

__all__ = (
    "DefaultSpec",
    "PromptSpec",
    "TaskType",
    "MockPromptSpec",
    "OpenAIInstructionParser",
)
