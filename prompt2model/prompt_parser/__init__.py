"""Import PromptSpec classes."""
from prompt2model.prompt_parser.base import DefaultSpec  # noqa: F401
from prompt2model.prompt_parser.base import PromptSpec, TaskType

__all__ = ("DefaultSpec", "PromptSpec", "TaskType")
