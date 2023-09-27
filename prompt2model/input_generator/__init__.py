"""Import Input Generator classes."""
from prompt2model.input_generator.base import InputGenerator
from prompt2model.input_generator.mock import MockInputGenerator

__all__ = (
    "InputGenerator",
    "MockInputGenerator",
)
