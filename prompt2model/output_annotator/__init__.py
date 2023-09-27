"""Import Input Generator classes."""
from prompt2model.output_annotator.base import OutputAnnotator
from prompt2model.output_annotator.mock import MockOutputAnnotator

__all__ = (
    "OutputAnnotator",
    "MockOutputAnnotator",
)
