"""Import DemoCreator functions."""

from prompt2model.demo_creator.create import create_gradio
from prompt2model.demo_creator.mock import mock_gradio_create

__all__ = (
    "mock_gradio_create",
    "create_gradio",
)
