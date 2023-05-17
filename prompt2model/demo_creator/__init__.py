"""Import DemoCreator functions."""
from prompt2model.demo_creator.mock import mock_gradio_create
from prompt2model.demo_creator.create import create_gradio

__all__ = (
    "mock_gradio_create",
    "create_gradio",
)
