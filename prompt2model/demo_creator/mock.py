"""An interface for creating Gradio demos automatically."""

import gradio as gr
import transformers

from prompt2model.prompt_parser.base import PromptSpec


def mock_gradio_create(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    """Create a Gradio interface automatically.

    Args:
        model: A trained model to expose via a Gradio interface.
        prompt_spec: A PromptSpec to help choose the visual interface.

    Returns:
        A Gradio interface for interacting with the model.

    """
    _ = model, prompt_spec  # suppress unused variable warnings
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
