"""An interface for creating Gradio demos automatically."""

import gradio as gr
import transformers
from prompt_parser.base import PromptSpec


def create_gradio(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    """Create a Gradio interface automatically.

    Args:
        model: A trained model to expose via a Gradio interface.
        prompt_spec: A PromptSpec to help choose the visual interface.

    Returns:
        A Gradio interface for interacting with the model.

    """
<<<<<<< HEAD:prompt2model/demo_creator.py
=======
    _ = model, prompt_spec  # suppress unused variable warnings
>>>>>>> 57fdb40f84636aea3428db9df6b2fa2cebd57dcc:prompt2model/demo_creator/gradio_creator.py
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
