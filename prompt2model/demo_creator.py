"""An interface for creating Gradio demos automatically.
"""

import transformers
import gradio as gr

from prompt_parser import PromptSpec


def create_demo(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
