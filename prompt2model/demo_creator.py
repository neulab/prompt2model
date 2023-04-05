"""An interface for creating Gradio demos automatically from a trained model
and a prompt specification.
"""

import gradio as gr
import transformers

from prompt_parser import PromptSpec


def create_demo(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    dummy_interface = gr.Interface(lambda input: None, "textbox", "label")
    return dummy_interface
