import transformers
import gradio as gr

from prompt_parser import PromptSpec

# Input: A huggingface models
# Output: A Gradio object


def create_demo(
    model: transformers.PreTrainedModel, prompt_spec: PromptSpec
) -> gr.Interface:
    return gr.Interface(lambda input: None, "textbox", "label")
