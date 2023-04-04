import transformers
import gradio as gr

# Input: A huggingface models
# Output: A Gradio object


def create_demo(model: transformers.PreTrainedModel) -> gr.Interface:
    return gr.Interface()
