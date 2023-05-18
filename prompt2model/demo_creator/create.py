"""Create a Gradio interface automatically."""

import gradio as gr

from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import OpenAIInstructionParser


def create_gradio(
    model_executor: GenerationModelExecutor, prompt_parser: OpenAIInstructionParser
) -> gr.Interface:
    """Create a Gradio interface automatically.

    Args:
        model_executor: A GenerationModelExecutor to expose via a Gradio interface.
        prompt_parser: An instance of OpenAIInstructionParser to parse the prompt.

    Returns:
        A Gradio interface for interacting with the model.

    """

    def chat(message, history):
        history = history or []
        prompt_parser.parse_from_prompt(message)
        response = model_executor.make_single_prediction(message).prediction
        history.append((message, response))
        return history, history

    gr_interface = gr.Interface(
        chat,
        ["text", "state"],
        ["chatbot", "state"],
        allow_screenshot=False,
        allow_flagging="never",
    )

    return gr_interface
