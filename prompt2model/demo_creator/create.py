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
    description = prompt_parser.get_instruction()
    article = prompt_parser.examples

    def response(message):
        prompt_parser.parse_from_prompt(message)
        response = model_executor.make_single_prediction(message)
        prediction = response.prediction
        confidence = response.confidence
        model_output = f"{prediction} \n (with {confidence} confidence)"
        return model_output

    def chat(message, history):
        history = history or []
        model_output = response(message)
        history.append((message, model_output))
        return history, history

    iface = gr.Interface(
        chat,
        ["text", "state"],
        ["chatbot", "state"],
        description=description,
        article=article,
        allow_screenshot=False,
        allow_flagging="never",
    )

    return iface
