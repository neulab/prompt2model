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
    article = prompt_parser.demonstration

    def chat(message):
        prompt_parser.parse_from_prompt(message)
        response = model_executor.make_single_prediction(message)
        prediction = response.prediction
        confidence = response.confidence
        model_output = f"{prediction} \n (with {confidence} confidence)"
        return model_output

    textbox = gr.Textbox(label="Input", placeholder="John Doe", lines=2)
    gr_interface = gr.Interface(
        fn=chat,
        inputs=textbox,
        description=description,
        article=article,
    )

    return gr_interface
