"""An interface for creating Gradio demos automatically."""

import gradio as gr

from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import OpenAIInstructionParser


def gradio_create(
    model_executor: GenerationModelExecutor, prompt_parser: OpenAIInstructionParser
) -> gr.Interface:
    """Create a Gradio interface automatically.

    Args:
        model_executor: A GenerationModelExecutor to expose via a Gradio interface.
        prompt_parser: An instance of OpenAIInstructionParser to parse the prompt.

    Returns:
        A Gradio interface for interacting with the model.

    """

    def predict(input_text):
        prompt_parser.parse_from_prompt(input_text)
        instruction = prompt_parser.get_instruction()

        # Generate model outputs based on the instruction
        model_executor.set_instruction(instruction)
        model_outputs = model_executor.make_predictions()

        # Extract predictions from model outputs
        predictions = [output.prediction for output in model_outputs]
        return predictions

    # Define the Gradio interface input
    interface_input = gr.inputs.Textbox(label="User Prompt")

    # Define the Gradio interface output
    interface_output = gr.outputs.Textbox(label="Model Predictions")

    # Create the Gradio interface with the predict function
    # and the defined inputs/outputs
    gr_interface = gr.Interface(
        fn=predict, inputs=interface_input, outputs=interface_output
    )

    return gr_interface
