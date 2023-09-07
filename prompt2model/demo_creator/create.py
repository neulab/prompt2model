"""Create a Gradio interface automatically."""

import gradio as gr
import mdtex2html

from prompt2model.dataset_processor import TextualizeProcessor
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import PromptBasedInstructionParser


def create_gradio(
    model_executor: GenerationModelExecutor, prompt_parser: PromptBasedInstructionParser
) -> gr.Blocks:
    """Create a Gradio interface automatically.

    Args:
        model_executor: A GenerationModelExecutor to expose via a Gradio interface.
        prompt_parser: An instance of PromptBasedInstructionParser to parse the prompt.

    Returns:
        A Gradio interface for interacting with the model.

    """
    description = prompt_parser.instruction
    examples = prompt_parser.examples

    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert((message)),
                None if response is None else mdtex2html.convert(response),
            )
        return y

    gr.Chatbot.postprocess = postprocess

    def response(message: str):
        if not message.startswith("<task 0>"):
            dataset_processor = TextualizeProcessor(has_encoder=True)
            message = dataset_processor.wrap_single_input(
                prompt_parser.instruction, message
            )
        response = model_executor.make_single_prediction(message)
        prediction = response.prediction
        return prediction

    def chat(message, history):
        history = history or []
        model_output = (
            response(message) if message != "" else "Please give valid input."
        )
        history.append((message, model_output))
        return history, history

    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        return [], []

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Prompt2Model</h1>""")
        gr.HTML(f"""<h2 align="center">Task Description: {description}</h2>""")
        gr.HTML(f"""<h2 align="center">Few-shot Examples: {examples}</h2>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Input...", lines=10
                    ).style(container=False)
                with gr.Column(min_width=32, scale=2):
                    submitBtn = gr.Button("Submit", variant="primary")
                    emptyBtn = gr.Button("Clear History")

        history = gr.State([])

        submitBtn.click(
            chat, [user_input, history], [chatbot, history], show_progress=True
        )
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    return demo
