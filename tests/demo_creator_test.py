"""Test the create_gradio function with two configurations."""

import gc

import gradio as gr

from prompt2model.demo_creator import create_gradio
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from test_helpers import create_gpt2_model_and_tokenizer, create_t5_model_and_tokenizer


def test_create_gradio_with_gpt2():
    """Test the `create_gradio` method with a GPT2 model."""
    # Create a GPT-2 model and tokenizer.
    gpt2_model_and_tokenizer = create_gpt2_model_and_tokenizer()
    gpt2_model = gpt2_model_and_tokenizer.model
    gpt2_tokenizer = gpt2_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    gpt2_executor = GenerationModelExecutor(
        model=gpt2_model,
        tokenizer=gpt2_tokenizer,
        batch_size=1,
    )

    # Create PromptBasedInstructionParser.
    gpt2_prompt_parser = MockPromptSpec(TaskType.TEXT_GENERATION)

    # Create Gradio interface.
    interface_gpt2 = create_gradio(gpt2_executor, gpt2_prompt_parser)

    # Perform assertions.
    assert isinstance(interface_gpt2, gr.Blocks)
    gc.collect()


def test_create_gradio_with_t5():
    """Test the `create_gradio` method with a T5 model."""
    # Create T5 model and tokenizer
    t5_model_and_tokenizer = create_t5_model_and_tokenizer()
    t5_model = t5_model_and_tokenizer.model
    t5_tokenizer = t5_model_and_tokenizer.tokenizer

    # Create a GenerationModelExecutor.
    t5_executor = GenerationModelExecutor(
        model=t5_model,
        tokenizer=t5_tokenizer,
        batch_size=1,
    )

    # Create PromptBasedInstructionParser.
    t5_prompt_parser = MockPromptSpec(task_type="generation")

    # Create Gradio interface.
    interface_t5 = create_gradio(t5_executor, t5_prompt_parser)

    # Perform assertions.
    assert isinstance(interface_t5, gr.Blocks)
    gc.collect()
