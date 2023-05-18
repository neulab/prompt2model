"""Testing demo creator with different configurations."""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
import gradio as gr
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import MockPromptSpec
from prompt2model.demo_creator import create_gradio
from prompt2model.prompt_parser import TaskType

def test_create_gradio_with_gpt2():
    # Create GPT-2 model and tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = "[PAD]"
        gpt2_model.config.pad_token_id = len(gpt2_tokenizer)
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        gpt2_model.config.attention_mask_fn = lambda input_ids: (
            input_ids != gpt2_model.config.pad_token_id
        ).float()

    # Create GenerationModelExecutor
    gpt2_executor = GenerationModelExecutor(
        model=gpt2_model,
        tokenizer=gpt2_tokenizer,
        test_set=None,
        input_column="model_input",
        batch_size=1,
    )

    # Create OpenAIInstructionParser
    gpt2_prompt_parser = MockPromptSpec(TaskType.TEXT_GENERATION)

    # Create Gradio interface
    interface_gpt2 = create_gradio(gpt2_executor, gpt2_prompt_parser)

    # Perform assertions
    assert isinstance(interface_gpt2, gr.Interface)
    assert len(interface_gpt2.inputs) == 2
    assert interface_gpt2.inputs[0].type == "text"
    assert len(interface_gpt2.outputs) == 2
    assert interface_gpt2.outputs[0].type == "text[]"


def test_create_gradio_with_t5():
    # Create T5 model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Create GenerationModelExecutor
    t5_executor = GenerationModelExecutor(
        model=t5_model,
        tokenizer=t5_tokenizer,
        test_set=None,
        input_column="model_input",
        batch_size=1,
    )

    # Create OpenAIInstructionParser
    t5_prompt_parser = OpenAIInstructionParser(task_type="generation", api_key=None)

    # Create Gradio interface
    interface_t5 = create_gradio(t5_executor, t5_prompt_parser)

    # Perform assertions
    assert isinstance(interface_t5, gr.Interface)
    assert len(interface_t5.inputs) == 2
    assert interface_t5.inputs[0].type == "text"
    assert len(interface_t5.outputs) == 2
    assert interface_t5.outputs[0].type == "text[]"
