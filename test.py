"""Test the create_gradio function with two configurations."""

import gradio as gr
from transformers import T5ForConditionalGeneration

from prompt2model.demo_creator import create_gradio
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import MockPromptSpec, TaskType

from transformers import AutoModelForCausalLM, T5Tokenizer  # isort:skip
from transformers import AutoTokenizer  # isort:skip


gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
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
interface_gpt2.launch()
