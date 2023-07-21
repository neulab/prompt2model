"""Tools for creating a GPT-2 model with padding tokenizer."""

from collections import namedtuple

from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

ModelAndTokenizer = namedtuple("ModelAndTokenizer", ["model", "tokenizer"])


def create_gpt2_model_and_tokenizer(full_size: bool = False) -> ModelAndTokenizer:
    """Create a GPT2 model with its padding tokenizer for batched input.

    Args:
        full_size: Whether to use the full size of the GPT-2 model. Defaults to False.
            Note that the full size of the GPT-2 model may occupy too much memory
            that lead to out of memory errors.

    Returns:
        gpt2_model_and_tokenizer: A namedtuple with gpt2 model and tokenizer.
    """
    if not full_size:
        gpt2_model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    else:
        gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    if gpt2_model.config.pad_token_id is None:
        gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id
    gpt2_model_and_tokenizer = ModelAndTokenizer(gpt2_model, gpt2_tokenizer)
    return gpt2_model_and_tokenizer


def create_t5_model_and_tokenizer(full_size: bool = False) -> ModelAndTokenizer:
    """Create a T5 model with its padding tokenizer for batched input.

    Args:
        full_size: Whether to use the full size of the T5 model. Defaults to False.
            Note that the full size of the T5 model may occupy too much memory
            that lead to out of memory errors.

    Returns:
        t5_model_and_tokenizer: A namedtuple with t5 model and tokenizer.
    """
    if not full_size:
        t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/t5-efficient-tiny"
        )
        t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-tiny")
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model_and_tokenizer = ModelAndTokenizer(t5_model, t5_tokenizer)
    return t5_model_and_tokenizer
