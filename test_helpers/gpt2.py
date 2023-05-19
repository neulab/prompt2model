"""Tools for creating a GPT-2 model with padding tokenizer."""

from collections import namedtuple

from transformers import AutoModelForCausalLM, AutoTokenizer

Gpt2ModelAndTokenizer = namedtuple("Gpt2ModelAndTokenizer", ["model", "tokenizer"])


def create_gpt2_model_and_tokenizer() -> Gpt2ModelAndTokenizer:
    """Create a GPT2 model with its padding tokenizer for batched input.

    Returns:
        Gpt2ModelAndTokenizer: A namedtuple with model and tokenizer.
    """
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = "[PAD]"
        gpt2_model.config.pad_token_id = len(gpt2_tokenizer)
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        gpt2_model.config.attention_mask_fn = lambda input_ids: (
            input_ids != gpt2_model.config.pad_token_id
        ).float()
    return Gpt2ModelAndTokenizer(gpt2_model, gpt2_tokenizer)
