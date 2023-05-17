"""Testing GenerationModelExecutor with different configurations."""

from datasets import Dataset
from transformers import T5ForConditionalGeneration

from prompt2model.model_executor import GenerationModelExecutor, ModelOutput

from transformers import AutoModelForCausalLM, T5Tokenizer  # isort:skip
from transformers import AutoTokenizer  # isort:skip


def test_t5_model():
    """Test the `GenerationModelExecutor` class with a T5 model.

    This function performs the following steps:
    1. Create a T5 model and tokenizer.
    2. Create a test dataset.
    3. Create a GenerationModelExecutor instance.
    4. Call `make_predictions()` with the T5 model and tokenizer.
    5. Perform assertions on the model outputs.
    """
    # Create T5 model and tokenizer
    t5_model_name = "t5-small"
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    # Create test dataset
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "This is the first test input.",
                "Another example for testing.",
                "One more test input.",
            ]
        }
    )

    # Create GenerationModelExecutor
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, test_dataset, "model_input"
    )

    # Test T5 model
    t5_outputs = model_executor.make_predictions()
    assert isinstance(t5_outputs, list)
    assert len(t5_outputs) == len(test_dataset)

    for output in t5_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert output.confidence is not None
        assert list(output.auxiliary_info.keys()) == [
            "example",
            "input_text",
            "logits",
            "probs",
        ]
        assert isinstance(output.auxiliary_info, dict)


def test_gpt2_model():
    """Test the `GenerationModelExecutor` class with a GPT-2 model.

    This function performs the following steps:
    1. Create a GPT-2 model and tokenizer.
    2. Create a test dataset.
    3. Create a GenerationModelExecutor instance.
    4. Call `make_predictions()` with the GPT-2 model and tokenizer.
    5. Perform assertions on the model outputs.
    """
    # Create GPT-2 model and tokenizer
    gpt2_model_name = "gpt2"
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = "[PAD]"
        gpt2_model.config.pad_token_id = len(gpt2_tokenizer)
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        gpt2_model.config.attention_mask_fn = lambda input_ids: (
            input_ids != gpt2_model.config.pad_token_id
        ).float()
    # Create test dataset
    test_dataset = Dataset.from_dict(
        {
            "model_input": [
                "This is the first test input.",
                "Another example for testing.",
                "One more test input.",
            ]
        }
    )

    # Create GenerationModelExecutor
    model_executor = GenerationModelExecutor(
        gpt2_model, gpt2_tokenizer, test_dataset, "model_input"
    )

    # Test GPT-2 model
    gpt2_outputs = model_executor.make_predictions()
    assert isinstance(gpt2_outputs, list)
    assert len(gpt2_outputs) == len(test_dataset)

    for output in gpt2_outputs:
        assert isinstance(output, ModelOutput)
        assert output.prediction is not None
        assert output.confidence is not None
        assert list(output.auxiliary_info.keys()) == [
            "example",
            "input_text",
            "logits",
            "probs",
        ]
        assert isinstance(output.auxiliary_info, dict)
