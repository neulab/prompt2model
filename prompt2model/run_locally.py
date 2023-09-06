"""A script to run the prompt2model pipeline locally."""
from __future__ import annotations

import argparse

from prompt2model.dataset_generator import DatasetSplit, MockDatasetGenerator
from prompt2model.dataset_processor import MockProcessor
from prompt2model.dataset_retriever import MockRetriever
from prompt2model.demo_creator import mock_gradio_create
from prompt2model.model_evaluator import MockEvaluator
from prompt2model.model_executor import MockModelExecutor
from prompt2model.model_retriever import MockModelRetriever
from prompt2model.model_trainer import MockTrainer
from prompt2model.param_selector import MockParamSelector
from prompt2model.prompt_parser import MockPromptSpec, PromptSpec, TaskType

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    nargs="+",
    required=True,
    help="Prompt (with optional few-shot examples) for language model",
)
parser.add_argument(
    "--metrics-output-path",
    type=str,
    help="Path to JSON file where we store model metrics",
    default="/tmp/metrics.json",
)


def process_input_prompt(prompt_tokens: list[str]) -> PromptSpec:
    """Preprocess the input prompt given by the user and parse.

    Args:
        prompt_tokens: Tokens in the prompt.

    Returns:
        A PromptSpec parsed from the processed prompt tokens.

    """
    prompt_str = " ".join(prompt_tokens).strip()
    start_quotations_present = False
    end_quotations_present = False
    quotation_marks = ['"', "“", "‟", "”"]
    for start_quote in quotation_marks:
        if prompt_str.startswith(start_quote):
            start_quotations_present = True
            break
    for end_quote in quotation_marks:
        if prompt_str.endswith(end_quote):
            end_quotations_present = True
            break
    if start_quotations_present and end_quotations_present:
        prompt_str = prompt_str[1:-1]

    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt_str)
    return prompt_spec


def run_skeleton(prompt_tokens: list[str], metrics_output_path: str) -> None:
    """Run the prompt2model pipeline locally using base/stub components."""
    prompt_spec = process_input_prompt(prompt_tokens)

    # Retrieve and generate datasets
    retriever = MockRetriever()
    retrieved_dataset_dicts = retriever.retrieve_dataset_dict(prompt_spec)

    generator = MockDatasetGenerator()
    expected_num_examples = {
        DatasetSplit.TRAIN: 40,
        DatasetSplit.VAL: 5,
        DatasetSplit.TEST: 5,
    }
    generated_dataset_dicts = generator.generate_dataset_dict(
        prompt_spec, expected_num_examples
    )

    processor = MockProcessor(has_encoder=True, eos_token="")
    retrieved_dataset_dicts, generated_dataset_dicts = processor.process_dataset_dict(
        instruction="", dataset_dicts=[retrieved_dataset_dicts, generated_dataset_dicts]
    )

    retrieved_training = [
        dataset_dict["train"] for dataset_dict in retrieved_dataset_dicts
    ]

    generated_training = generated_dataset_dicts[DatasetSplit.TRAIN.value]
    validation = generated_dataset_dicts[DatasetSplit.VAL.value]
    testing = generated_dataset_dicts[DatasetSplit.TEST.value]
    all_training = retrieved_training + [generated_training]

    model_retriever = MockModelRetriever("cardiffnlp/twitter-roberta-base-sentiment")
    retrieved_model_name = model_retriever.retrieve(prompt_spec)

    trainer = MockTrainer(retrieved_model_name[0])
    selector = MockParamSelector(trainer)
    model, tokenizer = selector.select_from_hyperparameters(
        all_training, validation, {}
    )

    model_executor = MockModelExecutor(model, tokenizer)
    predictions = model_executor.make_prediction(testing, "input_col")

    evaluator = MockEvaluator()
    metrics_dict = evaluator.evaluate_model(
        testing, "output_col", predictions, "input_col", []
    )
    evaluator.write_metrics(metrics_dict, metrics_output_path)
    mock_gradio_create(model_executor, prompt_spec)


if __name__ == "__main__":
    args = parser.parse_args()
    run_skeleton(args.prompt, args.metrics_output_path)
