"""A script to run the prompt2model pipeline locally."""

import argparse

from prompt2model.dataset_generator import DatasetSplit, MockDatasetGenerator
from prompt2model.dataset_retriever import MockRetriever
from prompt2model.demo_creator.gradio_creator import create_gradio
from prompt2model.evaluator import MockEvaluator
from prompt2model.model_executor import MockModelExecutor
from prompt2model.param_selector import MockParamSelector
from prompt2model.prompt_parser import DefaultSpec, PromptSpec, TaskType
from prompt2model.trainer import MockTrainer

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

    prompt_spec = DefaultSpec(TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt_str)
    return prompt_spec


def run_skeleton(prompt_tokens: list[str], metrics_output_path: str) -> None:
    """Run the prompt2model pipeline locally using base/stub components."""
    prompt_spec = process_input_prompt(prompt_tokens)

    # Retrieve and generate datasets
    retriever = MockRetriever()
    retrieved_training = retriever.retrieve_datasets(prompt_spec)
    generator = MockDatasetGenerator()

    num_examples = {
        DatasetSplit.TRAIN: 40,
        DatasetSplit.VAL: 5,
        DatasetSplit.TEST: 5,
    }
    generated_data = dict(generator.generate_datasets(prompt_spec, num_examples))
    generated_training = generated_data[DatasetSplit.TRAIN]
    validation = generated_data[DatasetSplit.VAL]
    testing = generated_data[DatasetSplit.TEST]
    all_training = retrieved_training + [generated_training]

    trainer = MockTrainer()
    selector = MockParamSelector(trainer)
    model = selector.select_from_hyperparameters(all_training, validation, {})

    model_executor = MockModelExecutor()
    predictions = model_executor.make_predictions(model, testing, "input_col")

    evaluator = MockEvaluator()
    metrics_dict = evaluator.evaluate_model(
        testing, "output_col", predictions, [], prompt_spec
    )
    evaluator.write_metrics(metrics_dict, metrics_output_path)
    create_gradio(model, prompt_spec)


if __name__ == "__main__":
    args = parser.parse_args()
    run_skeleton(args.prompt, args.metrics_output_path)