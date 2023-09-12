"""An commend line demo to run the whole system."""

import json
import logging
import os
import time

import pyfiglet
import yaml
from termcolor import colored

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.prompt_based import PromptBasedDatasetGenerator
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.prompt_parser import (
    MockPromptSpec,
    PromptBasedInstructionParser,
    TaskType,
)
from prompt2model.utils.logging_utils import get_formatted_logger


def line_print(input_str: str) -> None:
    """Print the given input string surrounded by horizontal lines.

    Args:
        input_str: The string to be printed.
    """
    print(f"{input_str}")


def print_logo():
    """Print the logo of Prompt2Model."""
    figlet = pyfiglet.Figlet(width=200)
    # Create ASCII art for each word and split into lines
    words = ["Prompt", "2", "Model"]
    colors = ["red", "green", "blue"]
    ascii_art_parts = [figlet.renderText(word).split("\n") for word in words]

    # Calculate the maximum height among the words
    max_height = max(len(part) for part in ascii_art_parts)

    # Equalize the height by adding empty lines at the bottom
    for part in ascii_art_parts:
        while len(part) < max_height:
            part.append("")

    # Zip the lines together, color them, and join them with a space
    ascii_art_lines = []
    for lines in zip(*ascii_art_parts):
        colored_line = " ".join(
            colored(line, color) for line, color in zip(lines, colors)
        )
        ascii_art_lines.append(colored_line)

    # Join the lines together to get the ASCII art
    ascii_art = "\n".join(ascii_art_lines)

    # Get the width of the terminal
    term_width = os.get_terminal_size().columns

    # Center the ASCII art
    centered_ascii_art = "\n".join(
        line.center(term_width) for line in ascii_art.split("\n")
    )

    line_print(centered_ascii_art)


def main():
    """The main function running the whole system."""
    print_logo()
    # Save the status of Prompt2Model for this session,
    # in case the user wishes to stop and continue later.
    if os.path.isfile("status.yaml"):
        with open("status.yaml", "r") as f:
            status = yaml.safe_load(f)
    else:
        status = {}

    while True:
        line_print("Do you want to start from scratch? (y/n)")
        answer = input()
        if answer.lower() == "n":
            if os.path.isfile("status.yaml"):
                with open("status.yaml", "r") as f:
                    status = yaml.safe_load(f)
                    print(f"Current status:\n{json.dumps(status, indent=4)}")
                    break
            else:
                status = {}
                break
        elif answer.lower() == "y":
            status = {}
            break
        else:
            continue

    propmt_has_been_parsed = status.get("prompt_has_been_parsed", False)
    dataset_has_been_retrieved = status.get("dataset_has_been_retrieved", False)
    dataset_has_been_generated = status.get("dataset_has_been_generated", False)
    if not propmt_has_been_parsed:
        prompt = ""
        line_print(
            "Enter your task description and few-shot examples (or 'done' to finish):"
        )
        time.sleep(2)
        while True:
            line = input()
            if line == "done":
                break
            prompt += line + "\n"
        line_print("Parsing prompt...")
        prompt_spec = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
        prompt_spec.parse_from_prompt(prompt)

        propmt_has_been_parsed = True
        status["instruction"] = prompt_spec.instruction
        status["examples"] = prompt_spec.examples
        status["prompt_has_been_parsed"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("Prompt parsed.")

    if propmt_has_been_parsed and not dataset_has_been_retrieved:
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        line_print("Retrieving dataset...")
        retriever = DescriptionDatasetRetriever()
        retrieved_dataset_dict = retriever.retrieve_dataset_dict(prompt_spec)
        dataset_has_been_retrieved = True
        if retrieved_dataset_dict is not None:
            retrieved_dataset_dict.save_to_disk("retrieved_dataset_dict")
            status["retrieved_dataset_dict_root"] = "retrieved_dataset_dict"
        else:
            status["retrieved_dataset_dict_root"] = None
        status["dataset_has_been_retrieved"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)

    # skipping the model retrieval

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and not dataset_has_been_generated
    ):
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        generator_logger = get_formatted_logger("DatasetGenerator")
        generator_logger.setLevel(logging.INFO)
        line_print("The dataset generation has not finished.")
        time.sleep(2)
        line_print(f"Your input instruction:\n\n{prompt_spec.instruction}")
        time.sleep(2)
        line_print(f"Your input few-shot examples:\n\n{prompt_spec.examples}")
        time.sleep(2)
        while True:
            line_print("Enter the number of examples you wish to generate:")
            line = 100
            try:
                num_expected = int(line)
                break
            except ValueError:
                line_print("Invalid input. Please enter a number.")
        while True:
            line_print("Enter the initial temperature:")
            line = input()
            try:
                initial_temperature = float(line)
                assert 0 <= initial_temperature <= 2.0
                break
            except Exception:
                line_print(
                    "Invalid initial temperature. Please enter a number (float) between 0 and 2."  # noqa E501
                )
        while True:
            line_print("Enter the max temperature (we suggest 1.4):")
            line = input()
            try:
                max_temperature = float(line)
                assert 0 <= max_temperature <= 2.0
                break
            except Exception:
                line_print(
                    "Invalid max temperature. Please enter a float between 0 and 2."
                )
        line_print("Starting to generate dataset. This may take a while...")
        time.sleep(2)
        unlimited_dataset_generator = PromptBasedDatasetGenerator(
            initial_temperature=initial_temperature,
            max_temperature=max_temperature,
            responses_per_request=3,
        )
        generated_dataset = unlimited_dataset_generator.generate_dataset_split(
            prompt_spec,
            num_expected,
            split=DatasetSplit.TRAIN,
            retrieved_data=retrieved_dataset_dict["train"],
            few_shot_method="dataset_swapout",
            dataset_max_instances=5,
        )
        generated_dataset.save_to_disk("generated_dataset_swapout5")
        dataset_has_been_generated = True
        status["dataset_has_been_generated"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("The generated dataset is ready.")
        time.sleep(2)


if __name__ == "__main__":
    main()
