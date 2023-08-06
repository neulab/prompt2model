"""An commend line demo to run the whole system."""

import logging
import os
import time
from pathlib import Path

import datasets
import openai
import pyfiglet
import torch
import transformers
import yaml
from datasets import concatenate_datasets, load_from_disk
from termcolor import colored

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.prompt_parser import MockPromptSpec, OpenAIInstructionParser, TaskType

openai.api_key = os.environ["OPENAI_API_KEY"]


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

    print(centered_ascii_art)


def main():
    """The main function running the whole system."""
    print_logo()
    if os.path.isfile("status.yaml"):
        with open("status.yaml", "r") as f:
            status = yaml.safe_load(f)
    else:
        status = {}

    while True:
        answer = input(
            "\n-------------------------------------------------\nDo you want to start again? (y/n) \n-------------------------------------------------\n"  # noqa 501
        )
        if answer.lower() == "n":
            if os.path.isfile("status.yaml"):
                with open("status.yaml", "r") as f:
                    status = yaml.safe_load(f)
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
    model_has_been_retrieved = status.get("model_has_been_retrieved", False)
    dataset_has_been_generated = status.get("dataset_has_been_generated", False)
    model_has_been_trained = status.get("model_has_been_trained", False)
    if not propmt_has_been_parsed:
        prompt = ""
        print(
            "\n-------------------------------------------------\nEnter your task description and few-shot examples (or 'done' to finish):\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        while True:
            line = input()
            if line == "done":
                break
            prompt += line + "\n"
        print(
            f"\n-------------------------------------------------\nYour prompt is:  \n-------------------------------------------------\n{prompt} \n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        print(
            "\n-------------------------------------------------\nParsing prompt...\n-------------------------------------------------\n"  # noqa 501
        )
        prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
        prompt_spec.parse_from_prompt(prompt)

        propmt_has_been_parsed = True
        status["instruction"] = prompt_spec.instruction
        status["examples"] = prompt_spec.examples
        status["prompt_has_been_parsed"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        print(
            "\n-------------------------------------------------\nPrompt parsed.\n-------------------------------------------------\n"  # noqa 501
        )

    if propmt_has_been_parsed and not dataset_has_been_retrieved:
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        print(
            "\n-------------------------------------------------\nRetreive dataset.\n-------------------------------------------------\n"  # noqa 501
        )
        retriever = DescriptionDatasetRetriever()
        # retriever.encode_dataset_descriptions(retriever.search_index_path)
        dataset_dict = retriever.retrieve_dataset_dict(
            prompt_spec, blocklist=["squad", "stanford question answering"]
        )
        retrieved_dataset_dict = dataset_dict
        print(retrieved_dataset_dict["train"][1])
        dataset_has_been_retrieved = True
        retrieved_dataset_dict.save_to_disk("retrieved_dataset_dict")
        status["dataset_has_been_retrieved"] = True
        status["retrieved_dataset_dict_root"] = "retrieved_dataset_dict"
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and not model_has_been_retrieved
    ):
        pass
        model_has_been_retrieved = True
        status["model_has_been_retrieved"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and model_has_been_retrieved
        and not dataset_has_been_generated
    ):
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        logger = logging.getLogger("DatasetGenerator")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        print(
            "\n-------------------------------------------------\nThe dataset generation has not finished.\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        print(
            f"\n-------------------------------------------------\nYour input Instruction: \n-------------------------------------------------\n{prompt_spec.instruction}\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        print(
            f"\n-------------------------------------------------\nYour input few-shot examples: \n-------------------------------------------------\n{prompt_spec.examples}\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        while True:
            line = input(
                "\n-------------------------------------------------\nEnter the number of your expected generated examples:\n-------------------------------------------------\n"  # noqa 501
            )
            try:
                num_expected = int(line)
                break
            except ValueError:
                print(
                    "\n-------------------------------------------------\nInvalid input. Please enter a number.\n-------------------------------------------------\n"  # noqa 501
                )
        while True:
            line = input(
                "\n-------------------------------------------------\nEnter the initial temperature:\n-------------------------------------------------\n"  # noqa 501
            )
            try:
                initial_temperature = float(line)
                assert 0 <= initial_temperature <= 2.0
                break
            except Exception:
                print(
                    "\n-------------------------------------------------\nInvalid initial temperature. Please enter a float between 0 and 2.\n-------------------------------------------------\n"  # noqa 501
                )
        while True:
            line = input(
                "\n-------------------------------------------------\nEnter the max temperature:\n-------------------------------------------------\n"  # noqa 501
            )
            try:
                max_temperature = float(line)
                assert 0 <= max_temperature <= 2.0
                break
            except Exception:
                print(
                    "\n-------------------------------------------------\nInvalid max temperature. Please enter a float between 0 and 2.\n-------------------------------------------------\n"  # noqa 501
                )
        print(
            "\n-------------------------------------------------\nStart to generated dataset. This may take a while...\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        unlimited_dataset_generator = OpenAIDatasetGenerator(
            initial_temperature=initial_temperature,
            max_temperature=max_temperature,
            responses_per_request=3,
            batch_size=5,
        )
        generated_dataset = unlimited_dataset_generator.generate_dataset_split(
            prompt_spec, num_expected, split=DatasetSplit.TRAIN
        )
        generated_dataset.save_to_disk("generated_dataset")
        dataset_has_been_generated = True
        status["dataset_has_been_generated"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        print(
            "\n-------------------------------------------------\nThe generated dataset is ready.\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and model_has_been_retrieved
        and dataset_has_been_generated
        and not model_has_been_trained
    ):
        print(
            "\n-------------------------------------------------\nThe model has not been trained.\n-------------------------------------------------\n"  # noqa 501
        )
        time.sleep(2)
        dataset_root = Path("generated_dataset")
        assert dataset_root.exists()
        trained_model_root = Path("result/trained_model")
        trained_tokenizer_root = Path("result/trained_tokenizer")
        RESULT_PATH = Path("result/result")
        trained_model_root.mkdir(parents=True, exist_ok=True)
        trained_tokenizer_root.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.mkdir(parents=True, exist_ok=True)
        dataset = load_from_disk(dataset_root)
        num_examples = len(dataset)
        num_train = min(int(num_examples * 0.6), 3000)
        num_valid = min(int(num_examples * 0.2), 3000)
        train_dataset = datasets.Dataset.from_dict(dataset[:num_train])
        val_dataset = datasets.Dataset.from_dict(
            dataset[num_train : num_train + num_valid]
        )
        test_dataset = datasets.Dataset.from_dict(dataset[num_train + num_valid :])
        dataset_dict = datasets.DatasetDict(
            {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        )
        cached_retrieved_dataset_dict = datasets.load_from_disk(
            status["retrieved_dataset_dict_root"]
        )
        validation_key = (
            "validation" if "validation" in cached_retrieved_dataset_dict else "val"
        )
        retrieved_dataset_dict = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    cached_retrieved_dataset_dict["train"][:3000]
                ),
                "val": datasets.Dataset.from_dict(
                    cached_retrieved_dataset_dict[validation_key][:1000]
                ),
                "test": datasets.Dataset.from_dict(
                    cached_retrieved_dataset_dict["test"][:1000]
                ),
            }
        )
        print("Processing datasets.")
        DATASET_DICTS = [dataset_dict, retrieved_dataset_dict]
        instruction = status["instruction"]
        logger = logging.getLogger("DatasetProcessor")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        t5_processor = TextualizeProcessor(has_encoder=True)
        t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
            instruction, DATASET_DICTS
        )
        logger = logging.getLogger("DatasetProcessor")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        training_datasets = []
        validation_datasets = []
        test_datasets = []
        for idx, modified_dataaset_dict in enumerate(t5_modified_dataset_dicts):
            modified_dataaset_dict.save_to_disk(f"./preprocessed/dataset_{idx}")
            training_datasets.append(modified_dataaset_dict["train"])
            validation_datasets.append(modified_dataaset_dict["val"])
            test_datasets.append(modified_dataaset_dict["test"])
        trainer_logger = logging.getLogger("ModelTrainer")
        trainer_logger.setLevel(logging.INFO)
        for handler in trainer_logger.handlers[:]:
            trainer_logger.removeHandler(handler)
        trainer_handler = logging.StreamHandler()
        trainer_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        trainer_handler.setFormatter(trainer_formatter)
        trainer_logger.addHandler(trainer_handler)

        # Set up logger for "ModelEvaluator"
        evaluator_logger = logging.getLogger("ModelEvaluator")
        evaluator_logger.setLevel(logging.INFO)
        for handler in evaluator_logger.handlers[:]:
            evaluator_logger.removeHandler(handler)
        evaluator_handler = logging.StreamHandler()
        evaluator_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        evaluator_handler.setFormatter(evaluator_formatter)
        evaluator_logger.addHandler(evaluator_handler)

        while True:
            line = input(
                "\n-------------------------------------------------\nEnter the training batch size:\n-------------------------------------------------\n"  # noqa 501
            )
            try:
                train_batch_size = int(line)
                assert 0 < train_batch_size
                break
            except Exception:
                print(
                    "\n-------------------------------------------------\nThe training batch size must be greater than 0.\n-------------------------------------------------\n"  # noqa 501
                )
        time.sleep(1)

        while True:
            line = input(
                "\n-------------------------------------------------\nEnter the number of your training epochs:\n-------------------------------------------------\n"  # noqa 501
            )
            try:
                num_epochs = int(line)
                break
            except ValueError:
                print(
                    "\n-------------------------------------------------\nInvalid input. Please enter a number.\n-------------------------------------------------\n"  # noqa 501
                )
        time.sleep(1)

        trainer = GenerationModelTrainer(
            "google/flan-t5-base",
            has_encoder=True,
            executor_batch_size=train_batch_size,
            tokenizer_max_length=1024,
            sequence_max_length=1280,
        )
        args_output_root = Path("result/training_output")
        args_output_root.mkdir(parents=True, exist_ok=True)
        print(
            "\n-------------------------------------------------\nStart training.\n-------------------------------------------------\n"  # noqa 501
        )
        # trained_model, trained_tokenizer = trainer.train_model(
        #     hyperparameter_choices={
        #         "output_dir": str(args_output_root),
        #         "save_strategy": "epoch",
        #         "num_train_epochs": num_epochs,
        #         "per_device_train_batch_size": train_batch_size,
        #         "evaluation_strategy": "epoch",
        #     },
        #     training_datasets=training_datasets,
        #     validation_datasets=validation_datasets,
        # )
        # trained_model.save_pretrained(trained_model_root)
        # trained_tokenizer.save_pretrained(trained_tokenizer_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            trained_model_root
        ).to(device)
        trained_tokenizer = transformers.AutoTokenizer.from_pretrained(
            trained_tokenizer_root
        )
        print(
            "\n-------------------------------------------------\nFinish training. Evaluate on the test set.\n-------------------------------------------------\n"  # noqa 501
        )
        test_dataset = concatenate_datasets(test_datasets)

        model_executor = GenerationModelExecutor(
            trained_model,
            trained_tokenizer,
            train_batch_size,
            tokenizer_max_length=1024,
            sequence_max_length=1280,
        )
        t5_outputs = model_executor.make_prediction(
            test_set=test_dataset, input_column="model_input"
        )
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            test_dataset,
            "model_output",
            t5_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        print(metric_values)
        with open(RESULT_PATH / "metric.txt", "w") as result_file:
            for metric_name, metric_value in metric_values.items():
                result_file.write(f"{metric_name}: {metric_value}\n")
        status["model_has_been_trained"] = model_has_been_trained = True
        status["trained_model_root"] = str(trained_model_root)
        status["trained_tokenizer_root"] = str(trained_tokenizer_root)
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        print(
            "\n-------------------------------------------------\nModel has been trained and evaluated.\n-------------------------------------------------\n"  # noqa 501
        )

    t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        status["trained_model_root"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        status["trained_tokenizer_root"]
    )
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, 1, tokenizer_max_length=1024, sequence_max_length=1280
    )

    while True:
        prompt = ""
        while True:
            line = input(
                '\n-------------------------------------------------\nEnter your input giving to the model: ("done" to finish entering and "exit" to exit the demo.)\n-------------------------------------------------\n'  # noqa 501
            )
            if line == "done":
                break
            if line == "exit":
                return
            prompt += line + "\n"
        t5_prediction = model_executor.make_single_prediction(prompt)
        print(
            f"\n-------------------------------------------------\n{t5_prediction.prediction}\n-------------------------------------------------\n"  # noqa 501
        )


if __name__ == "__main__":
    main()
