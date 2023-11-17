"""An commend line demo to run the whole system."""

import json
import logging
import os
import time
from pathlib import Path

import datasets
import pyfiglet
import torch
import transformers
import yaml
from datasets import concatenate_datasets, load_from_disk
from termcolor import colored

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.prompt_based import PromptBasedDatasetGenerator
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.demo_creator import create_gradio
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_retriever import DescriptionModelRetriever
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.param_selector import OptunaParamSelector
from prompt2model.prompt_parser import (
    MockPromptSpec,
    PromptBasedInstructionParser,
    TaskType,
)
from prompt2model.utils.config import DEFAULT_HYPERPARAMETERS_SPACE
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


def parse_model_size_limit(line: str, default_size=3e9) -> float:
    """Parse the user input for the maximum size of the model.

    Args:
        line: The user input.
        default_size: The default size to use if the user does not specify a size.
    """
    if len(line.strip()) == 0:
        return default_size
    model_units = {"B": 1e0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15}
    unit_disambiguations = {
        "KB": ["Kb", "kb", "kilobytes"],
        "MB": ["Mb", "mb", "megabytes"],
        "GB": ["Gb", "gb", "gigabytes"],
        "TB": ["Tb", "tb", "terabytes"],
        "PB": ["Pb", "pb", "petabytes"],
        "B": ["b", "bytes"],
    }
    unit_matched = False
    for unit, disambiguations in unit_disambiguations.items():
        for unit_name in [unit] + disambiguations:
            if line.strip().endswith(unit_name):
                unit_matched = True
                break
        if unit_matched:
            break
    if unit_matched:
        numerical_part = line.strip()[: -len(unit_name)].strip()
    else:
        numerical_part = line.strip()

    if not str.isdecimal(numerical_part):
        raise ValueError(
            "Invalid input. Please enter a number (integer " + "or number with units)."
        )
    scale_factor = model_units[unit] if unit_matched else 1
    return int(numerical_part) * scale_factor


# tasks = [
# """
# In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous.

# "input": "Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will \"give him lodgings for nothing, and never ask for change\". Grateful for the unexpected assistance, Oliver follows the Dodger to the \"old gentleman's\" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.",
# "output": "Who believes Fagin's gang make wallets and handkerchiefs?.",
# "explanation": "This question is based on the following sentence in the passage \"He believes they make wallets and handkerchiefs\". It evaluates the understanding that the pronoun \"he\" refers to name \"Oliver\". You can ask questions like this one about most pronouns in a paragraph."

# "input": "Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will \"give him lodgings for nothing, and never ask for change\". Grateful for the unexpected assistance, Oliver follows the Dodger to the \"old gentleman's\" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.",
# "output": "What is the alias of the person whose sidekick had a humorous nature?.",
# "explanation": "This question is based on the following sentence in the passage \"Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates\". The pronoun \"his\" refers to a person with multiple names. But since the question explicitly asks for the alias, the answer is unambiguous."
# """,
# """
# "In this task, you are given a date in \"mm/dd/yyyy\" format. You need to check if the date is valid or not. Return 1 if it is valid, else return 0. A date is valid is the components month(\"mm\"), day(\"dd\") and year(\"yyyy\") are all valid individually. A day(dd) is valid if it is greater than or equal to 1 and less than 30 or 31 depending upon the month(mm). Months which have 31 days are January, March, May, July, August, October, December. Rest of the months have 30 days except February which has 28 days if it is not a leap year and 29 days if it is a leap year. A month(mm) is valid if it lies in the range from 1 to 12 as there are 12 months in a year. A year is always valid if it is expressed in the form of \"yyyy\"."

# "input": "14/25/1405",
# "output": "0",
# "explanation": "It is an invalid date as the month(mm) is 14 which does not lie in the range 1 to 12."

# "input": "07/29/1617",
# "output": "1",
# "explanation": "It is a valid date as month(mm), day(dd) and year(yyyy) are all valid."

# """,
# """
# Given an Amazon review, indicate whether it is a 'Positive Review' or 'Negative Review'.
#  "input": "Bought cables in 3ft, 6ft and 9ft.  NONE of them worked.  NO FUNCTIONALITY WHATSOEVER.  Tested many times, its as if the copper wires are just not connected to the terminations.  Do these even go through Quality Control before they leave the factory?  Waste of money and time.",
#  "output": "Negative Review",
#  "explanation": "User did not like cables at all and found all of them useless so it is a negative review."
# """
# ]


# tasks =[
# """
#  "You are given a sentence in English. Your job is to translate the English sentence into Italian."
#  {
#             "input": "Why? Because that profit allows whatever solution we've created to be infinitely scalable.",
#             "output": "Perché? Perché quel profitto fa sì che qualunque soluzione da noi creata sia infinitamente riproducibile su scala.",
#             "explanation": "The English sentence is correctly translated into Italian, because the meaning is preserved."
#         },
#         {
#             "input": "I chose to build there a blessed life.",
#             "output": "Ho scelto di costruirmi una vita fortunata.",
#             "explanation": "The English sentence is correctly translated into Italian, because the meaning is preserved."
#         },
#         {
#             "input": "These are really hybrids, not pure animals.",
#             "output": "Questi sono ibridi veri, non animali puri.",
#             "explanation": "The English sentence is correctly translated into Italian, because the meaning is preserved."
#         }
# """,
# """
# "You need to answer a given question containing a blank (_). Your answer must be one of the two objects mentioned in the question, for example \"trophy\" and \"suitcase\". Your answer must not contain a word that is not present in the question. Please don't use articles (e.g., the, a) before the answer."
# {
#             "input": "The trophy doesn't fit into the brown suitcase because _ is too large.",
#             "output": "trophy",
#             "explanation": "Answer is one of the objects (\"trophy\" and \"suitcase\") in the question. Since the blank is a \"large\" object that didn't fit the \"suitcase\", the answer must be \"trophy\"."
#         },
#         {
#             "input": "Grace was happy to trade me her sweater for my jacket. She thinks _ looks dowdy on her.",
#             "output": "sweater",
#             "explanation": "The word \"dowdy\" decides the answer among the objects (\"sweater\" and \"jacket\") present in the question."
#         },
#         {
#             "input": "While redecorating her home, Sam took out the carpet and replaced it with wood floors. The _ was old.",
#             "output": "carpet",
#             "explanation": "The blank is \"old\", it must be what gets \"replaced\", which has to be \"carpet\"."
#         },
# """,
# """
#  "You are given a question and some answer options (associated with \"A\", \"B\", \"C\", \"D\"). You should choose the correct answer based on commonsense knowledge. Avoid answering questions based on associations, the set of answers are chosen deliberately to capture common sense beyond associations. Do not generate anything else apart from one of the following characters: 'A', 'B, 'C', 'D', 'E' and only give one answer for each question."
#   {
#             "input": "Where would you find magazines along side many other printed works?\n(A)doctor (B)bookstore (C)market (D)train station (E)mortuary",
#             "output": "B",
#             "explanation": "libraries contains magazines and many other printed works."
#         },
#         {
#             "input": "What island country is ferret popular?\n(A)own home (B)north carolina (C)great britain (D)hutch (E)outdoors",
#             "output": "C",
#             "explanation": "great britain is the only island country in the choices."
#         }
# """

# ]


def main():
    """The main function running the whole system."""
    # print_logo()
    # Save the status of Prompt2Model for this session,
    # in case the user wishes to stop and continue later.
    # if os.path.isfile("status.yaml"):
    #     with open("status.yaml", "r") as f:
    #         status = yaml.safe_load(f)
    # else:
    status = {}
    import pandas as pd

    df = pd.read_csv("prompt2model/dataset_retriever/reranking.csv")
    for idx, row in df.iterrows():
        task = row["Task"]

        propmt_has_been_parsed = status.get("prompt_has_been_parsed", False)
        dataset_has_been_retrieved = status.get("dataset_has_been_retrieved", False)
        model_has_been_retrieved = status.get("model_has_been_retrieved", False)
        dataset_has_been_generated = status.get("dataset_has_been_generated", False)
        model_has_been_trained = status.get("model_has_been_trained", False)
        # if not propmt_has_been_parsed:
        if True:
            prompt = ""
            line_print(
                "Enter your task description and few-shot examples (or 'done' to finish):"
            )
            time.sleep(2)
            # while True:
            #     line = input()
            #     if line == "done":
            #         break
            #     prompt += line + "\n"
            # line_print("Parsing prompt...")
            prompt_spec = PromptBasedInstructionParser(
                task_type=TaskType.TEXT_GENERATION
            )
            print(task)
            prompt_spec.parse_from_prompt(task)

            propmt_has_been_parsed = True
            status["instruction"] = prompt_spec.instruction
            status["examples"] = prompt_spec.examples
            status["prompt_has_been_parsed"] = False
            with open("status.yaml", "w") as f:
                yaml.safe_dump(status, f)
            line_print("Prompt parsed.")

        # if propmt_has_been_parsed and not dataset_has_been_retrieved:
        if True:
            prompt_spec = MockPromptSpec(
                TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
            )
            line_print("Retrieving dataset...")
            retriever = DescriptionDatasetRetriever()
            # TODO: Change me back
            (
                retrieved_dataset_dict,
                reranking_prompt,
                dataset_name,
                config_name,
            ) = retriever.retrieve_dataset_dict(prompt_spec)
            df["Prompt"] = reranking_prompt
            df["Reranker_Dataset"] = dataset_name
            df["Reranker_Config"] = config_name

            dataset_has_been_retrieved = True
            if retrieved_dataset_dict is not None:
                print("After everything: Dataset retrieved successfully!!")
                retrieved_dataset_dict.save_to_disk("retrieved_dataset_dict")
                status["retrieved_dataset_dict_root"] = "retrieved_dataset_dict"
            else:
                print("After everything: Dataset not retrieved successfully.")
                status["retrieved_dataset_dict_root"] = None
            status["dataset_has_been_retrieved"] = False
            with open("status.yaml", "w") as f:
                yaml.safe_dump(status, f)
    df.to_csv("prompt2model/dataset_retriever/reranking2.csv", index=False)
    return

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and not model_has_been_retrieved
    ):
        line_print(
            "Enter the maximum size of the model (by default, enter nothing "
            + "and we will use 3GB as the limit). You can specify a unit (e.g. "
            + "3GB, 300Mb). If no unit is given, we assume the size is given in bytes."
        )
        max_size_line = input()
        max_size = parse_model_size_limit(max_size_line)
        line_print(f"Maximum model size set to {max_size} bytes.")

        line_print("Retrieving model...")
        prompt_spec = MockPromptSpec(
            TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
        )
        retriever = DescriptionModelRetriever(
            model_descriptions_index_path="huggingface_data/huggingface_models/model_info/",  # noqa E501
            use_bm25=True,
            use_HyDE=True,
            model_size_limit_bytes=max_size,
        )
        top_model_name = retriever.retrieve(prompt_spec)
        line_print("Here are the models we retrieved.")
        for idx, each in enumerate(top_model_name):
            line_print(f"# {idx + 1}: {each}")
        while True:
            line_print(
                "Enter the number of the model you want to use. Range from 1 to 5."
            )
            line = input()
            try:
                rank = int(line)
                assert 1 <= rank <= 5
                break
            except Exception:
                line_print("Invalid input. Please enter a number.")
        model_has_been_retrieved = True
        status["model_has_been_retrieved"] = True
        status["model_name"] = top_model_name[rank - 1]
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
            line = input()
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
            prompt_spec, num_expected, split=DatasetSplit.TRAIN
        )
        generated_dataset.save_to_disk("generated_dataset")
        dataset_has_been_generated = True
        status["dataset_has_been_generated"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("The generated dataset is ready.")
        time.sleep(2)

    if (
        propmt_has_been_parsed
        and dataset_has_been_retrieved
        and model_has_been_retrieved
        and dataset_has_been_generated
        and not model_has_been_trained
    ):
        line_print("The model has not been trained.")
        time.sleep(2)
        dataset_root = Path("generated_dataset")
        if not dataset_root.exists():
            raise ValueError("Dataset has not been generated yet.")
        trained_model_root = Path("result/trained_model")
        trained_tokenizer_root = Path("result/trained_tokenizer")
        RESULT_PATH = Path("result/result")
        trained_model_root.mkdir(parents=True, exist_ok=True)
        trained_tokenizer_root.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.mkdir(parents=True, exist_ok=True)
        dataset = load_from_disk(dataset_root)
        if status["retrieved_dataset_dict_root"] is not None:
            cached_retrieved_dataset_dict = datasets.load_from_disk(
                status["retrieved_dataset_dict_root"]
            )
            dataset_list = [dataset, cached_retrieved_dataset_dict["train"]]
        else:
            dataset_list = [dataset]

        line_print("Processing datasets.")
        instruction = status["instruction"]
        t5_processor = TextualizeProcessor(has_encoder=True)
        t5_modified_dataset_dicts = t5_processor.process_dataset_lists(
            instruction,
            dataset_list,
            train_proportion=0.7,
            val_proportion=0.1,
            maximum_example_num={"train": 3500, "val": 500, "test": 1000},
        )
        processor_logger = get_formatted_logger("DatasetProcessor")
        processor_logger.setLevel(logging.INFO)
        training_datasets = []
        validation_datasets = []
        test_datasets = []
        for idx, modified_dataset_dict in enumerate(t5_modified_dataset_dicts):
            training_datasets.append(modified_dataset_dict["train"])
            validation_datasets.append(modified_dataset_dict["val"])
            test_datasets.append(modified_dataset_dict["test"])
        trainer_logger = get_formatted_logger("ModelTrainer")
        trainer_logger.setLevel(logging.INFO)
        evaluator_logger = get_formatted_logger("ModelEvaluator")
        evaluator_logger.setLevel(logging.INFO)

        train_batch_size = None

        while True:
            line = input(
                "Are you interested to train the model with automatic hyperparameter search? Type 'y' for Yes and 'n' for No. "  # noqa E501
            )
            try:
                assert line in ["y", "n"]
                break
            except Exception:
                line_print("The answer should be either y or n")
        time.sleep(1)

        if line == "y":
            line_print("Starting training with hyperparameter selection.")
            default_min_num_epochs = DEFAULT_HYPERPARAMETERS_SPACE[
                "min_num_train_epochs"
            ]
            min_num_epochs = input(
                f"Enter min number of epochs. Press enter to use default value ({default_min_num_epochs}): "  # noqa E501
            )
            default_max_num_epochs = DEFAULT_HYPERPARAMETERS_SPACE[
                "max_num_train_epochs"
            ]
            max_num_epochs = input(
                f"Enter max number of epochs. Press enter to use default value ({default_max_num_epochs}): "  # noqa E501
            )
            default_num_trials = 10
            num_trials = input(
                f"Enter the number of trials (maximum number of hyperparameter configurations to consider) for hyperparameter search. Press enter to use default value ({default_num_trials}): "  # noqa E501
            )
            default_batch_size = DEFAULT_HYPERPARAMETERS_SPACE[
                "per_device_train_batch_size"
            ]  # noqa E501
            max_batch_size = input(
                "Enter the max batch size. "
                + f"Press enter to use default ({default_batch_size}): "
            )

            min_num_epochs = (
                default_min_num_epochs if min_num_epochs == "" else eval(min_num_epochs)
            )
            max_num_epochs = (
                default_max_num_epochs if max_num_epochs == "" else eval(max_num_epochs)
            )
            num_trials = 1 if num_trials == "" else eval(num_trials)

            max_batch_size = (
                DEFAULT_HYPERPARAMETERS_SPACE["per_device_train_batch_size"]
                if max_batch_size == ""
                else eval(max_batch_size)
            )

            trainer = GenerationModelTrainer(
                status["model_name"],
                has_encoder=True,
                executor_batch_size=max_batch_size,
                tokenizer_max_length=1024,
                sequence_max_length=1280,
            )
            args_output_root = Path("result/training_output")
            args_output_root.mkdir(parents=True, exist_ok=True)
            line_print("Starting training.")

            trained_model, trained_tokenizer = OptunaParamSelector(
                n_trials=num_trials,
                trainer=trainer,
            ).select_from_hyperparameters(
                training_datasets=training_datasets,
                validation=validation_datasets,
                hyperparameters={
                    "min_num_train_epochs": min_num_epochs,
                    "max_num_train_epochs": max_num_epochs,
                    "per_device_train_batch_size": [max_batch_size],
                },
            )
            train_batch_size = max_batch_size

        else:
            line_print("Starting training without hyperparameter selection.")
            while True:
                line = input("Enter the training batch size:")
                try:
                    train_batch_size = int(line)
                    assert 0 < train_batch_size
                    break
                except Exception:
                    line_print("The training batch size must be greater than 0.")
            time.sleep(1)

            while True:
                line = input("Enter the number of epochs to train for:")
                try:
                    num_epochs = int(line)
                    break
                except ValueError:
                    line_print("Invalid input. Please enter a number.")
            time.sleep(1)

            trainer = GenerationModelTrainer(
                status["model_name"],
                has_encoder=True,
                executor_batch_size=train_batch_size,
                tokenizer_max_length=1024,
                sequence_max_length=1280,
            )
            args_output_root = Path("result/training_output")
            args_output_root.mkdir(parents=True, exist_ok=True)
            line_print("Starting training.")
            trained_model, trained_tokenizer = trainer.train_model(
                hyperparameter_choices={
                    "output_dir": str(args_output_root),
                    "save_strategy": "epoch",
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": train_batch_size,
                    "evaluation_strategy": "epoch",
                },
                training_datasets=training_datasets,
                validation_datasets=validation_datasets,
            )

        trained_model.save_pretrained(trained_model_root)
        trained_tokenizer.save_pretrained(trained_tokenizer_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            trained_model_root
        ).to(device)
        trained_tokenizer = transformers.AutoTokenizer.from_pretrained(
            trained_tokenizer_root
        )
        line_print("Finished training. Now evaluating on the test set.")
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
        line_print(metric_values)
        with open(RESULT_PATH / "metric.txt", "w") as result_file:
            for metric_name, metric_value in metric_values.items():
                result_file.write(f"{metric_name}: {metric_value}\n")
        status["model_has_been_trained"] = model_has_been_trained = True
        status["trained_model_root"] = str(trained_model_root)
        status["trained_tokenizer_root"] = str(trained_tokenizer_root)
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        line_print("Model has been trained and evaluated.")

    t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        status["trained_model_root"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        status["trained_tokenizer_root"]
    )
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, 1, tokenizer_max_length=1024, sequence_max_length=1280
    )
    prompt_spec = MockPromptSpec(
        TaskType.TEXT_GENERATION, status["instruction"], status["examples"]
    )
    interface_t5 = create_gradio(model_executor, prompt_spec)
    interface_t5.launch(share=True)


if __name__ == "__main__":
    main()
