"""An dual-encoder dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os
import random
import urllib.request

import datasets
import torch

from prompt2model.dataset_retriever.base import DatasetInfo, DatasetRetriever
from prompt2model.dataset_retriever.column_selection_prompt import (
    construct_prompt_for_column_selection,
)
from prompt2model.dataset_retriever.reranking_prompt import (
    construct_prompt_for_dataset_reranking,
)
from prompt2model.dataset_transformer.prompt_based import PromptBasedDatasetTransformer
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, get_formatted_logger, retrieve_objects
from prompt2model.utils.dataset_utils import get_dataset_size
from prompt2model.utils.parse_responses import parse_prompt_to_fields

datasets.utils.logging.disable_progress_bar()
logger = get_formatted_logger("DescriptionDatasetRetriever")


class DescriptionDatasetRetriever(DatasetRetriever):
    """Retrieve a dataset from HuggingFace, based on similarity to the prompt."""

    def __init__(
        self,
        search_index_path: str = "huggingface_data/huggingface_datasets/"
        + "huggingface_datasets_datafinder_index",
        first_stage_search_depth: int = 1000,
        max_search_depth: int = 25,
        encoder_model_name: str = "viswavi/datafinder-huggingface-prompt-queries",
        dataset_info_file: str = "huggingface_data/huggingface_datasets/"
        + "dataset_index.json",
        reranking_dataset_info_file="huggingface_data/huggingface_datasets/"
        + "reranking_dataset_index.json",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        max_number_of_dataset_rows=3000,
        allow_gated_datasets=False,
    ):
        """Initialize a dual-encoder retriever against a search index.

        Args:
            search_index_path: Where to store the search index (e.g. encoded vectors).
            first_stage_search_depth: The # of datasets to retrieve before filtering.
            max_search_depth: The number of most-relevant datasets to retrieve.
            encoder_model_name: The name of the model to use for the dual-encoder.
            dataset_info_file: The file containing dataset names and descriptions.
            reranking_dataset_info_file: File containing dataset info used for reranking
            device: The device to use for encoding text for our dual-encoder model.
            max_number_of_dataset_rows: Limit the number of rows for large datasets.
            allow_gated_datasets: Use only if the user explicitly wants gated datasets
        """
        self.search_index_path = search_index_path
        self.first_stage_search_depth = first_stage_search_depth
        self.max_search_depth = max_search_depth
        self.encoder_model_name = encoder_model_name
        self.device = device
        self.dataset_info_file = dataset_info_file
        self.reranking_dataset_info_file = reranking_dataset_info_file
        self.max_number_of_dataset_rows = max_number_of_dataset_rows
        self.allow_gated_datasets = allow_gated_datasets
        self.initialize_search_index()

    def initialize_search_index(self) -> None:
        """Initialize the search index."""
        self.dataset_infos: list[DatasetInfo] = []
        if not os.path.exists(self.dataset_info_file):
            # Download the dataset search index if one is not on disk already.
            logger.info("Downloading the dataset search index")
            os.makedirs(os.path.dirname(self.dataset_info_file), exist_ok=True)
            urllib.request.urlretrieve(
                "http://phontron.com/data/prompt2model/dataset_index.json",
                self.dataset_info_file,
            )
        self.full_dataset_metadata = json.load(open(self.dataset_info_file, "r"))
        for dataset_name in sorted(self.full_dataset_metadata.keys()):
            self.dataset_infos.append(
                DatasetInfo(
                    name=dataset_name,
                    description=self.full_dataset_metadata[dataset_name]["description"],
                    score=0.0,
                )
            )
        if not os.path.exists(self.reranking_dataset_info_file):
            # Download the reranking index if one is not on disk already.
            logger.info("Downloading the Reranking Dataset Index File")
            urllib.request.urlretrieve(
                "http://phontron.com/data/prompt2model/dataset_reranking_index.json",
                self.reranking_dataset_info_file,
            )
        with open(self.reranking_dataset_info_file, "r") as f:
            self.reranking_datasets_infos = json.load(f)

        if os.path.isdir(self.search_index_path):
            raise ValueError(
                "Search index must either be a valid file or not exist yet. "
                "But {self.search_index_path} is provided."
            )
        if not os.path.exists(self.search_index_path):
            logger.info("Creating dataset descriptions")
            encode_text(
                self.encoder_model_name,
                text_to_encode=[x.description for x in self.dataset_infos],
                encoding_file=self.search_index_path,
                device=self.device,
            )

    # ---------------------------- Utility Functions ----------------------------
    @staticmethod
    def _input_string():
        """Utility function to read a string from stdin."""
        description = str(input())
        return description

    @staticmethod
    def _input_y_n() -> bool:
        """Utility function to get a yes/no answer from the user via stdin."""
        y_n = str(input())
        return not (y_n.strip() == "" or y_n.strip().lower() == "n")

    @staticmethod
    def _print_divider():
        """Utility function to assist with the retriever's command line interface."""
        print("\n-------------------------------------------------\n")

    def choose_dataset_by_cli(self, top_datasets: list[DatasetInfo]) -> str | None:
        """Have the user choose an appropriate dataset from a list of top datasets.

        Args:
            top_datasets: A list of top datasets to choose from.

        Returns:
            The name of the chosen dataset, or None if no dataset is chosen as relevant.
        """
        self._print_divider()
        print("Here are the datasets I've retrieved for you:")
        print("#\tName\tSize[MB]\tDescription")
        for i, d in enumerate(top_datasets):
            description_no_space = d.description.replace("\n", " ")
            print(
                f"{i+1}):\t{d.name}\t{get_dataset_size(d.name)}\t{description_no_space}"
            )

        self._print_divider()
        print(
            "If none of these are relevant to your prompt, we'll only use "
            + "generated data. Are any of these datasets relevant? (y/N)"
        )
        any_datasets_relevant = self._input_y_n()
        if any_datasets_relevant:
            print(
                "Which dataset would you like to use? Give the number between "
                + f"1 and {len(top_datasets)}."
            )
            dataset_idx = int(input())
            chosen_dataset_name = top_datasets[dataset_idx - 1].name
        else:
            chosen_dataset_name = None
        self._print_divider()
        return chosen_dataset_name

    @staticmethod
    def canonicalize_dataset_using_columns_for_split(
        dataset_split: datasets.Dataset,
        input_columns: list[str],
        output_column: str,
        max_number_of_rows: int,
    ) -> datasets.DatasetDict:
        """Canonicalize a single dataset split into a suitable text-to-text format."""
        input_col = []
        output_col = []
        for i in range(min(len(dataset_split), max_number_of_rows)):
            curr_string = ""
            for col in input_columns:
                curr_string += f"{col}: {dataset_split[i][col]}\n"
            curr_string = curr_string.strip()
            input_col.append(curr_string)
            output_col.append(dataset_split[i][output_column])
        return datasets.Dataset.from_dict(
            {"input_col": input_col, "output_col": output_col}
        )

    def get_all_dataset_infos(self, dataset_list: list[str]) -> dict:
        """Gather all information about a list of datasets.

        This function iterates over a list of dataset names and retrieves
        their information from a stored dataset information dictionary. It
        filters out datasets based on whether users allows for gated_datasets and
        limits the number of configurations to a maximum of 5 for each dataset.

        Args:
            dataset_list: A list of dataset names to retrieve information for.

        Returns:
            dict: A dictionary containing information about the requested datasets.
                The keys are dataset names and the values are dictionaries
                with dataset information.
        """
        dataset_info_dict = {}
        for dataset_name in dataset_list:
            if dataset_name not in self.reranking_datasets_infos:
                continue
            if (
                self.reranking_datasets_infos[dataset_name]["is_gated"]
                != self.allow_gated_datasets
            ):
                continue
            curr_dataset = self.reranking_datasets_infos[dataset_name]
            if len(curr_dataset["configs"]) > 5:
                curr_dataset["configs"] = dict(
                    random.sample(list(curr_dataset["configs"].items()), 5)
                )
            dataset_info_dict[dataset_name] = curr_dataset

        return dataset_info_dict

    @staticmethod
    def automatic_column_selection(
        instruction: str,
        dataset_name: str,
        dataset_description: str,
        dataset_columns: str,
        example_rows: dict,
    ) -> tuple[list[str], str]:
        """Find appropriate input and output columns for a given dataset and tasks."""
        prompt = construct_prompt_for_column_selection(
            instruction,
            dataset_name,
            dataset_description,
            dataset_columns,
            example_rows,
        )
        required_keys = ["input", "output"]
        optional_keys = ["ambiguous", "irrelevant"]

        response = parse_prompt_to_fields(prompt, required_keys, optional_keys)
        input_columns = response["input"]
        output_column = response["output"]
        if len(input_columns) < 1 or len(output_column) != 1:
            raise RuntimeError(
                "Input columns length was less than 1 or output column length was not 1"
            )

        dataset_columns = dataset_columns
        incorrect_columns = [
            col for col in input_columns + output_column if col not in dataset_columns
        ]
        if len(incorrect_columns) > 0:
            raise RuntimeError(
                f"One or more columns ({incorrect_columns}) were output that were "
                f"not in the list of columns in the dataset ({dataset_columns})."
            )

        return input_columns, output_column[0]

    def canonicalize_dataset_using_columns(
        self,
        dataset: datasets.DatasetDict,
        input_columns: list[str],
        output_columns: str,
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        dataset_dict = {}
        for split in dataset:
            dataset_dict[split] = self.canonicalize_dataset_using_columns_for_split(
                dataset[split],
                input_columns,
                output_columns,
                self.max_number_of_dataset_rows,
            )
        return datasets.DatasetDict(dataset_dict)

    def canonicalize_dataset_by_cli(
        self, dataset_name: str, prompt_spec
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format.

        Args:
            dataset_name: The name of the dataset to canonicalize.

        Returns:
            A canonicalized dataset.
        """
        configs = datasets.get_dataset_config_names(dataset_name)
        chosen_config = None
        if len(configs) == 1:
            chosen_config = configs[0]
        else:
            self._print_divider()
            print(f"Multiple dataset configs available: {configs}")
            while chosen_config is None:
                print("Which dataset config would you like to use for this?")
                user_response = self._input_string()
                if user_response in configs:
                    chosen_config = user_response
                else:
                    print(
                        f"Invalid config provided: {user_response}. Please choose "
                        + "from {configs}\n\n"
                    )
            self._print_divider()

        dataset_info = self.get_all_dataset_infos([dataset_name])
        if len(dataset_info.keys()) == 0:
            return None
        dataset_info = dataset_info[dataset_name]["configs"][chosen_config]
        if dataset_info is None:
            return None
        try:
            input_columns, output_column = self.automatic_column_selection(
                prompt_spec.instruction,
                dataset_info["dataset_name"],
                dataset_info["dataset_description"],
                dataset_info["columns"],
                dataset_info["sample_row"],
            )
        except RuntimeError:
            logger.error(f"{dataset_name} did not work. Try another!")
            return None  # Returning None means that the dataset chosen didn't work,
            # and we would rather generate a dataset.

        print(f"Will use the columns {json.dumps(input_columns)} as input.\n")
        print(f'Will use the column "{output_column}" as our target.\n')
        self._print_divider()

        dataset = datasets.load_dataset(
            dataset_info["dataset_name"], dataset_info["config_name"]
        ).flatten()
        dataset = dataset.rename_columns(dataset_info["columns_mapping"])

        canonicalized_dataset = self.canonicalize_dataset_using_columns(
            dataset, input_columns, output_column
        )
        return canonicalized_dataset

    def retrieve_top_datasets(
        self,
        prompt_spec: PromptSpec,
    ) -> list[str]:
        """Retrieve the top datasets for a prompt.

        Specifically, the datasets are scored using a dual-encoder retriever model
        and the datasets with the highest similarity scores with the query are returned.

        Args:
            prompt_spec: A prompt whose instruction field we use to retrieve datasets.

        Returns:
            A list of the top datasets for the prompt according to retriever score.
        """
        query_vector = encode_text(
            self.encoder_model_name,
            text_to_encode=prompt_spec.instruction,
            device=self.device,
        )

        ranked_list = retrieve_objects(
            query_vector,
            self.search_index_path,
            [x.name for x in self.dataset_infos],
            self.first_stage_search_depth,
        )
        top_dataset_infos = []
        dataset_name_to_dataset_idx = {
            d.name: i for i, d in enumerate(self.dataset_infos)
        }
        for dataset_name, dataset_score in ranked_list:
            dataset_idx = dataset_name_to_dataset_idx[dataset_name]
            self.dataset_infos[dataset_idx].score = dataset_score
            top_dataset_infos.append(self.dataset_infos[dataset_idx])

        sorted_list = sorted(top_dataset_infos, key=lambda x: x.score, reverse=True)[
            : self.max_search_depth
        ]
        if len(sorted_list) == 0:
            raise ValueError("No datasets retrieved from search index.")
        dataset_names = [x.name for x in sorted_list]
        return dataset_names

    def rerank_datasets(self, dataset_list: list[str], prompt_spec: PromptSpec):
        """Rerank datasets based on relevance to a given prompt specification.

        This function takes a list of datasets and a prompt specification,
        and reranks the datasets based on their relevance to the prompt. It
        first gathers detailed information about each dataset in the list using the
        `get_all_dataset_infos` method. Then, it constructs a prompt for reranking
        and parses its response to identify the most relevant dataset and
        configuration. The function also includes checks for the validity of the
        response(hallucinations) and the confidence level of the dataset
        recommendation.

        Args:
            dataset_list: A list of dataset names to be reranked.
            prompt_spec: An object containing the prompt specification,
                        ncluding instruction and examples, used for reranking datasets.

        Returns:
            dict or None: The most relevant dataset configuration, or None if
                no suitable dataset is found or if the confidence level
                in the recommendation is low.
        """
        dataset_info_dict = self.get_all_dataset_infos(dataset_list)

        if len(dataset_info_dict.keys()) == 0:
            return None
        prompt = construct_prompt_for_dataset_reranking(
            prompt_spec.instruction, prompt_spec.examples, dataset_info_dict
        )
        response = parse_prompt_to_fields(prompt=prompt, module_name="rerank")

        dataset_name, config_name, confidence_level = (
            response["dataset_name"],
            response["config_name"],
            response["confidence_level"],
        )
        if (
            dataset_name not in dataset_info_dict
            or config_name not in dataset_info_dict[dataset_name]["configs"]
        ):
            logger.warning("LLM hallucinated dataset/config name")
            return None
        if confidence_level == "low":
            logger.warning("Confidence in retrieved dataset is low.")
            return None

        return dataset_info_dict[dataset_name]["configs"][config_name]

    def canonicalize_dataset_automatically(
        self,
        top_dataset_info: dict,
        prompt_spec: PromptSpec,
        auto_transform_data: bool = False,
        num_points_to_transform: int = 10,
    ):
        """Automatically canonicalize dataset (instead of cli).

        This function automates the canonicalization of the
        dataset identified as the most relevant. It starts by checking if
        the top dataset information exists. If so, it proceeds to automatically
        select the input and output columns based on the task instruction. The
        dataset is then loaded, flattened, and renamed according to the columns
        mapping. If auto_transform_data is true, num_points_to_transform points
        from the dataset are transformed by an LLM to desired format according
        to the prompt_spec, and transformed dataset is returned. If
        auto_transform_data is false, the dataset is canonicalized using the
        selected columns.

        Args:
            top_dataset_info: Contains info about the top-ranked dataset.
            prompt_spec: prompt object storing the original task and examples.
            auto_transform_data: Specifies whether a dataset is to be
            transformed. Samples from the original dataset will be transformed
            by an LLM to match a desired format as specified by prompt_spec.
            num_points_to_transform: Number of data points you wish to
            transform. Number must be greater than zero. If number is greater
            than size of dataset, whole dataset will be transformed. ignored
            if data_transform is False.

        Returns:
            The canonicalized dataset, or None if the dataset is invalid or
            if column selection fails, or if any other error occurs
            during the process.
        """
        task_instruction = prompt_spec.instruction
        if top_dataset_info is None:
            logger.warning("None of the retrieved datasets were relevant.")
            return None
        try:
            input_columns, output_column = self.automatic_column_selection(
                task_instruction,
                top_dataset_info["dataset_name"],
                top_dataset_info["dataset_description"],
                top_dataset_info["columns"],
                top_dataset_info["sample_row"],
            )
        except Exception as e:
            logger.warning("Column selection failed: ", e)
            return None
        logger.info("Column selection completed")
        full_dataset = (
            datasets.load_dataset(
                top_dataset_info["dataset_name"], top_dataset_info["config_name"]
            )
            .shuffle()
            .flatten()
        )
        full_dataset = full_dataset.rename_columns(top_dataset_info["columns_mapping"])
        logger.info("Dataset loaded")

        if auto_transform_data:
            # remove columns not selected by automatic column selection
            full_dataset = full_dataset.remove_columns(
                [
                    col_name
                    for col_name in full_dataset["train"].column_names
                    if col_name not in input_columns + [output_column]
                ]
            )
            logger.info("Unnecessary columns removed")

            dataset_transformer = PromptBasedDatasetTransformer()
            canonicalized_dataset = dataset_transformer.transform_data(
                prompt_spec=prompt_spec,
                dataset=full_dataset["train"],
                num_points_to_transform=num_points_to_transform,
            )
            logger.info("Data transformation completed")

            example_rows = json.dumps(canonicalized_dataset["train"][0], indent=4)

            logger.info(f"Transformed dataset. Example row:\n{example_rows}\n")

            return canonicalized_dataset
        else:
            canonicalized_dataset = self.canonicalize_dataset_using_columns(
                full_dataset, input_columns, output_column
            )
            logger.info(
                f"No transformation. Using dataset {top_dataset_info['dataset_name']}"
            )  # noqa E501
            return canonicalized_dataset

    def retrieve_dataset_dict(
        self,
        prompt_spec: PromptSpec,
        auto_transform_data: bool = False,
        num_points_to_transform: int = 10,
    ) -> datasets.DatasetDict | None:
        """Select a dataset from a prompt using a dual-encoder retriever.

        Args:
            prompt_spec: prompt object storing the original task and examples.
            auto_transform_data: Specifies whether a dataset is to be
            transformed. Samples from the original dataset will be transformed
            by an LLM to match a desired format as specified by prompt_spec.
            num_points_to_transform: Number of data points you wish to
            transform. Number must be greater than zero. If number is greater
            than size of dataset, whole dataset will be transformed. ignored
            if data_transform is False.

        Return:
            The most relevant dataset, canonicalized;
            or None if there are no relevant datasets.
        """
        sorted_list = self.retrieve_top_datasets(prompt_spec)
        logger.info(f"Top datasets retrieved. Top datasets: {sorted_list}")
        top_dataset_info = self.rerank_datasets(sorted_list, prompt_spec)
        logger.info(f"Rerank completed. Top dataset info: {top_dataset_info}")
        return self.canonicalize_dataset_automatically(
            top_dataset_info, prompt_spec, auto_transform_data, num_points_to_transform
        )
