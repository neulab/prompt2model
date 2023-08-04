"""An dual-encoder dataset retriever using HuggingFace datasets descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os

import datasets
import numpy as np
import torch
from tqdm import tqdm

from prompt2model.dataset_retriever.base import DatasetRetriever
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects

NO_DATASET_FOUND = "NO_DATASET"
class DatasetInfo:
    """Store the dataset name, description, and query-dataset score for each dataset."""

    def __init__(
        self,
        name: str,
        description: str,
        score: float,
    ):
        """Initialize a DatasetInfo object.

        Args:
            name: The name of the dataset.
            description: The description of the dataset.
            score: The similarity of the dataset to a given prompt from a user.
        """
        self.name = name
        self.description = description
        self.score = score


class DescriptionDatasetRetriever(DatasetRetriever):
    """Retrieve a dataset from HuggingFace, based on similarity to the prompt."""

    def __init__(
        self,
        search_index_path: str = "huggingface_data/huggingface_datasets/huggingface_datasets_datafinder_index",
        search_depth: int = 5,
        encoder_model_name: str = "viswavi/datafinder-scibert-nl-queries-dataset-description-only",
        dataset_info_file: str = "huggingface_data/huggingface_datasets/dataset_index.json",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize a dual-encoder retriever against a search index.

        Args:
            search_index_path: Where to store the search index (e.g. encoded vectors).
            search_depth: The number of most-relevant datasets to retrieve.
            encoder_model_name: The name of the model to use for the dual-encoder.
            dataset_info_file: The file containing dataset names and descriptions.
            device: The device to use for encoding text for our dual-encoder model.
        """
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.encoder_model_name = encoder_model_name
        self.dataset_info = json.load(open(dataset_info_file, 'r'))
        self.dataset_names = list(self.dataset_info.keys())
        self.dataset_infos: list[DatasetInfo] = []
        for dataset_name in self.dataset_info:
            self.dataset_infos.append(
                DatasetInfo(
                    name=dataset_name,
                    description=self.dataset_info[dataset_name]["description"],
                    score=0.0,
                )
            )
        self.device = device

        assert not os.path.isdir(
            search_index_path
        ), f"Search index must either be a valid file or not exist yet. But {search_index_path} is provided."  # noqa 501


    def encode_dataset_descriptions(self, search_index_path) -> np.ndarray:
        """Encode dataset descriptions into a vector for indexing."""
        dataset_descriptions = [self.dataset_info[dataset_name] for dataset_name in self.dataset_names]
        dataset_vectors = encode_text(
            self.encoder_model_name,
            text_to_encode=dataset_descriptions,
            encoding_file=search_index_path,
            device=self.device,
        )
        return dataset_vectors

    def choose_dataset(self, top_datasets: list[str]) -> str:
        """Have the user choose an appropriate dataset from a list of top datasets."""
        raise NotImplementedError

    def canonicalize_dataset(self, dataset_name: str) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        return datasets.load_dataset(dataset_name)

    def retrieve_dataset_dict(
        self,
        prompt_spec: PromptSpec,
        similarity_threshold: float = 0.0,
    ) -> list[datasets.DatasetDict]:
        """Select a dataset from a prompt using a dual-encoder retriever.

        Args:
            prompt: A prompt whose instruction field we use to select relevant datasets.

        Return:
            A list of relevant datasets dictionaries.
        """
        if not os.path.exists(self.search_index_path):
            self.encode_model_descriptions(self.search_index_path)

        query_text = prompt_spec.instruction

        query_vector = encode_text(
            self.encoder_model_name,
            text_to_encode=query_text,
            device=self.device,
        )
        ranked_list = retrieve_objects(
            query_vector, self.search_index_path, self.search_depth
        )
        for dataset_idx_str, dataset_score in ranked_list:
            dataset_idx = int(dataset_idx_str)
            self.dataset_infos[dataset_idx].score = dataset_score

        ranked_list = sorted(self.dataset_infos, key=lambda x: x.score, reverse=True)
        assert len(ranked_list) > 0, "No datasets retrieved from search index."
        top_dataset_similarity = ranked_list[0].score
        if top_dataset_similarity < similarity_threshold:
            return [datasets.DatasetDict()]
        top_datasets = [dataset_info.name for dataset_info in ranked_list]
        chosen_dataset = self.choose_dataset(top_datasets)
        return self.canonicalize_dataset(chosen_dataset)
