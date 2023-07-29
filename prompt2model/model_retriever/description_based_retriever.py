"""An dual-encoder model retriever using HuggingFace model descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from prompt2model.model_retriever.base import ModelRetriever
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects


class ModelInfo:
    """Store the model name, description, and query-model score for each model."""

    def __init__(
        self,
        name: str,
        description: str,
        score: float,
    ):
        """Initialize a ModelInfo object.

        Args:
            name: The name of the model.
            description: The description of the model.
            score: The similarity of the model to a given prompt from a user.
        """
        self.name = name
        self.description = description
        self.score = score


class DescriptionModelRetriever(ModelRetriever):
    """Retrieve a model from among HuggingFace models."""

    def __init__(
        self,
        search_index_path: str,
        search_depth: int = 5,
        encoder_model_name: str = "OpenMatch/cocodr-base-msmarco",
        model_descriptions_index_path="huggingface_models/model_info/",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize a dual-encoder retriever against a search index.

        Args:
            search_index_path: Where to store the search index (e.g. encoded vectors).
            search_depth: The number of most-relevant models to retrieve.
            encoder_model_name: The name of the model to use for the dual-encoder.
            model_descriptions_index_path: The directory of models to search against.
            device: The device to use for encoding text for our dual-encoder model.
        """
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.encoder_model_name = encoder_model_name
        self.model_descriptions_index_path = model_descriptions_index_path
        self.device = device

        # Blocklist certain models' organizations to exclude from model retrieval
        # search results; certain organizations programmatically create models which
        # are unlikely to be useful for task-specific finetuning.
        self.model_blocklist_organizations = ["huggingtweets"]
        self.load_model_info()

        assert not os.path.isdir(
            search_index_path
        ), f"Search index must either be a valid file or not exist yet. But {search_index_path} is provided."  # noqa 501

    def load_model_info(self):
        """Load metadata (e.g. downloads, publication date) about various models.

        We load metadata from json files in the model_descriptions_index_path
        directory, filter out models from certain organizations, and initialize a list
        of ModelInfo objects corresponding to the models we want to search against.
        """
        description_files = os.listdir(self.model_descriptions_index_path)
        # We store model names and descriptions in a list of ModelInfo objects.
        self.model_infos: list[ModelInfo] = []
        for f in tqdm(description_files):
            if (
                f.startswith(".")
                or len(open(os.path.join(self.model_descriptions_index_path, f)).read())
                == 0
            ):
                continue
            block = False
            for org in self.model_blocklist_organizations:
                if f.startswith(org + "/"):
                    block = True
                    break
            if block:
                continue
            model_dict = json.load(
                open(os.path.join(self.model_descriptions_index_path, f))
            )
            model_name = model_dict["pretrained_model_name"]
            model_info = ModelInfo(
                name=model_name, description=model_dict["description"], score=None
            )
            self.model_infos.append(model_info)

    def encode_model_descriptions(self, search_index_path) -> np.ndarray:
        """Encode model descriptions into a vector for indexing."""
        model_descriptions = [model.description for model in self.model_infos]
        model_vectors = encode_text(
            self.encoder_model_name,
            text_to_encode=model_descriptions,
            encoding_file=search_index_path,
            device=self.device,
        )
        return model_vectors

    def retrieve(
        self,
        prompt: PromptSpec,
    ) -> list[str]:
        """Select a model from a prompt using a dual-encoder retriever.

        Args:
            prompt: A prompt whose instruction field we use to select relevant models.

        Return:
            A list of relevant models' HuggingFace names.
        """
        if not os.path.exists(self.search_index_path):
            self.encode_model_descriptions(self.search_index_path)

        query_text = prompt.instruction

        query_vector = encode_text(
            self.encoder_model_name,
            text_to_encode=query_text,
            device=self.device,
        )
        ranked_list = retrieve_objects(
            query_vector, self.search_index_path, self.search_depth
        )
        for model_idx_str, model_score in ranked_list:
            model_idx = int(model_idx_str)
            self.model_infos[model_idx].score = model_score

        ranked_list = sorted(self.model_infos, key=lambda x: x.score, reverse=True)
        assert len(ranked_list) > 0, "No models retrieved from search index."
        return [model_tuple.name for model_tuple in ranked_list]
