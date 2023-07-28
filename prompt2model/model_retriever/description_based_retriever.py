"""An dual-encoder model retriever using HuggingFace model descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os
from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm

from prompt2model.model_retriever.base import ModelRetriever
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects

ModelInfo = namedtuple("ModelInfo", ["name", "description"])
ModelScorePair = namedtuple("ModelScorePair", ["model", "score"])


class DescriptionModelRetriever(ModelRetriever):
    """Retrieve a model from among HuggingFace models."""

    def __init__(
        self,
        search_index_path: str,
        search_depth: int = 5,
        encoder_model_name: str = "OpenMatch/cocodr-base-msmarco",
        model_descriptions_index_path="huggingface_models/model_info/",
        device: torch.device = torch.device("cuda:0"),
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

        self.model_blocklist_organizations = ["huggingtweets"]
        self.load_model_info()

        assert not os.path.isdir(
            search_index_path
        ), f"Search index must either be a valid file or not exist yet. But {search_index_path} is provided."  # noqa 501

    def load_model_info(self):
        """Load metadata (e.g. downloads, publication date) about various models."""
        self.model_side_info = {}
        description_files = os.listdir(self.model_descriptions_index_path)
        self.models = []
        self.model_metadata = {}
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
            model_info = ModelInfo(model_name, model_dict["description"])
            self.models.append(model_info)
            self.model_metadata[model_name] = model_dict

    def encode_model_descriptions(self, search_index_path) -> np.ndarray:
        """Encode model descriptions into a vector for indexing."""
        model_descriptions = [model.description for model in self.models]
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
        ranked_model_list = [
            ModelScorePair(self.models[int(model_idx_str)], float(model_score))
            for (model_idx_str, model_score) in ranked_list
        ]
        ranked_model_list = sorted(
            ranked_model_list, key=lambda x: x.score, reverse=True
        )
        assert len(ranked_model_list) > 0, "No models retrieved from search index."
        return [model_tuple.model.name for model_tuple in ranked_model_list]