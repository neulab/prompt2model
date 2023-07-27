"""An dual-encoder model retriever using HuggingFace model descriptions."""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from prompt2model.model_retriever.base import ModelRetriever
from prompt2model.model_retriever.generate_hypothetical_document import (
    generate_hypothetical_model_description,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects


class DescriptionModelRetriever(ModelRetriever):
    """Retrieve a model from among HuggingFace models."""

    def __init__(
        self,
        search_index_path: str,
        search_depth: int = 5,
        model_name: str = "OpenMatch/cocodr-base-msmarco",
        model_descriptions_index="huggingface_models/model_info/",
        openai_api_key: str | None = None,
    ):
        """Initialize a dual-encoder retriever against a search index."""
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.model_name = model_name
        self.model_descriptions_index = model_descriptions_index

        self.model_blocklist_organizations = ["huggingtweets"]
        self.load_model_info()

        assert not os.path.isdir(search_index_path), (
            "Search index must either be a valid file or not exist yet; "
            + f"{search_index_path} provided."
        )
        self.openai_api_key = openai_api_key

    def load_model_info(self):
        """Load metadata (e.g. downloads, publication date) about various models."""
        self.model_side_info = {}
        description_files = os.listdir(self.model_descriptions_index)
        self.model_names = []
        self.model_descriptions = []
        self.model_metadata = {}
        for f in tqdm(description_files):
            if (
                f.startswith(".")
                or len(open(os.path.join(self.model_descriptions_index, f)).read()) == 0
            ):
                continue
            block = False
            for org in self.model_blocklist_organizations:
                if f.startswith(org + "/"):
                    block = True
                    break
            if block:
                continue
            model_dict = json.load(open(os.path.join(self.model_descriptions_index, f)))
            self.model_descriptions.append(model_dict["description"])
            model_name = model_dict["pretrained_model_name"]
            self.model_names.append(model_name)
            self.model_metadata[model_name] = model_dict

    def encode_model_descriptions(self, prompt: PromptSpec) -> np.ndarray:
        """Encode model descriptions into a vector for indexing."""
        model_vectors = encode_text(
            self.model_name,
            text_to_encode=self.model_descriptions,
            encoding_file=self.search_index_path,
            device=torch.device("cpu"),
        )
        return model_vectors

    def retrieve(
        self,
        prompt: PromptSpec,
        similarity_threshold: float = 0.5,
    ) -> list[str]:
        """Select a model from a prompt using a dual-encoder retriever.

        Args:
            prompt: A prompt to use to select relevant models.
            similarity_threshold: The minimum similarity score for retrieving a model.

        Return:
            A list of relevant models' HuggingFace names.
        """
        if not os.path.exists(self.search_index_path):
            raise ValueError(
                "Search index does not exist; encode model descriptions first."
            )

        query_text = prompt.instruction

        query_vector = encode_text(
            self.model_name,
            text_to_encode=query_text,
            device=torch.device("cpu"),
        )
        ranked_list = retrieve_objects(
            query_vector, self.search_index_path, self.search_depth
        )
        ranked_model_list = [
            (self.model_names[int(model_idx_str)], float(model_score))
            for (model_idx_str, model_score) in ranked_list
        ]
        ranked_model_list = sorted(ranked_model_list, key=lambda x: x[1], reverse=True)
        assert len(ranked_model_list) > 0, "No models retrieved from search index."
        return [model_tuple[0] for model_tuple in ranked_model_list]
