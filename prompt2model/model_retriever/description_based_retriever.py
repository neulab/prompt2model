"""An dual-encoder model retriever using HuggingFace model descriptions."""

import json
import os

import numpy as np
import torch

from prompt2model.model_retriever import ModelRetriever
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
    ):
        """Initialize a dual-encoder retriever against a search index."""
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.model_name = model_name
        self.model_descriptions_index = model_descriptions_index
        description_files = os.listdir(self.model_descriptions_index)
        self.model_names = []
        self.model_descriptions = []
        for f in description_files:
            model_dict = json.load(open(os.path.join(self.model_descriptions_index, f)))
            self.model_names.append(model_dict["pretrained_model_name"])
            self.model_descriptions.append(model_dict["description"])
        assert not os.path.isdir(search_index_path), (
            "Search index must either be a valid file or not exist yet; "
            + f"{search_index_path} provided."
        )

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
        default_model: str = "t5-base",
    ) -> str:
        """Select a model from a prompt using a dual-encoder retriever.

        Args:
            prompt: A prompt to use to select relevant models.
            similarity_threshold: The minimum similarity score for retrieving a model.
            default_model: The default model to use if no model meets the similarity
                           threshold.


        Return:
            A relevant model's HuggingFace name.
        """
        if not os.path.exists(self.search_index_path):
            raise ValueError(
                "Search index does not exist; encode model descriptions first."
            )
        query_vector = encode_text(
            self.model_name,
            text_to_encode=prompt.instruction,
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
        top_model_similarity = ranked_model_list[0][1]
        if top_model_similarity >= similarity_threshold:
            top_model_name = ranked_model_list[0][0]
            return top_model_name
        else:
            return default_model
