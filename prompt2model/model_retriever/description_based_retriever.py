"""An dual-encoder model retriever using HuggingFace model descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os

import numpy as np
import retriv
import torch
from tqdm import tqdm

from prompt2model.model_retriever.base import ModelRetriever
from prompt2model.model_retriever.generate_hypothetical_document import (
    generate_hypothetical_model_description,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects


class ModelInfo:
    """Store the model name, description, and query-model score for each model."""

    def __init__(
        self,
        name: str,
        description: str,
        score: float,
        size_in_bytes: int,
        num_downloads: int,
    ):
        """Initialize a ModelInfo object.

        Args:
            name: The name of the model.
            description: The description of the model.
            score: The similarity of the model to a given prompt from a user.
            size_in_bytes: The size of the model on disk, in bytes.
            num_downloads: The number of downoads for this model on HuggingFace.
        """
        self.name = name
        self.description = description
        self.score = score
        self.size_in_bytes = size_in_bytes
        self.num_downloads = num_downloads


class DescriptionModelRetriever(ModelRetriever):
    """Retrieve a model from among HuggingFace models."""

    def __init__(
        self,
        search_index_path: str,
        search_depth: int = 5,
        first_stage_depth: int = 1000,
        encoder_model_name: str = "OpenMatch/cocodr-base-msmarco",
        model_descriptions_index_path="huggingface_models/model_info/",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        model_size_limit_bytes=3e9,
        use_bm25: bool = True,
        use_HyDE: bool = False,
        openai_api_key: str | None = None,
    ):
        """Initialize a dual-encoder retriever against a search index.

        Args:
            search_index_path: Where to store the search index (e.g. encoded vectors).
            search_depth: The number of most-relevant models to retrieve.
            first_stage_depth: The number of models to retrieve purely by similarity,
                before reranking by scaling with model size and number of downloads.
            encoder_model_name: The name of the model to use for the dual-encoder.
            model_descriptions_index_path: The directory of models to search against.
            device: The device to use for encoding text for our dual-encoder model.
            model_size_limit_bytes: The maximum size (in bytes) of a model to retrieve.
            use_bm25: Whether to use BM25 to retrieve the top-k models. If False, we
                use a dual-encoder retriever.
            use_HyDE: Whether to use HyDE to replace the query with a hypothetical
                model description. generated by an LLM.
            openai_api_key: OpenAI API key. If None, use the OPENAI_API_KEY environment
                variable.
        """
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.first_stage_depth = first_stage_depth
        self.encoder_model_name = encoder_model_name
        self.model_descriptions_index_path = model_descriptions_index_path
        self.device = device
        self.model_size_limit_bytes = model_size_limit_bytes
        # If use_bm25 is True, then we use BM25 to retrieve the top-k models.
        # Otherwise, we use a dual-encoder retriever.
        self.use_bm25 = use_bm25
        self.use_HyDE = use_HyDE
        self.openai_api_key = openai_api_key

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
                name=model_name,
                description=model_dict["description"],
                score=None,
                size_in_bytes=model_dict["size_bytes"],
                num_downloads=model_dict["downloads"],
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

    def scaled_similarity_score(
        self, model_info: ModelInfo, model_score: float
    ) -> float:
        """Adjust the search score using the model size and number of downloads.

        Args:
            model_info: The name of the model we are scoring.
            model_score: The similarity score of this model for this particular query.
        """
        num_downloads = int(model_info.num_downloads)
        log_num_downloads = np.log10(num_downloads + 1)
        model_size_bytes = int(model_info.size_in_bytes)
        if model_size_bytes > self.model_size_limit_bytes or model_size_bytes == 0:
            return -np.inf
        return model_score * log_num_downloads

    def construct_bm25_index(self):
        """Construct a retriv BM25 index for model descriptions."""
        collection = []
        for model in self.model_infos:
            collection.append({"id": model.name, "text": model.description})
        search_engine = retriv.SearchEngine("new-index").index(collection)
        return search_engine

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

        if self.use_HyDE:
            query_text = generate_hypothetical_model_description(
                prompt, self.openai_api_key
            )
        else:
            query_text = prompt.instruction

        if self.use_bm25:
            search_engine = self.construct_bm25_index()
            results = search_engine.search(query_text, cutoff=self.first_stage_depth)
            ranked_list: list[tuple[str, float]] = []
            for result in results:
                ranked_list.append((result["id"], result["score"]))
        else:
            query_vector = encode_text(
                self.encoder_model_name,
                text_to_encode=query_text,
                device=self.device,
            )
            model_names = [model.name for model in self.model_infos]
            ranked_list = retrieve_objects(
                query_vector,
                self.search_index_path,
                model_names,
                self.first_stage_depth,
            )

        model_name_to_model_info = {}
        for model_info in self.model_infos:
            model_name_to_model_info[model_info.name] = model_info

        top_models_list = []
        for model_name, model_score in ranked_list:
            model_info = model_name_to_model_info[model_name]
            scaled_model_score = self.scaled_similarity_score(model_info, model_score)
            model_info.score = scaled_model_score
            top_models_list.append(model_info)

        top_models_list = sorted(top_models_list, key=lambda x: x.score, reverse=True)
        assert len(top_models_list) > 0, "No models retrieved from search index."
        return [model_info.name for model_info in top_models_list]
