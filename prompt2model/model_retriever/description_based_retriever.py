"""An dual-encoder model retriever using HuggingFace model descriptions."""

import json
import os

import numpy as np
import pickle
import torch
from tqdm import tqdm
import yaml

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
        first_stage_depth: int = 20000,
        model_name: str = "OpenMatch/cocodr-base-msmarco",
        model_descriptions_index="huggingface_models/model_info/",
        parameter_limit=8e8,
        use_HyDE: bool = False,
        openai_api_key: str | None = None,
    ):
        """Initialize a dual-encoder retriever against a search index."""
        self.search_index_path = search_index_path
        self.search_depth = search_depth
        self.first_stage_depth = first_stage_depth
        self.model_name = model_name
        self.model_descriptions_index = model_descriptions_index
        self.parameter_limit = parameter_limit
        self.model_blocklist_organizations = ["huggingtweets"]
        self.load_model_info()

        assert not os.path.isdir(search_index_path), (
            "Search index must either be a valid file or not exist yet; "
            + f"{search_index_path} provided."
        )
        self.use_HyDE = use_HyDE
        self.openai_api_key = openai_api_key

    def estimate_number_of_parameters(self, model_size_bytes: str) -> int:
        return int(model_size_bytes) / 4

    @staticmethod
    def load_model_metadata(model_description: str):
        if len(model_description.split("---")) < 2:
            return ""
        metadata_segment = model_description.split("---")[1].strip()
        metadata_dict = yaml.load(metadata_segment, Loader=yaml.Loader)
        if metadata_dict is None:
            return ""
        if "model-index" in metadata_dict:
            for i, model in enumerate(metadata_dict["model-index"]):
                if "results" in model:
                    for k, task in enumerate(model["results"]):
                        if "metrics" in task:
                            del metadata_dict["model-index"][i]["results"][k]["metrics"]
        return yaml.dump(metadata_dict)

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
            try:
                model_metadata = self.load_model_metadata(model_dict["description"])
            except:
                model_metadata = ""

            model_name = model_dict["pretrained_model_name"]
            self.model_names.append(model_name)
            self.model_descriptions.append(model_metadata)
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

    def scaled_similarity_score(
        self, similarity_score: float, model_name: str
    ) -> float:
        model_size = int(self.model_metadata[model_name]["size_bytes"])
        # Do not upweight models that are smaller than 400MB.
        # Models that are larger than 400MB will be downweighted proportionally
        # to their size.
        estimated_num_parameters = self.estimate_number_of_parameters(
            self.model_metadata[model_name]["size_bytes"]
        )
        # floored_model_size = max(model_size, BASELINE_MODEL_SIZE) / BASELINE_MODEL_SIZE
        num_downloads = int(self.model_metadata[model_name]["downloads"])
        log_num_downloads = np.log(num_downloads + 1)
        if estimated_num_parameters > self.parameter_limit or model_size == 0:
            return -np.inf, similarity_score, log_num_downloads, estimated_num_parameters
        return similarity_score * log_num_downloads, similarity_score, log_num_downloads, estimated_num_parameters

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

        if self.use_HyDE:
            hyde_cache_file = "/tmp/hyde_cache.pkl"
            if os.path.exists(hyde_cache_file):
                hyde_cache = pickle.load(open(hyde_cache_file, "rb"))
            else:
                hyde_cache = {}
            if prompt.instruction in hyde_cache:
                hypothetical_document = hyde_cache[prompt.instruction]
            else:
                hypothetical_document = generate_hypothetical_model_description(
                    prompt, self.openai_api_key
                )
                hyde_cache[prompt.instruction] = hypothetical_document
            pickle.dump(hyde_cache, open(hyde_cache_file, "wb"))
            print(f"Generated query vector: {hypothetical_document}")
            query_text = self.load_model_metadata(hypothetical_document)

        else:
            query_text = prompt.instruction


        query_vector = encode_text(
            self.model_name,
            text_to_encode=query_text,
            device=torch.device("cpu"),
        )
        ranked_list = retrieve_objects(
            query_vector, self.search_index_path, self.first_stage_depth
        )
        ranked_model_list = [
            (self.model_names[int(model_idx_str)], float(model_score))
            for (model_idx_str, model_score) in ranked_list
        ]
        ranked_model_list_scores_scaled = [
            (name, self.scaled_similarity_score(score, name))
            for (name, score) in ranked_model_list
        ]
        ranked_model_list_scores_scaled = sorted(
            ranked_model_list_scores_scaled, key=lambda x: x[1][0], reverse=True
        )
        assert (
            len(ranked_model_list_scores_scaled) > 0
        ), "No models retrieved from search index."
        top_model_similarity = ranked_model_list_scores_scaled[0][1][0]
        if top_model_similarity >= similarity_threshold:
            return [model_tuple[0] for model_tuple in ranked_model_list_scores_scaled[:self.search_depth]]
        else:
            return [default_model]
