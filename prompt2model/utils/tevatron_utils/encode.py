"""Tools ffor encoding and serializing a search index with a contextual encoder."""

from __future__ import annotations  # noqa FI58

import argparse
import glob
import json
import os
import pickle
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
from tevatron.arguments import DataArguments
from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.datasets import HFCorpusDataset, HFQueryDataset
from tevatron.faiss_retriever import BaseFaissIPRetriever
from tevatron.modeling import DenseModelForInference, DenseOutput
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from prompt2model.prompt_parser import PromptSpec

parser = argparse.ArgumentParser()
parser.add_argument("--model-name-or-path", type=str, default="bert-base-uncased")
parser.add_argument("--config-name", type=str, default=None)
parser.add_argument("--num-shards", type=int, default=1)


def load_tevatron_model(model_name_or_path: str, model_cache_dir: str | None = None) -> tuple[DenseModelForInference, PreTrainedTokenizerBase]:
    """Load a Tevatron model from a model name/path.

    Args:
        model_name_or_path: The HuggingFace model name or path to the model.
        model_cache_dir: The directory to cache the model.

    Returns:
        A Tevatron model for dense retrieval.
    """

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        use_fast=False,
    )
    model = DenseModelForInference.build(
        model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=model_cache_dir,
    )
    return model, tokenizer

def encode_text(
    model_name_or_path: str,
    file_to_encode: str | None = None,
    text_to_encode: list[str] | str | None = None,
    encode_query: bool = False,
    encoding_file: str | None = None,
    max_len: int = 128,
    device: torch.device = torch.device("cpu"),
    dataloader_num_workers: int = 0,
    model_cache_dir: str | None = None,
    data_cache_dir: str = "~/.cache/huggingface/datasets",
    batch_size=8,
    fp16: bool = False,
) -> np.ndarray:
    """Encode a query or documents.
    This code is largely duplicated from tevatron/driver/encode.py in the Tevatron
    repository.

    Args:
        model_name_or_path: The HuggingFace model name or path to the model.
        file_to_encode: JSON or JSONL file containing `"text"` fields to encode.
        text_to_encode: String or list of strings to encode.
        encode_query: Whether or not we are encoding a query or documents.
        encoding_file: If given, store the encoded data in this file.
        max_len: Truncate the input to this length (in tokens).
        device: Device that Torch will use to encode the text.
        dataloader_num_workers: Number of workers to use for the dataloader.
        model_cache_dir: The directory to cache the model.
        data_cache_dir: The directory to cache the tokenized dataset.
        batch_size: Batch size to use for encoding.
        fp16: Whether or not to run inference in fp16 for more-efficient encoding.

    Returns:
        A numpy array of shape `(num_examples, embedding_dim)` containing text
        encoded by the specified model.
    """
    model, tokenizer = load_tevatron_model(model_name_or_path, model_cache_dir)

    if file_to_encode is None and text_to_encode is None:
        raise ValueError("Must provide either a dataset file or text to encode.")
    if file_to_encode is not None and text_to_encode is not None:
        raise ValueError("Provide either dataset file or text to encode, not both.")

    using_temp_file = False
    try:
        if text_to_encode is not None:
            using_temp_file = True
            if isinstance(text_to_encode, str):
                text_to_encode = [text_to_encode]
            temporary_file = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            )
            text_rows = []
            for i, text in enumerate(text_to_encode):
                text_rows.append({"text_id": i, "text": text})
            json.dump(text_rows, temporary_file)
            file_to_encode = temporary_file.name
            temporary_file.close()

        data_args = DataArguments(
            encoded_save_path=encoding_file,
            encode_in_path=file_to_encode,
            encode_is_qry=encode_query,
            data_cache_dir=data_cache_dir,
        )
        if encode_query:
            data_args.q_max_len = max_len
            hf_dataset = HFQueryDataset(
                tokenizer=tokenizer,
                data_args=data_args,
                cache_dir=data_args.data_cache_dir or model_cache_dir,
            )
        else:
            data_args.p_max_len = max_len
            hf_dataset = HFCorpusDataset(
                tokenizer=tokenizer,
                data_args=data_args,
                cache_dir=data_args.data_cache_dir or model_cache_dir,
            )

        encode_dataset = EncodeDataset(
            hf_dataset.process(1, 0), tokenizer, max_len=max_len
        )

        encode_loader = DataLoader(
            encode_dataset,
            batch_size=batch_size,
            collate_fn=EncodeCollator(
                tokenizer, max_length=max_len, padding="max_length"
            ),
            shuffle=False,
            drop_last=False,
            num_workers=dataloader_num_workers,
        )
        encoded = []
        lookup_indices = []
        model = model.to(device)
        model.eval()

        for (batch_ids, batch) in encode_loader:
            lookup_indices.extend(batch_ids)
            with torch.cuda.amp.autocast() if fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    if data_args.encode_is_qry:
                        model_output = model(query=batch)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output = model(passage=batch)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        if encoding_file:
            with open(encoding_file, "wb") as f:
                pickle.dump((encoded, lookup_indices), f)

        return encoded
    finally:
        if using_temp_file and file_to_encode and os.path.exists(file_to_encode):
            os.unlink(file_to_encode)


def encode_search_corpus(corpus: list[str], encoding_file_path: str, model_name_or_path: str):
    encoding_vectors = encode_text(model_name_or_path,
                                   text_to_encode=corpus,
                                   encoding_file=encoding_file_path)
    return encoding_vectors


def retrieve_objects(prompt: PromptSpec, model_name_or_path: str, encoded_datasets_path: str, depth: int) -> list[tuple[str, int]]:
    """Return a ranked list of model names and their scores.

    Args:
        prompt: Prompt provided by user as query to retrieve datasets.
        model_name_or_path: Model to encode query (should match dataset index encoder).
        encoded_datasets_path: Path to file containing encoded dataset index.
        depth: Number of documents to return
    
    Returns:
        Ranked list of model names and similarity scores (with respect to query).
    """
    text_query = prompt.get_instruction()
    query_vector = encode_text(model_name_or_path, text_to_encode=text_query)

    with open(encoded_datasets_path, 'rb') as f:
        passage_reps, passage_lookup = pickle.load(f)
    retriever = BaseFaissIPRetriever(passage_reps)
    
    all_scores, all_indices = retriever.search(query_vector, depth)
    psg_indices = [[str(passage_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)

    # Zip these two variables into a single list of tuples
    # all_scores, psg_indices
    return zip(all_scores, psg_indices)
