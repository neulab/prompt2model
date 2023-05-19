"""A script for encoding and serializing a search index using a contextual encoder."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import numpy as np
import os
import pickle
import tempfile
import torch

from torch.utils.data import DataLoader
from tevatron.arguments import DataArguments
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.datasets import HFQueryDataset, HFCorpusDataset
from tevatron.modeling import DenseOutput, DenseModelForInference
from tevatron.faiss_retriever import BaseFaissIPRetriever
from transformers import AutoConfig, AutoTokenizer

from prompt2model.prompt_parser import PromptSpec

parser = argparse.ArgumentParser()
parser.add_argument("--model-name-or-path", type=str, default="bert-base-uncased")
parser.add_argument("--config-name", type=str, default=None)
parser.add_argument("--num-shards", type=int, default=1)



def encode_text(model_name_or_path: str,
                encoding_file: str | None = None,
                file_to_encode: str | None = None,
                text_to_encode: list[str] | str | None = None,
                config_name: str | None = None,
                tokenizer_name: str | None = None,
                encode_query: bool = False,
                max_len: int = 128,
                device: torch.device = torch.device("cpu"),
                dataloader_num_workers: int = 0,
                model_cache_dir: str | None = None,
                data_cache_dir: str = "~/.cache/huggingface/datasets",
                per_device_eval_batch_size = 8,
                fp16: bool = False):
    """Encode a query or documents.
    
    Args:
        model_name_or_path (str): The HuggingFace model name or path to the model.
        cache_dir (str): The directory to cache the model.
        encoding_directory (str): The path to store the encoded data.
        encode_in_path (str): The path to save the encoded text.
        config_name (str | None): The HuggingFace model name or path to the config.
                                  If left empty, we will use `model_name_or_path`.
        tokenizer_name (str | None): The HuggingFace model name or path to the
                                     tokenizer locally. If left empty, we will use
                                    `model_name_or_path`.
        encode_query (bool): Whether to encode queries or documents.
        dataloader_num_workers (int): 
    """
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        # num_labels=num_labels,
        cache_dir=model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        cache_dir=model_cache_dir,
        use_fast=False,
    )
    model = DenseModelForInference.build(
        model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=model_cache_dir,
    )

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
            temporary_file = tempfile.NamedTemporaryFile(mode = "w", delete=False, suffix=".json")
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
            hf_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                            cache_dir=data_args.data_cache_dir or model_cache_dir)
        else:
            data_args.p_max_len = max_len
            hf_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_cache_dir)

        encode_dataset = EncodeDataset(hf_dataset.process(1, 0),
                                        tokenizer,
                                        max_len=max_len)

        encode_loader = DataLoader(
            encode_dataset,
            batch_size=per_device_eval_batch_size,
            collate_fn=EncodeCollator(
                tokenizer,
                max_length=max_len,
                padding='max_length'
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
                        model_output: DenseOutput = model(query=batch)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: DenseOutput = model(passage=batch)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        if encoding_file:
            with open(encoding_file, 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)

        return encoded
    finally:
        if using_temp_file and os.path.exists(file_to_encode):
            os.unlink(file_to_encode)

def encode_query(query: str):
    raise NotImplementedError

def encode_search_corpus():
    raise NotImplementedError

def retrieve_objects(query: PromptSpec) -> list[tuple[str, int]]:
    """Return a ranked list of model names and their scores."""
    raise NotImplementedError