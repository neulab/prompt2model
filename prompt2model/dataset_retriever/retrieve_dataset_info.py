"""Tools for retrieving dataset metadata from HuggingFace."""

import aiohttp
import errno
import functools
import importlib
import json
import os
import pickle
import requests
import signal
from tqdm import tqdm
from multiprocessing import Pool
import zipfile
import huggingface_hub

from prompt2model.dataset_retriever.temp_utils import store_dataset_metadata


from datasets import get_dataset_config_names, load_dataset_builder

hf_eval_utils = importlib.import_module("model-evaluator.utils")

def get_dataset_names():
    API_URL = "https://datasets-server.huggingface.co/valid"
    response = requests.get(API_URL)
    datasets_list = response.json()
    fully_supported_datasets = datasets_list["viewer"]
    return fully_supported_datasets




def get_dataset_columns(dataset):
    dataset_metadata = hf_eval_utils.get_metadata(dataset, "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")
    # 
    '''
    hf_eval_utils.get_metadata("acronym_identification", "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")
    
    [{  'config': 'default',
        'task': 'token-classification',
        'task_id': 'entity_extraction',
        'splits': {'eval_split': 'test'},
        'col_mapping': {'tokens': 'tokens', 'labels': 'tags'}}]
    '''
    if dataset_metadata == None:
        return None
    else:
        return dataset_metadata


SUPPORTED_TASKS = ['text-classification', 'text2text-generation', 'question-answering', 'summarization', 'text-generation', 'token-classification']

def load_dataset_metadata(dataset_names, dataset_metadata_cache_file = "/tmp/dataset_metadata_cache.pkl"):
    if os.path.exists(dataset_metadata_cache_file):
        dataset_metadata_cache = pickle.load(open(dataset_metadata_cache_file, "rb"))
    else:
        dataset_metadata_cache = {}
    all_dataset_metadata = {}
    for dataset in tqdm(dataset_names):
        if dataset in dataset_metadata_cache:
            dataset_metadata = dataset_metadata_cache[dataset]
        else:
            try:
                dataset_metadata = get_dataset_columns(dataset)
            except:
                dataset_metadata = None
        if dataset_metadata is not None:
            filtered_task_metadata = []
            for task_metadata in dataset_metadata:
                if "task" in task_metadata and task_metadata["task"] in SUPPORTED_TASKS:
                    filtered_task_metadata.append(task_metadata)
            if len(filtered_task_metadata) > 0:
                all_dataset_metadata[dataset] = filtered_task_metadata
        dataset_metadata_cache[dataset] = dataset_metadata

    pickle.dump(dataset_metadata_cache, open(dataset_metadata_cache_file, "wb"))
    return all_dataset_metadata

if __name__ == "__main__":
    dataset_names = get_dataset_names()
    squad_idx = [i for i, x in enumerate(dataset_names) if x == "squad"][0]
    dataset_names = ["squad"] + dataset_names[:squad_idx] + dataset_names[squad_idx+1:]
    all_dataset_metadata = load_dataset_metadata(dataset_names)

    dataset_configs_directory = "/tmp/dataset_configs"
    os.makedirs(dataset_configs_directory, exist_ok=True)
    dataset_config_descriptions = {}

    with Pool(6) as p:
        p.map(store_dataset_metadata, dataset_names)

    '''
    for dataset in tqdm(dataset_names):
        dataset_name_sanitized = dataset.replace("/", "_")
        dataset_info_file = os.path.join(dataset_configs_directory, dataset_name_sanitized + ".pkl")
        if os.path.exists(dataset_info_file):
            single_dataset_infos = pickle.load(open(dataset_info_file, "rb"))
        else:
            try:
                single_dataset_infos = get_dataset_info(dataset)
            except (FileNotFoundError, TypeError, ImportError, aiohttp.client_exceptions.ClientOSError) as e:
                print(f"Skipping {dataset} due to error: {e}")
                single_dataset_infos = {}
            pickle.dump(single_dataset_infos, open(dataset_info_file, "wb"))
        for config, dataset_info in single_dataset_infos.items():
            dataset_config_descriptions[(dataset, config)] = dataset_info
    '''
            
    breakpoint()
