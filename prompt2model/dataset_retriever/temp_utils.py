import aiohttp
import json
import os
import pickle
import zipfile
import huggingface_hub

from wrapt_timeout_decorator import *

from datasets import get_dataset_config_names, load_dataset_builder

@timeout(5)
def get_dataset_info(dataset_name):
    configs = get_dataset_config_names(dataset_name)
    config_to_description = {}
    for config_name in configs:
        dataset_info = load_dataset_builder(dataset_name, config_name)
        dataset_description = dataset_info.info
        config_to_description[config_name] = dataset_description
    return config_to_description

def store_dataset_metadata(dataset):
    dataset_configs_directory = "/tmp/dataset_configs"
    dataset_name_sanitized = dataset.replace("/", "_")
    dataset_info_file = os.path.join(dataset_configs_directory, dataset_name_sanitized + ".pkl")
    if os.path.exists(dataset_info_file):
        single_dataset_infos = pickle.load(open(dataset_info_file, "rb"))
    else:
        try:
            single_dataset_infos = get_dataset_info(dataset)
        except (FileNotFoundError,
                TypeError,
                ImportError,
                aiohttp.client_exceptions.ClientOSError,
                json.decoder.JSONDecodeError,
                zipfile.BadZipFile,
                huggingface_hub.utils._errors.HfHubHTTPError) as e:
            print(f"Skipping {dataset} due to error: {e}")
            single_dataset_infos = {}
        pickle.dump(single_dataset_infos, open(dataset_info_file, "wb"))