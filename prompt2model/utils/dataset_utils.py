"""Util functions for datasets."""
import requests


def query(API_URL):
    """Returns a response json for a URL."""
    response = requests.get(API_URL)
    return response.json()


def get_dataset_size(dataset_name):
    """Fetches dataset size for a dataset from huggingface API."""
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={dataset_name}"
    data = query(API_URL)
    return "{:.2f}".format(data["size"]["dataset"]["num_bytes_memory"] / 1024 / 1024)
