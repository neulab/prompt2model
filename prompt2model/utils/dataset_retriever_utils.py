from __future__ import annotations  # noqa FI58

import json
import threading
import time
from collections.abc import MutableMapping
from typing import Any

import requests


def get_dataset_validity(dataset_name, max_retries=5):
    """Get the list of loadable datasets from HuggingFace with backoff strategy."""
    API_URL = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    retries = 0
    backoff = 10

    while retries < max_retries:
        response = requests.get(API_URL)

        if response.status_code == 200:
            response = response.json()
            return (
                "preview" in response
                and "viewer" in response
                and response["preview"] & response["viewer"]
            )

        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else backoff
            time.sleep(wait)
            backoff *= 2  # Exponential increase
            retries += 1
        else:
            # Handle other HTTP errors
            break

    return False


def replace_duplicate_columns(original_dataset_columns):
    """Utility function to remove duplicate columns, after flattening dataset."""
    columns_mapping: dict[str, str] = {}
    new_columns = []
    counter: dict[str, int] = {}
    # convert flattened columns like answer.text -> answer_text
    for col in original_dataset_columns:
        new_col = col.replace(".", "_")
        if new_col in columns_mapping.values():
            counter[new_col] = counter.get(new_col, 0) + 1
            new_col = f"{new_col}_{counter[new_col]}"
        columns_mapping[col] = new_col
        new_columns.append(new_col)
    return new_columns, columns_mapping


def fetch_first_row_with_timeout(dataset, timeout=30):
    """Don't load dataset if it takes more than 30s."""

    def fetch_sample_row(container):
        try:
            container.append(next(iter(dataset)))
        except Exception as e:
            container.append(e)

    result_container = []
    fetch_thread = threading.Thread(target=fetch_sample_row, args=(result_container,))
    fetch_thread.start()
    fetch_thread.join(timeout)

    if fetch_thread.is_alive():
        # Operation took too long
        return None

    return result_container[0]


def truncate_row(example_row: dict, max_length=50) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """Utility function to flatten Streaming HuggingFace datasets."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
