"""Tools for creating a mock search index."""

import pickle

import numpy as np


def create_test_search_index(index_file_name: str) -> None:
    """Utility function to create a test search index.

    This search index represents 3 models, each represented with a hand-written vector.
    Given a query of [0, 0, 1], the 3rd model will be the most similar.
    """
    mock_model_encodings = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]])
    mock_lookup_indices = [0, 1, 2]
    with open(index_file_name, "wb") as f:
        pickle.dump((mock_model_encodings, mock_lookup_indices), f)


def create_test_search_index_class_method(self, index_file_name):
    """Utility function to create a test search index as a simulated class method."""
    _ = self
    create_test_search_index(index_file_name)
