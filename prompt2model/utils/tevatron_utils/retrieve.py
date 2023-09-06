"""Tools for doing efficient similarity search via the Tevatron/faiss libraries."""
from __future__ import annotations

import pickle

import numpy as np
from tevatron.faiss_retriever import BaseFaissIPRetriever


def retrieve_objects(
    query_vector: np.ndarray,
    encoded_datasets_path: str,
    document_names: list[str],
    depth: int,
) -> list[tuple[str, float]]:
    """Return a ranked list of object indices and their scores.

    Args:
        query vector: Vector representation of query.
        encoded_datasets_path: Path to file containing encoded dataset index.
        depth: Number of documents to return.

    Returns:
        Ranked list of object names and their inner product similarity to the query.
    """
    if query_vector.shape[0] != 1:
        raise ValueError("Only a single query vector is expected.")
    if len(query_vector.shape) != 2:
        raise ValueError("Query vector must be 1-D.")

    with open(encoded_datasets_path, "rb") as f:
        passage_reps, passage_lookup = pickle.load(f)
    retriever = BaseFaissIPRetriever(passage_reps)
    retriever.add(passage_reps)

    all_scores, all_indices = retriever.search(query_vector, depth)
    if not (len(all_scores) == len(all_indices) == 1):
        raise ValueError("Only one query's ranking should be returned.")

    psg_scores = all_scores[0]
    ranked_document_names = [document_names[passage_lookup[x]] for x in all_indices[0]]
    score_tuples = list(zip(ranked_document_names, psg_scores))
    return score_tuples
