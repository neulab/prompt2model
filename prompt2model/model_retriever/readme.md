# Model Retriever

## Overview

- `ModelRetriever`: An interface for retrieving several models from
HuggingFace.
- `DescriptionModelRetriever` offers functions to vectorize
descriptions, adjust relevance scores, build a BM25 index, and
retrieve models based on a prompt.
- `ModelInfo` stores a model's HuggingFace name, description,
identifier, relevance score, disk size, and download count.

## Getting Started

- **Import Required Modules**:

```python
from prompt2model.model_retriever import DescriptionModelRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType
```

- **Initialize the Prompt**:

```python
prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
prompt = "..."
prompt_spec._instruction = prompt
```

- **Initialize and Run the Retriever**:

```python
retriever = DescriptionModelRetriever(
    search_index_path="path_to_bm25_search_index.pkl",
    model_descriptions_index_path="path_to_model_info_directory",
    use_bm25=True,
    use_HyDE=True,
)
top_model_name = retriever.retrieve(prompt_spec)
```

## Notes

- Ensure that the paths provided to the retriever (e.g.,
`search_index_path` and `model_descriptions_index_path`) point to the
correct locations.
