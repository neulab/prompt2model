# Dataset Retriever

## Overview

- `DatasetRetriever`: Interface for retrieving datasets based on a
prompt.
- `DescriptionDatasetRetriever`: Retrieves HuggingFace datasets using
similarity to a given prompt.

## Getting Started

- Import Modules

```python
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType
```

- Initialize Retriever

```python
retriever = DescriptionDatasetRetriever()
```

Various parameters like search index path, model name, and search
depth can be customized during initialization.

- Prepare the Prompt

```python
task_type = TaskType.TEXT_GENERATION
prompt_text = "..."
prompt_spec = MockPromptSpec(task_type)
prompt_spec._instruction = prompt_text
```

- Retrieve Dataset

```python
dataset_dict = retriever.retrieve_dataset_dict(
    prompt_spec, blocklist=[]
)
```

`dataset_dict` will contain the dataset splits (train/val/test) most
relevant to the given prompt.
