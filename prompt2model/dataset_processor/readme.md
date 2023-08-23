# Dataset Processor

## Overview

- `BaseProcessor`: The base class for dataset post-processing.
- `TextualizeProcessor`: Transforms all datasets into a consistent
text-to-text generation format.

## Getting Started

- **Import the Module**:

```python
from prompt2model.dataset_processor.textualize import TextualizeProcessor
```

- **Initialize TextualizeProcessor**:

```python
processor = TextualizeProcessor(has_encoder=<True/False>)
# <True/False>: Whether the model you want to finetune has an encoder.
```

Choose encoder type:

- `has_encoder=True` for encoder-decoder models (e.g., T5).
- `has_encoder=False` for decoder-only/autoregressive models (e.g., GPT2).

- **Process Datasets**:

```python
instruction = "<your-instruction>"
dataset_dicts = [...]  # List of DatasetDict
modified_dataset_dicts = processor.process_dataset_dict(instruction, dataset_dicts)
```
