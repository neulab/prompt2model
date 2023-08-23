# Dataset Generator

## Overview

- `DatasetGenerator`: An abstract class to generate datasets.
- `DatasetSplit`: An enumeration class defining dataset types (`TRAIN`,
`VALIDATION`, `TEST`).
- `OpenAIDatasetGenerator`: A concrete class
for dataset generation using GPT-3.5 API.

## Getting Started

- **Import the Modules**:

```python
from prompt2model.dataset_generator import OpenAIDatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
```

- **Setup OpenAI API Key**:

```python
api_key = "<your-api-key>"
dataset_generator = OpenAIDatasetGenerator(api_key)
```

Or, set as an environment variable:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

- **Parse the Prompt**:

```python
prompt_spec = OpenAIInstructionParser(task_type=TaskType.<task_type>)
# Refer the document string of DatasetSplit for more details.
prompt = "<your-prompt>"
prompt_spec.parse_from_prompt(prompt)
```

Or you can mock a `PromptSpec` object, as
shown in [PromptParser](./../prompt_parser/readme.md).

**Generate Dataset**:

For a specific split:

```python
expected_num_examples = 100
split = DatasetSplit.TRAIN
dataset = dataset_generator.generate_dataset_split(
    prompt_spec
    , expected_num_examples
    , split
    )
```

Or, for multiple splits:

```python
expected_num_examples = {
    DatasetSplit.TRAIN: 1000,
    DatasetSplit.VALIDATION: 100,
    DatasetSplit.TEST: 200
}
dataset_dict = dataset_generator.generate_dataset_dict(prompt_spec, expected_num_examples)
```
