# DatasetGenerator Usage

## DatasetGenerator

The `DatasetGenerator` is an abstract class that provides a standard interface and
necessary methods for generating datasets based on prompts.

To create a dataset with the `DatasetGenerator`, you need to implement the
following methods:

- `generate_dataset_split()`: Generates a dataset for a specific split (train,
validation, or test) based on a given prompt specification and the desired
number of examples.
- `generate_dataset_dict()`: Generates multiple datasets
splits (train, validation, and test) at once.

We already use the latest version of `gpt-3.5-turbo` from OpenAI
in [openai_tools.py](../utils/openai_tools.py).
Feel free to subclass the `DatasetGenerator` class to implement
a custom dataset generation logic based on different API services
or approaches.

## DatasetSplit

The `DatasetSplit` is an enumeration class that defines the different types of
dataset splits, including `TRAIN`, `VALIDATION`, and `TEST`. It provides a
standard way to refer to other dataset parts during generation and
analysis.

You can use `DatasetSplit.TRAIN`, `DatasetSplit.VALIDATION`, and
`DatasetSplit.TEST` to specify the desired dataset split when generating
datasets using the `DatasetGenerator` or its concrete implementations.

## OpenAIDatasetGenerator

The `OpenAIDatasetGenerator` is a concrete implementation of the
`DatasetGenerator` leverages OpenAI's GPT-3.5 API to generate datasets. It
enables the generation of datasets by providing a prompt specification and the
desired number of examples per split.

## Usage

- Import the necessary modules:

```python
from prompt2model.dataset_generator import OpenAIDatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
```

- Initialize an instance of the `OpenAIDatasetGenerator` with your OpenAI API
key:

```python
api_key = "<your-api-key>"
dataset_generator = OpenAIDatasetGenerator(api_key)
```

- You can also set the OPENAI_API_KEY environment variable by:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

- Use the `OpenAIInstructionParser` to parse the prompt and obtain the
instruction and examples:

```python
prompt_spec = OpenAIInstructionParser(task_type=TaskType.<task_type>)
prompt = "<your-prompt>"
prompt_spec.parse_from_prompt(prompt)
```

- Use the dataset generator to generate datasets:

```python
expected_num_examples = 100
split = DatasetSplit.TRAIN
dataset = dataset_generator.generate_dataset_split(
    prompt_spec, expected_num_examples, split
    )
```

The `generate_dataset_split()` method generates a dataset for the specified
split using the `prompt_spec` and the desired number of examples.

You can also use the `generate_dataset_dict()` method to generate multiple
dataset splits (e.g., train, validation, and test) at once:

```python
expected_num_examples = {
    DatasetSplit.TRAIN: 1000,
    DatasetSplit.VALIDATION: 100,
    DatasetSplit.TEST: 200
}
dataset_dict = dataset_generator.generate_dataset_dict(
    prompt_spec, expected_num_examples
    )
```

The `generate_dataset_dict()` method returns a `DatasetDict` object that
contains the generated dataset splits.
