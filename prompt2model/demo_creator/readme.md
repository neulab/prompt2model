# DatasetGnerator Usage

## DatasetGenerator

The `DatasetGenerator` is an abstract class that serves as the base for
generating datasets. It provides a common interface and defines the necessary
methods for generating datasets from prompts.

To create a dataset using the `DatasetGenerator`, you need to implement the
following methods:

- `generate_dataset_split()`: Generates a dataset for a specific split (train,
validation, or test) based on a given prompt specification and the desired
number of examples. - `generate_dataset_dict()`: Generates multiple datasets
splits (train, validation, and test) at once based on a prompt specification and
a dictionary specifying the number of examples for each split.

The `DatasetGenerator` class can be subclassed to implement custom dataset
generation logic based on different API services or approaches.

To see an example of how to use `DatasetGenerator` and its subclasses, you can
refer to the unit tests in the
[dataset_generator_test.py](../../tests/dataset_generator_test.py) file.

## DatasetSplit

The `DatasetSplit` is an enumeration class that defines the different types of
dataset splits, including `TRAIN`, `VALIDATION`, and `TEST`. It provides a
convenient way to refer to different parts of the dataset during generation and
analysis.

You can use `DatasetSplit.TRAIN`, `DatasetSplit.VALIDATION`, and
`DatasetSplit.TEST` to specify the desired dataset split when generating
datasets using the `DatasetGenerator` or its concrete implementations.

## OpenAIDatasetGenerator

The `OpenAIDatasetGenerator` is a concrete implementation of the
`DatasetGenerator` that leverages OpenAI's GPT-3.5 API to generate datasets. It
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

- Use the `OpenAIInstructionParser` to parse the prompt and obtain the
instruction and examples:

```python
prompt_spec = OpenAIInstructionParser(task_type=TaskType.<task_type>)
prompt = "<your-prompt>"
prompt_spec.parse_from_prompt(prompt)
```

- Use the dataset generator to generate datasets:

```python
num_examples = 100
split = DatasetSplit.TRAIN
dataset = dataset_generator.generate_dataset_split(prompt_spec, num_examples, split)
```

The `generate_dataset_split()` method generates a dataset for the specified
split using the prompt specification and the desired number of examples.

You can also use the `generate_dataset_dict()` method to generate multiple
dataset splits (e.g., train, validation, and test) at once:

```python
num_examples = {
    DatasetSplit.TRAIN: 1000,
    DatasetSplit.VALIDATION: 100,
    DatasetSplit.TEST: 200
}
dataset_dict = dataset_generator.generate_dataset_dict(prompt_spec, num_examples)
```

The `generate_dataset_dict()` method returns a `DatasetDict` object that
contains the generated dataset splits.

Please refer to the unit tests and examples provided by the
`OpenAIDatasetGenerator` for detailed usage information.
