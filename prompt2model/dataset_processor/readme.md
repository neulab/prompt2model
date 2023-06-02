# DatasetProcessor Usage

## BaseProcessor

The `BaseProcessor` is a base class for dataset processors. It provides a common
interface and defines the necessary methods for post-processing datasets.

To create a dataset processor using the `BaseProcessor`, you need to implement
the following method:

- `post_process_example()`: Modifies the input column of a given example
dictionary based on the task-specific requirements.

The `BaseProcessor` class can be subclassed to implement custom dataset
processing logic based on different task requirements or data formats.

To see an example of how to use `BaseProcessor` and its subclasses, you can
refer to the unit tests in the
[dataset_processor_test.py](../../tests/dataset_processor_test.py) file.

## TextualizeProcessor

The `TextualizeProcessor` is a dataset processor that converts datasets into a
Text2Text fashion. It modifies the input column of each example in the dataset
to include task-specific instructions and prefixes.

## Usage

- Import the necessary modules:

```python
from prompt2model.dataset_processor.textualize import TextualizeProcessor
```

- Initialize an instance of the `TextualizeProcessor`:

```python
processor = TextualizeProcessor(has_encoder=<True / False>)
```

The `has_encoder` parameter indicates whether the retrieved model has an
encoder. For encoder-decoder models like T5, set `has_encoder=True`. For
decoder-only models like GPT, set `has_encoder=False`.

- Use the dataset processor to process dataset dictionaries:

```python
instruction = "<your-instruction>"
dataset_dicts = [...]  # List of DatasetDicts
modified_dataset_dicts = processor.process_dataset_dict(instruction, dataset_dicts)
```

The `process_dataset_dict()` method modifies the input column of each example in
the dataset by adding task-specific instructions and prefixes. It returns a list
of modified `DatasetDicts`, where all examples are converted into a text-to-text
fashion.

Feel free to adjust the code and configuration based on your specific
requirements and the structure of your datasets.
