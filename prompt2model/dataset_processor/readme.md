# DatasetProcessor Usage

## BaseProcessor

The `BaseProcessor` is a foundational class for dataset processors, providing a
standard interface and defining essential methods for post-processing datasets
before passing into the `ModelTrainer`.

To create a dataset processor using the `BaseProcessor`, you need to subclass it
and implement the following method:

- `post_process_example()`: This method modifies the inputs of a given
example dictionary based on specific requirements.

The `BaseProcessor` class can be subclassed to implement a custom dataset
processing logic based on different task requirements or data formats.

Refer to the unit tests in the
[dataset_processor_test.py](../../tests/dataset_processor_test.py) file for
examples of how to use `BaseProcessor` and its subclasses.

## TextualizeProcessor

The `TextualizeProcessor` is a dataset processor that transforms
datasets into a conditional generation format. It modifies the
input of each example in the dataset to include task-specific
instructions, prefixes, and eos_token if necessary.

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
decoder-only models like GPT2, set `has_encoder=False`.

- Use the dataset processor to process dataset dictionaries:

```python
instruction = "<your-instruction>"
dataset_dicts = [...]  # List of DatasetDict
modified_dataset_dicts = processor.process_dataset_dict(instruction, dataset_dicts)
```

The `process_dataset_dict()` method modifies the input of examples in
the `dataset_dicts` through a mapping on `post_process_example`. It returns a list
of modified `DatasetDict`, where all examples are converted into the desired
conditional generation format.

Feel free to adjust the code and configuration based on your specific
requirements and the structure of your datasets.
