# Model Executor

## Overview

- **ModelExecutor**: An interface for executing predictions across
various models.
- **GenerationModelExecutor**: Tailored for generative models such as
T5 (encoder-decoder) and GPT (autoregressive). It supports multiple
generation strategies.
- **ModelOutput**: Denotes the output from the model for a given
example, consolidating the prediction and any auxiliary information.

## Getting Started

- **Import the Necessary Modules**:

```python
from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
```

- **Prepare the Input Data**:

```python
input_dataset = ... # A dataset containing input examples.
input_example = ... # A singular input in string format.
```

- **Initialize the ModelExecutor**:

```python
model = ... # A HuggingFace model instance.
tokenizer = ... # A corresponding HuggingFace tokenizer.
model_executor = GenerationModelExecutor(model, tokenizer)
```

- **Generate Predictions**:

For multiple inputs:

```python
outputs = model_executor.make_prediction(
  test_set=input_dataset, # A dataset object.
  input_column="..."
  # The input column is the name of the column containing the input in the input_dataset.
)
```

For a single input:

```python
test_input = "..."
output = model_executor.make_single_prediction(test_input)
```

- **Choose a Generation Strategy**:

Specify the desired decoding strategy. For more details, see the
 document string `GenerationModelExecutor.generate`.

```python
hyperparameters = {"generate_strategy": "beam", "num_beams": 4}
model_output = model_executor.make_single_prediction(test_input, hyperparameters)
```
