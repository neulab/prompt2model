# ModelExecutor Usage

## ModelExecutor

The `ModelExecutor` is an abstract base class that is the interface for
executing models to generate predictions. It provides a common interface for
both the encoder-decoder and autoregressive models with the necessary
methods for model execution.

To create a model executor, you need to subclass the `ModelExecutor` and
implement the following methods:

- `make_prediction()`: Evaluates the model on a test set and returns a list of
model outputs, one for each element in the test set.
- `make_single_prediction()`: Makes a prediction on a single example
and returns a single model output. This is usually used in `DemoCreator`.

The `ModelExecutor` class can be subclassed to implement the specific model
execution logic based on different types of models and their associated
tokenizers.

## ModelOutput

The `ModelOutput` data class represents the output of a model for a single
example. It contains the following attributes:

- `prediction`: The prediction made by the model.
- `auxiliary_info`: Any other auxiliary information provided by the model.

The `ModelOutput` class is used to encapsulate the prediction and
auxiliary information returned by the model.

## GenerationModelExecutor

The `GenerationModelExecutor` is a concrete implementation of the
`ModelExecutor` that is specifically designed for generative models,
including T5-type (encoder-decoder) and GPT-type (autoregressive)
models. It extends the `ModelExecutor` class and provides
the implementation for executing generation models with five generate
strategies: "top_k sampling", "top_p sampling", "greedy search",
"beam search" and "intersect between top_k and top_p sampling".

The `GenerationModelExecutor` class includes the following methods:

- `make_prediction()`: Evaluates a T5-type or GPT-type
model on a test set or a single model input. If `single_model_input` is `None`,
the model executor will make predictions on the test set. Otherwise,
it will make a prediction on that single input.

- `make_single_prediction(model_input)`: Makes a prediction on a single
input.

The `GenerationModelExecutor` class utilizes the model and tokenizer provided
during initialization to make predictions. It handles batch processing of inputs
and returns a list of `ModelOutput` objects representing the model's
predictions.

## Usage

- Import the necessary modules:

```python
from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
```

- Prepare the input dataset or example for the executor:

```python
input_dataset = Dataset.from_dict(
  {
    "model_input": [
      "Translate French to English: cher",
      "Translate French to English: Bonjour",
      "Translate French to English: raisin",
    ]
  }
)

input_example = "Translate Chinese to English: 你好"
```

- Create the model executor:

```python
t5_model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")
t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-tiny")
model_executor = GenerationModelExecutor(t5_model, t5_tokenizer, test_dataset, "model_input")
```
