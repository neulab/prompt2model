# ModelExecutor Usage

## ModelExecutor

The `ModelExecutor` is an abstract base class that serves as the interface for
executing models and making predictions. It provides a common interface for
different types of models and defines the necessary methods for model execution.

To create a model executor, you need to subclass the `ModelExecutor` and
implement the following methods:

- `make_prediction()`: Evaluates the model on a test set and returns a list of
model outputs, one for each element in the test set.
- `make_single_prediction(model_input)`: Makes a prediction on a single example
and returns a single model output.

The `ModelExecutor` class can be subclassed to implement specific model
execution logic based on different types of models and their associated
tokenizers.

To see an example of how to use `ModelExecutor` and its subclasses, you can
refer to the unit tests in the
[model_executor_test.py](../../tests/model_executor_test.py) file.

## ModelOutput

The `ModelOutput` data class represents the output of a model for a single
example. It contains the following attributes:

- `prediction`: The prediction made by the model.
- `confidence`: A confidence
value in the prediction, or `None` if confidence is unavailable.
- `auxiliary_info`: Any other auxiliary information provided by the model.

The `ModelOutput` class is used to encapsulate the prediction, confidence, and
auxiliary information returned by the model.

## GenerationModelExecutor

The `GenerationModelExecutor` is a concrete implementation of the
`ModelExecutor` that is specifically designed for generative models, including
T5-type and GPT-type models. It extends the `ModelExecutor` class and provides
the implementation for executing generation models.

The `GenerationModelExecutor` class includes the following methods:

- `make_prediction(single_model_input=None)`: Evaluates a T5-type or GPT-type
model on a test set or a single model input. If `single_model_input` is `None`,
the model executor will make predictions on the test set. If
`single_model_input` is provided, it will make a prediction on that single
input.
- `make_single_prediction(model_input)`: Makes a prediction on a single
example.

The `GenerationModelExecutor` class utilizes the model and tokenizer provided
during initialization to make predictions. It handles batch processing of inputs
and returns a list of `ModelOutput` objects representing the model's
predictions.

Please refer to the specific subclasses of `GenerationModelExecutor` for
detailed implementation and usage instructions based on different types of
generative models.

Feel free to subclass the `ModelExecutor` and `GenerationModelExecutor` to
implement your own model execution logic for different types of models.
