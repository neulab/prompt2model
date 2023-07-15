# ModelTrainer Usage

## BaseTrainer

The `BaseTrainer` is an abstract base class in the `prompt2model` library. It
provides a standard interface and defines the necessary methods for training
models with a fixed set of hyperparameters.

To create a custom trainer using the `BaseTrainer`, you need to subclass it and
implement the following method:

- `train_model(training_datasets, hyperparameter_choices)`: This method trains a
model with the given hyperparameters and returns the trained model and
tokenizer.

The `BaseTrainer` class can be subclassed to implement custom trainers based on
different model architectures or training strategies.

## GenerationModelTrainer

The `GenerationModelTrainer` is a concrete implementation of the `BaseTrainer`
class specifically designed for training conditional generation models.
It supports encoder-decoder (T5-type) and autoregressive (GPT-type) models.

## Usage

- Import the necessary modules:

```python
from prompt2model.model_trainer.generate import GenerationModelTrainer
```

- Initialize an instance of the `GenerationModelTrainer`:

```python
trainer = GenerationModelTrainer(pretrained_model_name, has_encoder, model_max_length)
```

Where:

- `pretrained_model_name`: The name of the pre-trained model from the Hugging
Face model hub.
- `has_encoder`: A boolean value indicating whether the model
has an encoder. Set it to `True` for encoder-decoder models (T5-type) and
`False` for decoder-only models (GPT-type).
- `model_max_length` (optional):
This parameter allows the model to truncate long sequences. It should be
noted that truncation may leeds to unexpected training result, so make sure
your model_max_length is enough.

- Prepare the training datasets:

Create a list of training datasets or load datasets from files. Each dataset
should contain the necessary model_input and model_output columns.

```python
training_datasets = [
datasets.Dataset.from_dict(
{
"model_input": [...],
"model_output": [...],
}
),
datasets.Dataset.from_dict(
{
"model_input": [...],
"model_output": [...],
}
),
]
```

- Train the model using the `train_model()` method:

```python
trained_model, trained_tokenizer = trainer.train_model(training_datasets, hyperparameter_choices)
```

Where:

- `training_datasets`: A list of training datasets.
- `hyperparameter_choices`:
A dictionary specifying the hyperparameters for training, such as output
directory, number of training epochs, batch size, etc.

The `train_model()` method returns the trained model and tokenizer, which can be
used for inference or further fine-tuning.

Please refer to the documentation and examples provided by the
`GenerationModelTrainer` class for detailed usage information.

Ensure you have the required dependencies and resources (pre-trained models,
training datasets) set up before using the `GenerationModelTrainer`.

Feel free adjust the code and configuration based on your specific
training requirements.
