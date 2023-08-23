# Model Trainer

## Overview

- `BaseTrainer`: A base class for model training.
- `GenerationModelTrainer`: A specialized class for training T5-type
(encoder-decoder) and GPT-type (decoder-only) models.
- `ValidationCallback`: A class for performing validation during
training.

## Getting Started

### Import Modules

```python
from prompt2model.model_trainer.generate import GenerationModelTrainer
```

### Initialize the Trainer

```python
pretrained_model_name = "..."  # Replace with a HuggingFace pretrained model name.
has_encoder = <True/False>
# Set to True if the model has an encoder, otherwise False.
executor_batch_size = <int>  # Set the batch size for model validation.
tokenizer_max_length = <int>  # Set the maximum length for the tokenizer.
sequence_max_length = <int>  # Set the maximum sequence length.
trainer = GenerationModelTrainer(
    pretrained_model_name,
    has_encoder,
    executor_batch_size,
    tokenizer_max_length,
    sequence_max_length
)
# For more details, refer to the docstring of GenerationModelTrainer.
```

### Prepare Dataset and Hyperparameters

```python
training_datasets = []  # A list of training datasets.
validation_datasets = []  # A list of validation datasets.
hyperparameter_choices = {...}  # A dictionary of hyperparameters.
# For more details, refer to the doc string of GenerationModelTrainer.train_model.
```

### Train the Model

```python
trained_model, trained_tokenizer = trainer.train_model(
    hyperparameter_choices,
    training_datasets,
    validation_datasets
)
```
