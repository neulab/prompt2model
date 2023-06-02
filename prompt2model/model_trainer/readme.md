# BaseTrainer

The `BaseTrainer` is an abstract class that serves as the base for trainers in the prompt2model library. It provides a common interface and defines the necessary methods for training models with a fixed set of hyperparameters.

To create a trainer using the `BaseTrainer`, you need to implement the following methods:

- `train_model(training_datasets, hyperparameter_choices)`: Trains a model with the given hyperparameters and returns the trained model and tokenizer.

The `BaseTrainer` class can be subclassed to implement custom trainers based on different model architectures or training strategies.

To see an example of how to use `BaseTrainer` and its subclasses, you can refer to the unit tests in the [model_trainer_test.py](../../tests/model_trainer_test.py) file.

# GenerationModelTrainer

The `GenerationModelTrainer` is a concrete implementation of the `BaseTrainer` class specifically designed for training generation models. It supports both encoder-decoder models (T5-type) and decoder-only models (GPT-type).

## Usage

1. Import the necessary modules:

   ```python
   from prompt2model.model_trainer.generate import GenerationModelTrainer
   ```

2. Initialize an instance of the `GenerationModelTrainer` with the desired pre-trained model and specify whether the model has an encoder or not:

   ```python
   trainer = GenerationModelTrainer(pretrained_model_name, has_encoder)
   ```

   - `pretrained_model_name`: The name of the pre-trained model from the Hugging Face model hub.
   - `has_encoder`: A boolean value indicating whether the model has an encoder. Set it to `True` for encoder-decoder models (T5-type) and `False` for decoder-only models (GPT-type).

3. Prepare the training datasets:

   - Create a list of training datasets or load datasets from files.

4. Train the model using the `train_model()` method:

   ```python
   trained_model, trained_tokenizer = trainer.train_model(training_datasets, hyperparameter_choices)
   ```

   - `training_datasets`: A list of training datasets. Each dataset should contain the necessary input and output columns.
   - `hyperparameter_choices`: A dictionary specifying the hyperparameters for training, such as output directory, number of training epochs, batch size, etc.

5. Access the trained model and tokenizer:

   The `train_model()` method returns the trained model and tokenizer, which can be used for inference or further fine-tuning.

Please refer to the documentation and examples provided by the `GenerationModelTrainer` class for detailed usage information.

Ensure you have the required dependencies and resources (pre-trained models, training datasets) set up before using the `GenerationModelTrainer`.
