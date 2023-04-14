"""An interface for trainers."""

from abc import ABC, abstractmethod
from typing import Any

import datasets
import transformers
from prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class Trainer(ABC):
    """Train a model with a fixed set of hyperparameters.

    TO IMPLEMENT IN SUBCLASSES:
    def __init__(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
        prompt_spec: PromptSpec,
    ):
        '''
        Initialize trainer with training dataset(s), hyperparameters,
        and a prompt specification.
        '''
        self.training_datasets = training_datasets
        self.hyperparameter_choices = hyperparameter_choices
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        '''Set up Weights & Biases logging.'''
        self.wandb = None
        raise NotImplementedError
    """

    @abstractmethod
    def train_model(self) -> transformers.PreTrainedModel:
        """Train a model with the given hyperparameters and return it.""" ""


class BaseTrainer(Trainer):
    """This dummy trainer does not actually train anything."""

    def __init__(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
        prompt_spec: PromptSpec,
    ):
        """Initialize a dummy BERT-based trainer.

        Args:
            training_datasets: A list of training datasets.
            hyperparameter_choices: A dictionary of hyperparameter choices.
            prompt_spec: A prompt specification.

        """
        self.training_datasets = training_datasets
        self.hyperparameter_choices = hyperparameter_choices
        self.wandb = None
        self.prompt_spec = prompt_spec

    def set_up_weights_and_biases(self) -> None:
        """Set up Weights & Biases logging."""
        self.wandb = None
        raise NotImplementedError

    def train_model(self) -> transformers.PreTrainedModel:
        """This dummy trainer returns an untrained BERT-base model."""
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
        return model
