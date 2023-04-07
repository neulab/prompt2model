"""An interface for trainers.
"""

from abc import abstractmethod
from typing import Any, Protocol

import datasets
import transformers

from prompt_parser import PromptSpec

# Input:
#    1) training dataset (datasets.Dataset)
#    2) Dictionary consisting of hyperparameter values to use
#       (e.g. base model, optimizer, LR, etc)
#
# Output:
#    transformers.PreTrainedModel


class Trainer(Protocol):
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
    """Train a model with a fixed set of hyperparameters."""

    def __init__(
        self,
        training_datasets: list[datasets.Dataset],
        hyperparameter_choices: dict[str, Any],
        prompt_spec: PromptSpec,
    ):
        """
        Initialize trainer with training dataset(s), hyperparameters,
        and a prompt specification.
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
        """Train a model with the given hyperparameters and return it.""" ""
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
        return model
