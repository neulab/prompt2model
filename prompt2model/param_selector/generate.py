"""This module provides a dummy trainer for testing purposes."""
from __future__ import annotations  # noqa FI58

from typing import Any

import optuna
from optuna.trial import Trial

import datasets
from transformers import PreTrainedModel  # noqa
from transformers import PreTrainedTokenizer

from prompt2model.model_trainer import BaseTrainer

# TODO: 
# - User tweaking hyperparameter
# - Dynamic initialization of hyperparamter range from task type and complexity
# - Using LLM to suggest hyperparameter

class AutomamatedParamSelector:
    def __init__(self, strategy, **kwargs) -> None:
        self.strategy = strategy

    @classmethod
    def search_hp_space_using_optuna(self, trial: Trial) -> dict:
        # right now let's assign the hyperparameters by default
        # after this we need to figure out how user can change some of them 
        # or tweak them 

        # FIXME: per_device_train_batch_size is something depends on cpu or gpu
        
        return {
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
            "save_strategy": trial.suggest_categorical("save_strategy", ["epoch", "steps", "no"]),
            "evaluation_strategy": trial.suggest_categorical("evaluation_strategy", ["epoch", "no"]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        }