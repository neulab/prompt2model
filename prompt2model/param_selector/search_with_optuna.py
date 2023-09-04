"""This module provides a dummy trainer for testing purposes."""
from __future__ import annotations  # noqa FI58

from typing import Any

import transformers
from optuna.trial import Trial

from datasets import Dataset
from transformers import PreTrainedModel  # noqa
from transformers import PreTrainedTokenizer
from prompt2model.prompt_parser.base import PromptSpec

from prompt2model.utils.config import DEFAULT_HYPERPARAMETERS, DEFAULT_HYPERPARAMETERS_SPACE
from prompt2model.model_trainer import BaseTrainer
from prompt2model.param_selector.base import ParamSelector

# TODO:
# - User tweaking hyperparameter
# - Dynamic initialization of hyperparamter range from task type and complexity
# - Using LLM to suggest hyperparameter


class OptunaParamSelector(ParamSelector):
    """Uses optuna for searching for hyperparameters"""

    def __init__(self, trainer: BaseTrainer):
        """Initialize with train/val datasets and a prompt specification"""
        self.trainer = trainer
    
    @property
    def _example_hyperparameters(self) -> dict[str, Any]:
        """Example hyperparameters (for testing only)."""
        return DEFAULT_HYPERPARAMETERS
    
    @property
    def _example_hyperparameter_space(self) -> dict[str, Any]:
        return DEFAULT_HYPERPARAMETERS_SPACE

    def select_from_hyperparameters(
        self,
        training_sets: list[Dataset],
        validation: Dataset,
        hyperparameters: dict[str, Any],
    ) -> transformers.Trainer:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            training_sets (list[Dataset]): One or more training datasets for the trainer.
            validation (Dataset): A dataset for computing validation metrics.
            hyperparameters (dict[str, list[Any]]): A dictionary of hyperparameter choices.

        Supported keys for the hyperparameters and their specs. Left side we have the key and
        have the expected type of the value in braces.
            - :key min_num_train_epochs: (int)
            - :key max_num_train_epochs: (int)
            - :key save_strategy: List[str]
            - :evaluation_strategy: List[str], available options: ["epoch", "no"]
            - :per_device_train_batch_size: List[int], available options: ["epoch", "steps", "no"],
            - :min_weight_decay: float
            - :max_weight_decay: float
            - :min_learning_rate: float
            - :max_learning_rate: float
        Here is a example of hyperparameters:
        ```python
        hyperparameters = {
            "min_num_train_epochs": 5,
            "max_num_train_epochs": 10,
            "save_strategy": ["epoch", "steps", "no"],
            "evaluation_strategy": ["epoch", "no"],
            "per_device_train_batch_size": [4, 8, 16, 32],
            "min_weight_decay": 1e-5,
            "max_weight_decay": 1e-1,
            "min_learning_rate": 1e-5,
            "max_learning_rate", 1e-1,
        }
        ```
        Returns:
            Returns the transformers.Trainer class with the optimal hyperparameters
        """
        # TODO:
        # - Find the industry standards for default values spec
        # - More asserts for other keys. Example checking the min or max values

        if "save_strategy" in hyperparameters:
            save_strategy_is_valid = any(
                strategy in ["epoch", "steps", "no"]
                for strategy in hyperparameters["save_strategy"]
            )
            assert (
                save_strategy_is_valid
            ), "save strategy should have either of these values: ['epoch', 'steps', 'no']"

        if "evaluation_strategy" in hyperparameters:
            evaluation_strategy_is_valid = any(
                strategy in ["epoch", "no"]
                for strategy in hyperparameters["evaluation_strategy"]
            )
            assert (
                evaluation_strategy_is_valid
            ), "Evaluation strategy should have either of these values ['epoch', 'no']"

        def hp_space(trial: Trial):
            return {
                "num_train_epochs": trial.suggest_int(
                    "num_train_epochs",
                    hyperparameters.get("min_train_epochs", 5),
                    hyperparameters.get("max_train_epochs", 10),
                ),
                "save_strategy": trial.suggest_categorical(
                    "save_strategy",
                    hyperparameters.get("save_strategy", ["epoch", "steps", "no"]),
                ),
                "evaluation_strategy": trial.suggest_categorical(
                    "evaluation_strategy",
                    hyperparameters.get("evaluation_strategy", ["epoch", "no"]),
                ),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size",
                    hyperparameters.get("per_device_train_batch_size", [4, 8, 16, 32]),
                ),
                "weight_decay": trial.suggest_loguniform(
                    "weight_decay",
                    hyperparameters.get("min_weight_decay", 1e-5),
                    hyperparameters.get("max_weight_decay", 1e-1),
                ),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate",
                    hyperparameters.get("min_learning_rate", 1e-5),
                    hyperparameters.get("max_learning_rate", 1e-2),
                ),
            }

        # prepare the training args
        # we are assuming here that the user will also provide these args with the additional
        # args for range. Or we can provide an another argument of search_args (dict) that will
        # tell the user to provide the arguments for doing hyperparamerter search

        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir=hyperparameters.get(
                "output_dir", DEFAULT_HYPERPARAMETERS.get("output_dir")
            ),
            logging_steps=hyperparameters.get(
                "logging_steps", DEFAULT_HYPERPARAMETERS.get("logging_steps")
            ),
            save_strategy=hyperparameters.get(
                "save_strategy", DEFAULT_HYPERPARAMETERS.get("save_strategy")
            ),
            num_train_epochs=hyperparameters.get(
                "num_train_epochs", DEFAULT_HYPERPARAMETERS.get("num_train_epochs")
            ),
            per_device_train_batch_size=hyperparameters.get(
                "per_device_train_batch_size",
                DEFAULT_HYPERPARAMETERS.get("per_device_train_batch_size"),
            ),
            warmup_steps=hyperparameters.get(
                "warmup_steps", DEFAULT_HYPERPARAMETERS.get("warmup_steps")
            ),
            weight_decay=hyperparameters.get(
                "weight_decay", DEFAULT_HYPERPARAMETERS.get("weight_decay")
            ),
            logging_dir=hyperparameters.get(
                "logging_dir", DEFAULT_HYPERPARAMETERS.get("logging_dir")
            ),
            learning_rate=hyperparameters.get(
                "learning_rate", DEFAULT_HYPERPARAMETERS.get("learning_rate")
            ),
            predict_with_generate=True,
        )
        # training args here
        trainer = self.trainer.trainer
        trainer.train_dataset = training_sets
        trainer.eval_dataset = validation
        trainer.args = training_args

        best_run = trainer.hyperparameter_search(
            n_trials=5,  # FIXME: Discussion needed, where to put this arg and visibility for the user
            direction="maximize",
            hp_space=hp_space,
        )

        for k, v in best_run.hyperparameters.items():
            setattr(training_args, k, v)

        return trainer

    def select_from_spec(self, training_sets: list[Dataset], validation: Dataset, prompt_spec: PromptSpec) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        return super().select_from_spec(training_sets, validation, prompt_spec)