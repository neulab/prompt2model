"""This module provides automatic hyperparameter selection using Optuna."""

from __future__ import annotations  # noqa FI58

import os
from pathlib import Path
from typing import Any, Optional

import optuna
import transformers
from datasets import Dataset, concatenate_datasets
from optuna.trial import Trial
from transformers import PreTrainedModel  # noqa
from transformers import PreTrainedTokenizer

from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.param_selector.base import ParamSelector
from prompt2model.utils.config import DEFAULT_HYPERPARAMETERS_SPACE


class OptunaParamSelector(ParamSelector):
    """Uses Optuna for searching for hyperparameters."""

    def __init__(self, trainer: GenerationModelTrainer, n_trials: int):
        """Initializes a new instance of OptunaParamSelector.

        Args:
            trainer (BaseTrainer): trainer object from GenerationModelTrainer
            n_trials (int): The maximum number of parameter configurations to evaluate
                during conducting hyperparameter search.
        """
        self.generation_model_trainer = trainer
        self.n_trials = n_trials

    def optimize_hyperparameters(
        self,
        training_datasets: list[Dataset],
        validation: Dataset,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Select a model among a set of hyperparameters (given or inferred).

        Args:
            training_datasets (list[Dataset]): One or more training datasets
                to use for training models.
            validation_sets (Dataset): A dataset for computing validation metrics.
            hyperparameter_space (Optional[dict[str, Any]], optional): The set
                of possible values of hyperparaneters values required for doing
                optimal hyperparameter search. Defaults to None.

        Returns:
            Returns a dict which contains the best hyperparameters.
        """
        supported_hp_space_keys = set(DEFAULT_HYPERPARAMETERS_SPACE.keys())
        if hyperparameters is not None:
            assert set(hyperparameters.keys()).issubset(
                supported_hp_space_keys
            ), f"Only support {supported_hp_space_keys} as training parameters."
        hyperparameter_space = self._build_hp_space(hyperparameters)

        concatenated_training_dataset = concatenate_datasets(training_datasets)
        train_dataset = self.generation_model_trainer.tokenize_dataset(
            concatenated_training_dataset
        )

        if isinstance(validation, list):
            validation = concatenate_datasets(validation)
            validation = self.generation_model_trainer.tokenize_dataset(validation)

        def objective(trial: Trial) -> float:
            model = self.generation_model_trainer.model
            training_args = transformers.TrainingArguments(
                output_dir="./checkpoint",
                learning_rate=trial.suggest_loguniform(
                    "learning_rate",
                    low=hyperparameter_space["min_learning_rate"],
                    high=hyperparameter_space["max_learning_rate"],
                ),
                weight_decay=trial.suggest_loguniform(
                    "weight_decay",
                    low=hyperparameter_space["min_weight_decay"],
                    high=hyperparameter_space["max_weight_decay"],
                ),
                num_train_epochs=trial.suggest_int(
                    "num_train_epochs",
                    low=hyperparameter_space["min_num_train_epochs"],
                    high=hyperparameter_space["max_num_train_epochs"],
                ),
            )
            objective_trainer = transformers.Trainer(
                model=model,
                args=training_args,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer=self.generation_model_trainer.tokenizer
                ),
                train_dataset=train_dataset,
                eval_dataset=validation,
            )

            _ = objective_trainer.train()
            optimization_targets = objective_trainer.evaluate()
            return optimization_targets["eval_loss"]

        study = optuna.create_study(
            study_name="automatic_hyperparameter_search", direction="minimize"
        )

        study.optimize(func=objective, n_trials=self.n_trials, gc_after_trial=True)
        best_hyperparameters = {
            "learning_rate": float(study.best_params["learning_rate"]),
            "weight_decay": float(study.best_params["weight_decay"]),
            "num_train_epochs": int(study.best_params["num_train_epochs"]),
        }
        return best_hyperparameters

    def select_from_hyperparameters(
        self,
        training_datasets: list[Dataset],
        validation: Dataset,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Select a model among a set of hyperparameters (given or inferred). # noqa D410

        Args:
            training_datasets: One or more training datasets for the trainer.
            validation: A dataset for computing validation metrics.
            hyperparameters: A dictionary of hyperparameter choices.

        If no hyperparameter_space is specified, then the default hyperparameter_space
        will be choosen. Here is the example of how the space looks like:
        hyperparameter_space = {
            "min_num_train_epochs": 5,
            "max_num_train_epochs": 10,
            "save_strategy": ["epoch", "steps", "no"],
            "evaluation_strategy": ["epoch", "no"],
            "per_device_train_batch_size": [4, 8, 16, 32],
            "min_weight_decay": 1e-5,
            "max_weight_decay": 1e-1,
            "min_learning_rate": 1e-5,
            "max_learning_rate": 1e-1,
        }
        Return:
            A model and tokenizer (with hyperparameters from given range).
        """
        model = self.generation_model_trainer.model
        tokenizer = self.generation_model_trainer.tokenizer
        best_model_path = Path("result/trained_model")
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)

        best_hyperparameters = self.optimize_hyperparameters(
            training_datasets=training_datasets,
            validation=validation,
            hyperparameters=hyperparameters,
        )
        final_hyperparameters = {
            "output_dir": "./best_model_checkpoint",
            **best_hyperparameters,
        }

        model, tokenizer = self.generation_model_trainer.train_model(
            hyperparameter_choices=final_hyperparameters,
            training_datasets=training_datasets,
        )

        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        return model, tokenizer

    def _build_hp_space(
        self, hyperparameter_space: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        if hyperparameter_space is None:
            return DEFAULT_HYPERPARAMETERS_SPACE
        hp_space = {}

        default_keys = list(DEFAULT_HYPERPARAMETERS_SPACE.keys())
        for key in list(hyperparameter_space.keys()):
            if key not in default_keys:
                print(
                    f"Key {key} is not present in DEFAULT_HYPERPARAMETERS_SPACE. Hence, it will be ignored.",  # noqa E501
                    "However, you can expose the key to the Trainer by adding it to DEFAULT_HYPERPARAMETERS_SPACE.",  # noqa E501
                )

        for key, default_value in DEFAULT_HYPERPARAMETERS_SPACE.items():
            hp_space[key] = hyperparameter_space.get(key, default_value)
        return hp_space
