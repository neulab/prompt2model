"""Import ModelTrainer classes."""
from prompt2model.model_trainer.base import ModelTrainer
from prompt2model.model_trainer.mock import MockTrainer

__all__ = ("MockTrainer", "ModelTrainer")
