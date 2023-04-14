"""Import Trainer classes."""
from prompt2model.trainer.base import Trainer
from prompt2model.trainer.mock import MockTrainer

__all__ = ("MockTrainer", "Trainer")
