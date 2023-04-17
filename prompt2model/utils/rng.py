"""Classes for setting random seeds."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod


class SeedGenerator(ABC):
    """Select a good model from among a set of hyperparameter choices."""

    @abstractmethod
    def get_seed(self) -> int:
        """Generate a random seed."""


class ConstantSeedGenerator(SeedGenerator):
    """A seed generator that always returns the same seed."""

    def __init__(self, seed: int = 2023):
        """Initialize with a constant seed (by default, 2023)."""
        self.seed = seed

    def get_seed(self) -> int:
        """Return a constant random seed."""
        return self.seed


seed_generator = ConstantSeedGenerator()
