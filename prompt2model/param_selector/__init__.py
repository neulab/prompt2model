"""Import model selector classes."""
from prompt2model.param_selector.base import ParamSelector
from prompt2model.param_selector.mock import MockParamSelector

__all__ = ("MockParamSelector", "ParamSelector")
