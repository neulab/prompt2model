"""Utility functions for testing."""
from contextlib import contextmanager


@contextmanager
def temp_setattr(obj, attr, value):
    """Temporarily set an attribute on an object."""
    original = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if original is not None:
            setattr(obj, attr, original)
        else:
            delattr(obj, attr)
