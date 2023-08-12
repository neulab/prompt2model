"""Utils for creating formatted logger."""

import logging


def get_formatted_logger(logger_name: str):
    """Create a formatted logger.

    Args:
        logger_name: The name of the logger, usually the name
            of the component that uses the logger.

    Returns:
        A logger object.
    """
    logger = logging.getLogger(logger_name)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
