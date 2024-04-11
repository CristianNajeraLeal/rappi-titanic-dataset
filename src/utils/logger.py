"""
This module provides logging utilities to configure custom loggers for applications.

Functions:
    setup_logger(name, level='INFO'): Configures and returns a logger with a
        specified name and log level.
"""


import logging


def setup_logger(name, level='INFO'):
    """
    Create and configure a logger with a specified name and log level.
    :param name: The name of the logger to create and configure.
                    Typically, the __name__ of the module.
    :param level: The log level for the logger. Defaults to 'INFO'. Other levels include
                    'DEBUG', 'WARNING', 'ERROR', and 'CRITICAL'.
    :return: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
