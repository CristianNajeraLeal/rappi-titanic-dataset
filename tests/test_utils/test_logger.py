import logging
from src.utils.logger import setup_logger


def test_setup_logger_default_info_level():
    logger_name = 'test_logger'
    logger = setup_logger(logger_name)

    assert logger.level == logging.INFO, "Logger level should be set to INFO by default"
    assert len(logger.handlers) == 1, "Logger should have exactly one handler"

    # Check if the handler is a StreamHandler with the correct level and formatter
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler), "Handler should be a StreamHandler"
    assert handler.level == logging.INFO, "Handler level should be set to INFO"
    assert isinstance(handler.formatter, logging.Formatter), "Handler should have a formatter"
    assert handler.formatter._fmt == '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "Formatter pattern does not match"


def test_setup_logger_custom_level():
    logger_name = 'test_logger_custom'
    custom_level = 'DEBUG'
    logger = setup_logger(logger_name, level=custom_level)

    assert logger.level == logging.DEBUG, "Logger level should be settable to DEBUG"
    # Ensuring only one handler is added even if called again
    logger_duplicate = setup_logger(logger_name, level=custom_level)
    assert len(logger_duplicate.handlers) == 1, "Logger should not add duplicate handlers"


def test_setup_logger_prevent_duplicate_handlers():
    logger_name = 'test_logger_no_duplicates'
    logger = setup_logger(logger_name)
    logger_duplicate = setup_logger(logger_name)

    assert len(logger.handlers) == 1, "Logger should not have duplicate handlers after calling setup_logger twice"
