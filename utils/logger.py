import logging
from logging import Logger


def log_message(message: str, logger: Logger, show_logs: bool = False) -> None:
    if show_logs:
        logger.info(message)


def get_logger(name: str, log_level: int = logging.INFO) -> Logger:
    # Create logger and set logging level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    # Set console handler format
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    # Add console handler to logger
    logger.addHandler(console_handler)
    return logger
