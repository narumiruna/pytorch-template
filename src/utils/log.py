import logging
from datetime import datetime
from pathlib import Path

FORMAT = '[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'

LEVEL_MAP = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOSET': logging.NOTSET,
}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # set level
    logger.setLevel(logging.DEBUG)

    # formatter
    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # file handler
    today = datetime.now().strftime('%y%m%d')
    log_file = Path(f'logs/{today}/{name}.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # prevent twice log
    logger.propagate = False
    return logger
