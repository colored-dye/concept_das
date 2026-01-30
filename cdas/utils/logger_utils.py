import logging
import sys

LOGGER_FMT_WITH_RANK = "%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s"
LOGGER_FMT_NO_RANK = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"


def logger_setup(logger: logging.Logger, level: int | str = logging.WARNING, rank: int = None):
    """Setup default logger."""
    logger.setLevel(level)

    if rank is None:
        formatter = logging.Formatter(
            fmt=LOGGER_FMT_NO_RANK,
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
    else:
        formatter = logging.Formatter(
            fmt=LOGGER_FMT_WITH_RANK.format(rank=rank),
            datefmt='%Y-%m-%d:%H:%M:%S',
        )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    return [console_handler]
