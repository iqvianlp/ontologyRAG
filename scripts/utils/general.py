import logging
import random

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s : %(name)s : %(levelname)s > %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def init_logger(name=None, level=logging.INFO):
    """
    Initialize and return a logger with a specific name and level
    :param name: logger name (if None or empty string, set to 'LOGGER random_num')
    :param level: log level
    :return: logger object
    """

    # Sanitize logger name, if needed
    if not isinstance(name, str) or len(name.strip()) == 0:
        name = f'LOGGER {random.randint(0, 1000000)}'
        logger.warning(f'Name not defined, when creating a new logger, set to: {name}')

    # Set up logger
    lg = logging.getLogger(name)
    hnd = logging.StreamHandler()
    hnd.setFormatter(logging.Formatter("%(asctime)s : %(name)s : %(levelname)s > %(message)s"))
    lg.addHandler(hnd)
    lg.setLevel(level)

    return lg
