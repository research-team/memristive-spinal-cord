import logging
import sys


class LoggingManager:

    def __init__(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(name)s:%(message)s',
            stream=sys.stdout
        )

