import sys
import logging


logger = logging.getLogger("global_logger")
logger.setLevel(logging.DEBUG)

print_handler = logging.StreamHandler(sys.stderr)
print_handler.setLevel(logging.DEBUG)
print_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

file_handler = logging.FileHandler("excution.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(print_handler)
logger.addHandler(file_handler)

logger.debug('debug message')
logger.info('info message')
logger.warning('warning message')
# logger.error('error message')
# logger.critical('critical message')

