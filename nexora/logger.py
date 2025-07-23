import os
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, ERROR
from colorlog import ColoredFormatter
import traceback as tb

# Create a logger instance
logger = getLogger(__name__)

# Define color styles for different log levels
log_colors = {
    'DEBUG': 'green',
    'INFO': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

# Setup ColoredFormatter with color styles
color_formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors=log_colors
)

# Create a stream handler with color formatter
console = StreamHandler()
console.setFormatter(color_formatter)

# Add the handler to the logger
logger.addHandler(console)
logger.setLevel(INFO)

# File logging setup
log_path = os.path.expanduser('~/.config/nexora/logs')
log_file = os.path.join(log_path, 'nexora.log')

os.makedirs(log_path, exist_ok=True)  # Ensure log directory exists

file_handler = FileHandler(log_file)
file_formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.setLevel(INFO)


def log_traceback(message):
    """
    Logs the traceback of an exception along with a custom error message.
    
    Args:
        message (str): The custom error message to log.
    """
    exc_info = tb.format_exc()
    logger.error("%s\nTraceback:\n%s", message, exc_info)

# Shortcut to use the custom traceback directly via logger
logger.traceback = log_traceback
