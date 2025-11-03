import logging
import sys
from datetime import datetime

COLORS = {
    "RESET": "\033[0m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
    "DEBUG": "\033[94m",
    "TIME": "\033[90m",
    "SERVICE": "\033[96m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Colored level
        level_color = COLORS.get(record.levelname, COLORS["RESET"])
        level = f"{level_color}{record.levelname:<8}{COLORS['RESET']}"

        # Service = file:function (no line number)
        service = f"{COLORS['SERVICE']}{record.pathname.split('/')[-1]}:{record.funcName}{COLORS['RESET']}"

        # Time
        time_str = f"{COLORS['TIME']}{ts}{COLORS['RESET']}"

        # Message
        message = record.getMessage()

        return f"{level} | {time_str} | {service:<25} | {message}"


def configure_logging() -> logging.Logger:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)

    # Quiet noisy deps
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Patch uvicorn
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers = [console_handler]
        uv_logger.propagate = True
        uv_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    return logger
