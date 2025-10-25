import logging
import sys
from typing import Optional
from pathlib import Path


class LoggerManager:
    """Centralized logger management for the multi-agent system"""

    _loggers = {}
    _configured = False

    @classmethod
    def setup_logging(
        cls,
        level: Optional[str] = None,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
    ):
        """Setup global logging configuration"""
        if cls._configured:
            return

        # Import here to avoid circular imports
        from configs import Config

        # Use config defaults if not provided
        level = level or Config.LOG_LEVEL
        log_file = log_file or Config.LOG_FILE
        format_string = format_string or Config.LOG_FORMAT

        # Configure root logger
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            handlers=handlers,
            force=True  # Override any existing configuration
        )

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if not cls._configured:
            cls.setup_logging()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def set_level(cls, level: str):
        """Set logging level for all loggers"""
        log_level = getattr(logging, level.upper())
        for logger in cls._loggers.values():
            logger.setLevel(log_level)


# class BaseLoggable:
#     """Base class that provides logging functionality to any class"""

#     def __init__(self, logger_name: Optional[str] = None):
#         if logger_name is None:
#             logger_name = self.__class__.__name__
#         self.logger = LoggerManager.get_logger(logger_name)

#     def log_debug(self, message: str, **kwargs):
#         """Log debug message"""
#         self.logger.debug(message, extra=kwargs)

#     def log_info(self, message: str, **kwargs):
#         """Log info message"""
#         self.logger.info(message, extra=kwargs)

#     def log_warning(self, message: str, **kwargs):
#         """Log warning message"""
#         self.logger.warning(message, extra=kwargs)

#     def log_error(self, message: str, **kwargs):
#         """Log error message"""
#         self.logger.error(message, extra=kwargs)

#     def log_critical(self, message: str, **kwargs):
#         """Log critical message"""
#         self.logger.critical(message, extra=kwargs)


# Convenience functions for modules that don't inherit from BaseLoggable
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return LoggerManager.get_logger(name)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    LoggerManager.setup_logging(level, log_file)

def set_log_level(level: str):
    """Set global log level"""
    LoggerManager.set_level(level)