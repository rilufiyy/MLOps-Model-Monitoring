import logging
import sys
from pathlib import Path
from src.utils.config import config

class logger:
    _loggers = {}
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        if name in logger._loggers:
            return logger._loggers[name]
        
        log = logging.getLogger(name)
        log.setLevel(getattr(logging, config.log_level))
        
        # Remove existing handlers
        log.handlers = []
        
        # Create formatters
        formatter = logging.Formatter(config.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        
        # File handler
        config.create_directories()
        log_file = config.logs_dir / config.log_file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        
        # Prevent propagation
        log.propagate = False
        
        logger._loggers[name] = log
        return log