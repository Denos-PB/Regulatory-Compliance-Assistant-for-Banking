import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return root

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "rag_pipeline.log")
    logger = root
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger