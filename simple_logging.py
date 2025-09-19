"""
Simple logging configuration for Docker containers
"""
import logging
import os
from datetime import datetime

def setup_simple_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup simple logging that works in Docker"""
    
    # Create log directory
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory {log_dir}: {e}")
        log_dir = "/tmp"  # Fallback to temp directory
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # Try to add file handler
    try:
        log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)
        print(f"Logging to file: {log_file}")
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Simple logging setup completed")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)
