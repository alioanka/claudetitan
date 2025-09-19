"""
Simple logging configuration for Docker containers
"""
import logging
import os
from datetime import datetime

def setup_simple_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup simple logging that works in Docker"""
    
    # Try multiple log directory options
    log_dirs_to_try = [
        log_dir,
        "/app/logs",
        "/tmp/logs",
        "/tmp"
    ]
    
    working_log_dir = None
    for test_dir in log_dirs_to_try:
        try:
            os.makedirs(test_dir, exist_ok=True)
            # Test write access
            test_file = os.path.join(test_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            working_log_dir = test_dir
            print(f"Successfully created and tested log directory: {test_dir}")
            break
        except Exception as e:
            print(f"Could not use log directory {test_dir}: {e}")
            continue
    
    if not working_log_dir:
        print("Warning: No writable log directory found, using console only")
        working_log_dir = None
    
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
    
    # Try to add file handler if we have a working directory
    if working_log_dir:
        try:
            log_file = os.path.join(working_log_dir, f"trading_bot_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            root_logger.addHandler(file_handler)
            print(f"✅ Logging to file: {log_file}")
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Simple logging setup completed")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {working_log_dir or 'console only'}")
    logger.info("=" * 50)

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)
