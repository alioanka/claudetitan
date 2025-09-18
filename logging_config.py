"""
Advanced Logging Configuration for Trading Bot
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Get current timestamp for log file naming
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handlers with rotation
    main_log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
    error_log_file = os.path.join(log_dir, f"trading_bot_errors_{timestamp}.log")
    debug_log_file = os.path.join(log_dir, f"trading_bot_debug_{timestamp}.log")
    
    # Main log file (all levels) - with fallback to console if permission denied
    try:
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(detailed_formatter)
    except PermissionError:
        print(f"Warning: Cannot write to log file {main_log_file}, falling back to console logging")
        main_handler = logging.StreamHandler()
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(detailed_formatter)
    
    # Error log file (ERROR and CRITICAL only) - with fallback to console if permission denied
    try:
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
    except PermissionError:
        print(f"Warning: Cannot write to error log file {error_log_file}, falling back to console logging")
        error_handler = logging.StreamHandler()
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
    
    # Debug log file (all levels including DEBUG) - with fallback to console if permission denied
    try:
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
    except PermissionError:
        print(f"Warning: Cannot write to debug log file {debug_log_file}, falling back to console logging")
        debug_handler = logging.StreamHandler()
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Add debug handler only if log level is DEBUG
    if log_level.upper() == "DEBUG":
        root_logger.addHandler(debug_handler)
    
    # Configure specific loggers
    configure_module_loggers(log_dir, timestamp)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Trading Bot Enhanced - Logging System Initialized")
    logger.info(f"Log Level: {log_level.upper()}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Main Log: {main_log_file}")
    logger.info(f"Error Log: {error_log_file}")
    if log_level.upper() == "DEBUG":
        logger.info(f"Debug Log: {debug_log_file}")
    logger.info("=" * 60)

def configure_module_loggers(log_dir: str, timestamp: str):
    """Configure specific loggers for different modules"""
    
    # Trading strategies logger - with fallback to console if permission denied
    strategy_logger = logging.getLogger('trading_strategies')
    try:
        strategy_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"strategies_{timestamp}.log"),
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        strategy_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        strategy_logger.addHandler(strategy_handler)
    except PermissionError:
        print(f"Warning: Cannot write to strategies log file, falling back to console logging")
        strategy_handler = logging.StreamHandler()
        strategy_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        strategy_logger.addHandler(strategy_handler)
    
    # Risk management logger - with fallback to console if permission denied
    risk_logger = logging.getLogger('risk_management')
    try:
        risk_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"risk_management_{timestamp}.log"),
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        risk_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        risk_logger.addHandler(risk_handler)
    except PermissionError:
        print(f"Warning: Cannot write to risk management log file, falling back to console logging")
        risk_handler = logging.StreamHandler()
        risk_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        risk_logger.addHandler(risk_handler)
    
    # Market data logger - with fallback to console if permission denied
    market_logger = logging.getLogger('market_data')
    try:
        market_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"market_data_{timestamp}.log"),
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        market_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        market_logger.addHandler(market_handler)
    except PermissionError:
        print(f"Warning: Cannot write to market data log file, falling back to console logging")
        market_handler = logging.StreamHandler()
        market_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        market_logger.addHandler(market_handler)
    
    # ML module logger - with fallback to console if permission denied
    ml_logger = logging.getLogger('ml_module')
    try:
        ml_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"ml_module_{timestamp}.log"),
            maxBytes=10*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        ml_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        ml_logger.addHandler(ml_handler)
    except PermissionError:
        print(f"Warning: Cannot write to ML module log file, falling back to console logging")
        ml_handler = logging.StreamHandler()
        ml_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        ml_logger.addHandler(ml_handler)
    
    # Dashboard logger - with fallback to console if permission denied
    dashboard_logger = logging.getLogger('dashboard')
    try:
        dashboard_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"dashboard_{timestamp}.log"),
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        dashboard_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        dashboard_logger.addHandler(dashboard_handler)
    except PermissionError:
        print(f"Warning: Cannot write to dashboard log file, falling back to console logging")
        dashboard_handler = logging.StreamHandler()
        dashboard_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        dashboard_logger.addHandler(dashboard_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)

# Log file cleanup function
def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """Clean up log files older than specified days"""
    import time
    from pathlib import Path
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    for log_file in log_path.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Error deleting {log_file}: {e}")

if __name__ == "__main__":
    # Test logging configuration
    setup_logging("DEBUG")
    logger = get_logger(__name__)
    
    logger.info("Testing info log")
    logger.warning("Testing warning log")
    logger.error("Testing error log")
    logger.debug("Testing debug log")
    
    print("Logging test completed. Check the logs/ directory for output files.")
