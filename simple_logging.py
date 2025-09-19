"""
Simple logging configuration for Docker containers
"""
import logging
import os
from datetime import datetime

def setup_simple_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup simple logging that works in Docker + captures uvicorn/fastapi."""

    import logging
    import logging.handlers
    import os
    from datetime import datetime
    from pathlib import Path

    # Allow override via env
    env_log_dir = os.environ.get("LOG_DIR")
    if env_log_dir:
        log_dir = env_log_dir

    # Try multiple locations (first writable wins)
    log_dirs_to_try = [log_dir, "/app/logs", "/var/log/claudetitan", "/tmp/logs", "/tmp"]
    working_log_dir = None
    for candidate in log_dirs_to_try:
        try:
            Path(candidate).mkdir(parents=True, exist_ok=True)
            test_file = Path(candidate) / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            working_log_dir = candidate
            break
        except Exception:
            continue

    # Build handlers
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    main_log_file = None
    handlers = []

    # Reset root handlers to avoid duplicates when reloading
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    if working_log_dir:
        main_log_file = os.path.join(working_log_dir, f"bot_{timestamp}.log")
        error_log_file = os.path.join(working_log_dir, f"errors_{timestamp}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        ))
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        ))
        handlers.extend([file_handler, error_handler])

    # Always keep a console handler for docker logs
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handlers.append(console)

    # Force clean basicConfig with our handlers
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), handlers=handlers, force=True)

    # Capture uvicorn/fastapi too
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        for h in handlers:
            lg.addHandler(h)
        lg.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        lg.propagate = False

    # Test
    logger = logging.getLogger(__name__)
    logger.info("✅ Simple logging setup completed")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {working_log_dir or 'console only'}")
    if working_log_dir:
        logger.info(f"Main log file: {main_log_file}")
    logger.info("=" * 50)

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)
