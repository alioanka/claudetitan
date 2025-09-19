#!/usr/bin/env python3
"""
Test script to verify logging is working correctly
"""
import os
import logging
from logging_config import setup_logging

def test_logging():
    """Test logging configuration"""
    print("Testing logging configuration...")
    
    # Get LOG_DIR from environment
    LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
    print(f"LOG_DIR: {LOG_DIR}")
    
    # Setup logging
    setup_logging("INFO", LOG_DIR)
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("✅ TEST-LINE: Logging is working correctly!")
    logger.warning("⚠️  TEST-WARNING: This is a test warning")
    logger.error("❌ TEST-ERROR: This is a test error")
    
    # Check if files were created
    try:
        files = os.listdir(LOG_DIR)
        print(f"Files in {LOG_DIR}: {files}")
        
        # Try to read the main log file
        main_log = os.path.join(LOG_DIR, f"trading_bot_{logging.datetime.now().strftime('%Y%m%d')}.log")
        if os.path.exists(main_log):
            with open(main_log, 'r') as f:
                lines = f.readlines()
                print(f"Last 3 lines from {main_log}:")
                for line in lines[-3:]:
                    print(f"  {line.strip()}")
        else:
            print(f"Main log file not found: {main_log}")
            
    except Exception as e:
        print(f"Error checking log directory: {e}")

if __name__ == "__main__":
    test_logging()
